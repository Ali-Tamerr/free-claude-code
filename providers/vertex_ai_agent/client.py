"""Vertex AI Agent Platform provider (OpenAI-compatible chat via API key query)."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, AsyncIterator

import httpx

from core.anthropic import ContentType, HeuristicToolParser, SSEBuilder, ThinkTagParser
from core.anthropic.sse import map_stop_reason
from core.trace import provider_chat_body_snapshot, trace_event
from providers.base import BaseProvider, ProviderConfig
from providers.error_mapping import map_error, user_visible_message_for_mapped_provider_error
from providers.exceptions import AuthenticationError
from providers.model_listing import ProviderModelInfo, model_infos_from_ids
from providers.rate_limit import GlobalRateLimiter
from providers.openai_compat import _iter_heuristic_tool_use_sse
from providers.vertex_ai.client import build_vertex_base_url
from providers.vertex_ai.request import build_request_body

_AGENT_MODEL_FALLBACK: frozenset[str] = frozenset(
    (
        "google/gemini-3.5-flash",
        "google/gemini-3.1-flash-lite",
        "google/gemini-live-2.5-flash-native-audio",
        "google/gemini-2.5-pro",
        "google/gemini-2.5-flash",
        "google/gemini-2.5-flash-lite",
        "google/gemini-2.5-flash-image",
        "google/gemini-2.0-flash-001",
        "google/gemini-2.0-flash-lite-001",
        "google/veo-3.1-generate-001",
        "google/veo-3.1-fast-generate-001",
        "google/veo-3.0-generate-001",
        "google/veo-3.0-fast-generate-001",
        "google/veo-2.0-generate-001",
        "google/gemini-embedding-001",
        "google/text-embedding-005",
        "google/text-embedding-004",
        "google/text-multilingual-embedding-002",
        "google/multimodalembedding@001",
    )
)


class VertexAIAgentProvider(BaseProvider):
    """Vertex AI Agent Platform using OpenAI-compatible chat with API key query."""

    def __init__(
        self,
        config: ProviderConfig,
        *,
        project_id: str = "",
        location: str = "",
    ):
        super().__init__(config)
        base_url = (config.base_url or "").strip()
        if not base_url and project_id.strip() and location.strip():
            base_url = build_vertex_base_url(project_id, location)
        if not base_url:
            raise AuthenticationError(
                "Vertex AI base URL not set. Set VERTEX_AI_BASE_URL or "
                "VERTEX_AI_PROJECT_ID + VERTEX_AI_LOCATION."
            )
        if not config.api_key.strip():
            raise AuthenticationError(
                "VERTEX_AI_API_KEY is required for Agent Platform requests."
            )
        self._provider_name = "VERTEX_AI_AGENT"
        self._api_key = config.api_key
        self._base_url = base_url.rstrip("/")
        self._global_rate_limiter = GlobalRateLimiter.get_scoped_instance(
            "vertex_ai_agent",
            rate_limit=config.rate_limit,
            rate_window=config.rate_window,
            max_concurrency=config.max_concurrency,
        )
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            proxy=config.proxy or None,
            timeout=httpx.Timeout(
                config.http_read_timeout,
                connect=config.http_connect_timeout,
                read=config.http_read_timeout,
                write=config.http_write_timeout,
            ),
        )

    async def cleanup(self) -> None:
        await self._client.aclose()

    async def list_model_ids(self) -> frozenset[str]:
        return frozenset(_AGENT_MODEL_FALLBACK)

    async def list_model_infos(self) -> frozenset[ProviderModelInfo]:
        return model_infos_from_ids(_AGENT_MODEL_FALLBACK)

    def _build_request_body(
        self, request: Any, thinking_enabled: bool | None = None
    ) -> dict:
        return build_request_body(
            request,
            thinking_enabled=self._is_thinking_enabled(request, thinking_enabled),
        )

    async def _send_stream_request(self, body: dict) -> httpx.Response:
        request = self._client.build_request(
            "POST",
            "/chat/completions",
            json=body,
            params={"key": self._api_key},
            headers={"Content-Type": "application/json"},
        )
        return await self._client.send(request, stream=True)

    async def _iter_openai_stream(self, response: httpx.Response) -> AsyncIterator[Any]:
        try:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line:
                    continue
                if not line.startswith("data:"):
                    continue
                payload = line.split(":", 1)[1].strip()
                if payload == "[DONE]":
                    break
                try:
                    data = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                yield _chunk_from_payload(data)
        finally:
            await response.aclose()

    async def stream_response(
        self,
        request: Any,
        input_tokens: int = 0,
        *,
        request_id: str | None = None,
        thinking_enabled: bool | None = None,
    ) -> AsyncIterator[str]:
        message_id = f"msg_{request_id or 'vertex'}"
        sse = SSEBuilder(
            message_id,
            request.model,
            input_tokens,
            log_raw_events=self._config.log_raw_sse_events,
        )

        body = self._build_request_body(request, thinking_enabled=thinking_enabled)
        thinking_enabled = self._is_thinking_enabled(request, thinking_enabled)
        trace_event(
            stage="provider",
            event="provider.request.sent",
            source="provider",
            provider=self._provider_name,
            gateway_model=request.model,
            downstream_model=body.get("model"),
            message_count=len(body.get("messages", [])),
            tool_count=len(body.get("tools", [])),
            body=provider_chat_body_snapshot(body),
        )

        yield sse.message_start()

        think_parser = ThinkTagParser()
        heuristic_parser = HeuristicToolParser()
        finish_reason = None
        usage_info = None

        async with self._global_rate_limiter.concurrency_slot():
            try:
                response = await self._send_stream_request(body)
                async for chunk in self._iter_openai_stream(response):
                    if getattr(chunk, "usage", None):
                        usage_info = chunk.usage

                    if not chunk.choices:
                        continue
                    choice = chunk.choices[0]
                    delta = choice.delta
                    if choice.finish_reason:
                        finish_reason = choice.finish_reason

                    reasoning = getattr(delta, "reasoning_content", None)
                    if thinking_enabled and reasoning:
                        for event in sse.ensure_thinking_block():
                            yield event
                        yield sse.emit_thinking_delta(reasoning)

                    if delta.content:
                        for part in think_parser.feed(delta.content):
                            if part.type == ContentType.THINKING:
                                if not thinking_enabled:
                                    continue
                                for event in sse.ensure_thinking_block():
                                    yield event
                                yield sse.emit_thinking_delta(part.content)
                            else:
                                filtered_text, detected_tools = heuristic_parser.feed(
                                    part.content
                                )
                                if filtered_text:
                                    for event in sse.ensure_text_block():
                                        yield event
                                    yield sse.emit_text_delta(filtered_text)
                                for tool_use in detected_tools:
                                    for event in _iter_heuristic_tool_use_sse(
                                        sse, tool_use
                                    ):
                                        yield event

                    if delta.tool_calls:
                        for event in sse.close_content_blocks():
                            yield event
                        for tc in delta.tool_calls:
                            for event in _emit_tool_call_delta(sse, tc):
                                yield event

            except Exception as e:
                mapped_e = map_error(e, rate_limiter=self._global_rate_limiter)
                error_message = user_visible_message_for_mapped_provider_error(
                    mapped_e,
                    provider_name=self._provider_name,
                    read_timeout_s=self._config.http_read_timeout,
                )
                for event in sse.close_all_blocks():
                    yield event
                for event in sse.emit_error(error_message):
                    yield event
                yield sse.message_delta("end_turn", 1)
                yield sse.message_stop()
                return

        remaining = think_parser.flush()
        if remaining and remaining.type == ContentType.TEXT:
            for event in sse.ensure_text_block():
                yield event
            yield sse.emit_text_delta(remaining.content)

        for event in sse.close_all_blocks():
            yield event

        output_tokens = (
            getattr(usage_info, "completion_tokens", None)
            if usage_info is not None
            else None
        )
        output_tokens = output_tokens if isinstance(output_tokens, int) else 0
        yield sse.message_delta(map_stop_reason(finish_reason), output_tokens)
        yield sse.message_stop()


def _chunk_from_payload(payload: dict) -> Any:
    usage = payload.get("usage")
    usage_obj = SimpleNamespace(**usage) if isinstance(usage, dict) else None
    choices = []
    for choice in payload.get("choices", []) or []:
        delta = choice.get("delta", {}) or {}
        tool_calls = []
        for tc in delta.get("tool_calls", []) or []:
            fn = tc.get("function", {}) or {}
            tool_calls.append(
                SimpleNamespace(
                    index=tc.get("index", 0),
                    id=tc.get("id"),
                    function=SimpleNamespace(
                        name=fn.get("name"),
                        arguments=fn.get("arguments", ""),
                    ),
                )
            )
        delta_obj = SimpleNamespace(
            content=delta.get("content"),
            tool_calls=tool_calls or None,
            reasoning_content=delta.get("reasoning_content"),
        )
        choices.append(
            SimpleNamespace(
                delta=delta_obj,
                finish_reason=choice.get("finish_reason"),
            )
        )
    return SimpleNamespace(choices=choices, usage=usage_obj)


def _emit_tool_call_delta(sse: SSEBuilder, tc: Any) -> list[str]:
    tool_index = tc.index if tc.index is not None else 0
    tool_id = tc.id or f"tool_{tool_index}"
    name = tc.function.name or "tool"
    args = tc.function.arguments or "{}"
    events: list[str] = []
    state = sse.blocks.tool_states.get(tool_index)
    if state is None or not state.started:
        events.append(sse.start_tool_block(tool_index, tool_id, name))
    events.append(sse.emit_tool_delta(tool_index, args))
    return events

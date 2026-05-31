"""Request builder for Vertex AI Generative Language API (streamGenerateContent)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from loguru import logger

from config.paths import config_dir_path
from core.anthropic import ReasoningReplayMode, build_base_request_body
from core.anthropic.conversion import OpenAIConversionError
from providers.exceptions import InvalidRequestError


def build_request_body(request_data: Any, *, thinking_enabled: bool) -> dict[str, Any]:
    """Build a Generative Language API request body from an Anthropic request."""
    logger.debug(
        "VERTEX_AI_REQUEST: conversion start model={} msgs={}",
        getattr(request_data, "model", "?"),
        len(getattr(request_data, "messages", [])),
    )
    try:
        openai_body = build_base_request_body(
            request_data,
            reasoning_replay=(
                ReasoningReplayMode.THINK_TAGS
                if thinking_enabled
                else ReasoningReplayMode.DISABLED
            ),
        )
    except OpenAIConversionError as exc:
        raise InvalidRequestError(str(exc)) from exc

    system_text, messages = _extract_system(openai_body.get("messages", []))
    contents = _openai_messages_to_contents(messages)

    body: dict[str, Any] = {"contents": contents}
    if system_text:
        body["system_instruction"] = {"parts": [{"text": system_text}]}

    if openai_body.get("tools"):
        body["tools"] = [
            {
                "functionDeclarations": [
                    _convert_function_declaration(tool) for tool in openai_body["tools"]
                ]
            }
        ]

    tool_choice = openai_body.get("tool_choice")
    if tool_choice and (tool_config := _convert_tool_choice(tool_choice)):
        body["toolConfig"] = tool_config

    generation_config = _build_generation_config(openai_body)
    if thinking_enabled:
        thinking = getattr(request_data, "thinking", None)
        budget = None
        if thinking:
            budget = (
                thinking.get("budget_tokens")
                if isinstance(thinking, dict)
                else getattr(thinking, "budget_tokens", None)
            )
        t_config: dict[str, Any] = {}
        if budget is not None:
            t_config["thinkingBudget"] = budget
        else:
            t_config["thinkingBudget"] = 2048
        generation_config["thinkingConfig"] = t_config
    else:
        generation_config["thinkingConfig"] = {"thinkingBudget": 0}

    if generation_config:
        body["generationConfig"] = generation_config

    logger.debug(
        "VERTEX_AI_REQUEST: conversion done model={} msgs={} tools={}",
        openai_body.get("model"),
        len(contents),
        len(openai_body.get("tools", [])),
    )
    return body


def _extract_system(messages: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    system_parts: list[str] = []
    remaining: list[dict[str, Any]] = []
    for msg in messages:
        if msg.get("role") == "system":
            content = msg.get("content")
            if isinstance(content, str) and content.strip():
                system_parts.append(content.strip())
        else:
            remaining.append(msg)
    return "\n\n".join(system_parts), remaining


def _openai_messages_to_contents(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    contents: list[dict[str, Any]] = []
    tool_id_map: dict[str, str] = {}

    for msg in messages:
        role = msg.get("role")
        if role == "tool":
            tool_call_id = str(msg.get("tool_call_id") or "")
            tool_name = tool_id_map.get(tool_call_id, "tool")
            response = _parse_json_or_text(msg.get("content", ""))
            contents.append(
                {
                    "role": "user",
                    "parts": [
                        {
                            "functionResponse": {
                                "name": tool_name,
                                "response": response,
                            }
                        }
                    ],
                }
            )
            continue

        content = msg.get("content", "")
        parts = _content_to_parts(content)

        tool_calls = msg.get("tool_calls")
        if role == "assistant" and tool_calls:
            # Drop empty text parts for assistant turns containing tool calls
            parts = [p for p in parts if p.get("text", "").strip()]
            for tc in tool_calls:
                fn = tc.get("function", {})
                name = fn.get("name") or "tool"
                args = fn.get("arguments", "") or "{}"
                parsed_args = _parse_json_or_text(args)

                # Prepend the associated thought signature if cached to avoid 400 rejection
                sig = lookup_thought_signature(name, parsed_args)
                part_dict: dict[str, Any] = {
                    "functionCall": {
                        "name": name,
                        "args": parsed_args,
                    }
                }
                if sig:
                    part_dict["thoughtSignature"] = sig

                parts.append(part_dict)
                if tc.get("id"):
                    tool_id_map[str(tc["id"])] = name

        if not parts:
            parts = [{"text": ""}]

        contents.append({"role": _map_role(role), "parts": parts})

    return contents


def _content_to_parts(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"text": content}]
    if isinstance(content, list):
        parts: list[dict[str, Any]] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append({"text": str(item.get("text", ""))})
            else:
                raise InvalidRequestError(
                    "Vertex AI Generative Language API only supports text content."
                )
        return parts
    if content is None:
        return []
    return [{"text": str(content)}]


def _sanitize_schema(schema: Any) -> Any:
    if not isinstance(schema, dict):
        return schema
    sanitized = {}
    allowed_keys = {
        "type",
        "properties",
        "required",
        "description",
        "items",
        "enum",
        "nullable",
        "format",
    }
    for k, v in schema.items():
        if k in allowed_keys:
            if k == "properties" and isinstance(v, dict):
                sanitized[k] = {pk: _sanitize_schema(pv) for pk, pv in v.items()}
            elif k == "items" and isinstance(v, dict):
                sanitized[k] = _sanitize_schema(v)
            else:
                sanitized[k] = v
    return sanitized


def _convert_function_declaration(tool: dict[str, Any]) -> dict[str, Any]:
    if tool.get("type") != "function":
        return {
            "name": "tool",
            "description": "",
            "parameters": {"type": "object", "properties": {}},
        }
    fn = tool.get("function", {})
    params = fn.get("parameters") or {"type": "object", "properties": {}}
    return {
        "name": fn.get("name") or "tool",
        "description": fn.get("description") or "",
        "parameters": _sanitize_schema(params),
    }


def _convert_tool_choice(tool_choice: Any) -> dict[str, Any] | None:
    if isinstance(tool_choice, str):
        return {
            "functionCallingConfig": {
                "mode": _tool_choice_mode(tool_choice),
            }
        }
    if isinstance(tool_choice, dict):
        choice_type = tool_choice.get("type")
        if choice_type in {"auto", "none", "required"}:
            return {"functionCallingConfig": {"mode": _tool_choice_mode(choice_type)}}
        if choice_type in {"function", "tool"}:
            name = (tool_choice.get("function", {}) or {}).get(
                "name"
            ) or tool_choice.get("name")
            f_config: dict[str, Any] = {
                "mode": "ANY",
            }
            if name:
                f_config["allowedFunctionNames"] = [str(name)]
            config: dict[str, Any] = {"functionCallingConfig": f_config}
            return config
    return None


def _tool_choice_mode(choice: str) -> str:
    if choice == "none":
        return "NONE"
    if choice == "required":
        return "ANY"
    return "AUTO"


def _build_generation_config(openai_body: dict[str, Any]) -> dict[str, Any]:
    config: dict[str, Any] = {}
    max_tokens = openai_body.get("max_tokens")
    if isinstance(max_tokens, int):
        config["maxOutputTokens"] = max_tokens
    temperature = openai_body.get("temperature")
    if isinstance(temperature, (int, float)):
        config["temperature"] = float(temperature)
    top_p = openai_body.get("top_p")
    if isinstance(top_p, (int, float)):
        config["topP"] = float(top_p)
    stop_sequences = openai_body.get("stop")
    if isinstance(stop_sequences, list) and stop_sequences:
        config["stopSequences"] = stop_sequences
    return config


def _parse_json_or_text(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        raw = value.strip()
        if raw:
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                return {"content": raw}
            if isinstance(parsed, dict):
                return parsed
        return {"content": raw}
    if value is None:
        return {"content": ""}
    return {"content": str(value)}


def _map_role(role: Any) -> str:
    if role == "assistant":
        return "model"
    return "user"


def _get_cache_file_path() -> Path:
    cache_dir = config_dir_path() / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / "vertex_ai_signatures.json"


def _load_cache() -> dict[str, str]:
    path = _get_cache_file_path()
    if not path.exists():
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                return data
    except Exception as e:
        logger.warning("Failed to load vertex thought signature cache: {}", e)
    return {}


def _save_cache(cache: dict[str, str]) -> None:
    path = _get_cache_file_path()
    try:
        temp_path = path.with_suffix(".tmp")
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
        temp_path.replace(path)
    except Exception as e:
        logger.warning("Failed to save vertex thought signature cache: {}", e)


def _make_cache_key(name: str, args: dict[str, Any]) -> str:
    try:
        serialized = json.dumps(args, sort_keys=True)
    except Exception:
        serialized = str(args)
    return f"{name}:{serialized}"


def save_thought_signature(name: str, args: dict[str, Any], signature: str) -> None:
    key = _make_cache_key(name, args)
    cache = _load_cache()
    if key not in cache and len(cache) >= 1000:
        first_key = next(iter(cache))
        cache.pop(first_key, None)
    cache[key] = signature
    _save_cache(cache)


def lookup_thought_signature(name: str, args: dict[str, Any]) -> str | None:
    key = _make_cache_key(name, args)
    cache = _load_cache()
    return cache.get(key)

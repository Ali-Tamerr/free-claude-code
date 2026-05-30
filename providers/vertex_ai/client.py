"""Google Cloud Vertex AI provider (OpenAI-compatible chat completions).

Supports two authentication modes:
1. API key or access token: set ``VERTEX_AI_API_KEY``.
2. Provide ``VERTEX_AI_BASE_URL`` or set ``VERTEX_AI_PROJECT_ID`` and
    ``VERTEX_AI_LOCATION`` to build it.
"""

from __future__ import annotations

from typing import Any

import openai

from providers.base import ProviderConfig
from providers.exceptions import AuthenticationError
from providers.openai_compat import OpenAIChatTransport

from .request import build_request_body

# Vertex AI OpenAI-compatible endpoint template.
_VERTEX_BASE_URL_TEMPLATE = (
    "https://{location}-aiplatform.googleapis.com/v1beta1/"
    "projects/{project_id}/locations/{location}/endpoints/openapi"
)

_VERTEX_MODEL_FALLBACK: frozenset[str] = frozenset(
    (
        # Gemini models (latest stable)
        "gemini-3.5-flash",
        "gemini-3.1-flash-lite",
        "gemini-live-2.5-flash-native-audio",
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
        "gemini-2.5-flash-image",
        "gemini-2.0-flash-001",
        "gemini-2.0-flash-lite-001",
        # Veo models
        "veo-3.1-generate-001",
        "veo-3.1-fast-generate-001",
        "veo-3.0-generate-001",
        "veo-3.0-fast-generate-001",
        "veo-2.0-generate-001",
        # Embeddings models
        "gemini-embedding-001",
        "text-embedding-005",
        "text-embedding-004",
        "text-multilingual-embedding-002",
        "multimodalembedding@001",
    )
)


def build_vertex_base_url(project_id: str, location: str) -> str:
    """Build the Vertex AI OpenAI-compat base URL from project + location."""
    return _VERTEX_BASE_URL_TEMPLATE.format(
        project_id=project_id.strip(),
        location=location.strip(),
    )


class VertexAIProvider(OpenAIChatTransport):
    """Vertex AI using the OpenAI-compatible chat completions endpoint."""

    def __init__(
        self,
        config: ProviderConfig,
        *,
        project_id: str = "",
        location: str = "",
    ):
        # Determine the base URL: explicit config > project+location > error.
        base_url = config.base_url
        if not base_url and project_id.strip() and location.strip():
            base_url = build_vertex_base_url(project_id, location)
        if not base_url:
            raise AuthenticationError(
                "Vertex AI base URL not set. Set VERTEX_AI_BASE_URL or "
                "VERTEX_AI_PROJECT_ID + VERTEX_AI_LOCATION."
            )

        super().__init__(
            config,
            provider_name="VERTEX_AI",
            base_url=base_url,
            api_key=config.api_key,
        )

    def _build_request_body(
        self, request: Any, thinking_enabled: bool | None = None
    ) -> dict:
        return build_request_body(
            request,
            thinking_enabled=self._is_thinking_enabled(request, thinking_enabled),
        )

    async def list_model_ids(self) -> frozenset[str]:
        """Return model ids or fallback list when Vertex AI lacks /models support."""
        try:
            return await super().list_model_ids()
        except openai.NotFoundError:
            return _VERTEX_MODEL_FALLBACK
        except openai.APIError as exc:
            if getattr(exc, "status_code", None) == 404:
                return _VERTEX_MODEL_FALLBACK
            raise

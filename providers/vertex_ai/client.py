"""Google Cloud Vertex AI provider (OpenAI-compatible chat completions).

Supports two authentication modes:
1. API key only: set ``VERTEX_AI_API_KEY``.
2. Project + location: set ``VERTEX_AI_PROJECT_ID`` and ``VERTEX_AI_LOCATION``.
   The base URL is built from the project/location and the API key is used
   as a bearer token (typically a service-account access token).
"""

from __future__ import annotations

from typing import Any

import openai

from providers.base import ProviderConfig
from providers.openai_compat import OpenAIChatTransport

from .request import build_request_body

# Vertex AI OpenAI-compatible endpoint template.
_VERTEX_BASE_URL_TEMPLATE = (
    "https://{location}-aiplatform.googleapis.com/v1beta1/"
    "projects/{project_id}/locations/{location}/endpoints/openapi"
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

        super().__init__(
            config,
            provider_name="VERTEX_AI",
            base_url=base_url or "",
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
        """Return model ids or empty set when Vertex AI lacks /models support."""
        try:
            return await super().list_model_ids()
        except openai.NotFoundError:
            return frozenset()
        except openai.APIError as exc:
            if getattr(exc, "status_code", None) == 404:
                return frozenset()
            raise

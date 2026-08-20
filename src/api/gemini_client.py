"""Gemini client using the supported Google Gen AI SDK and Vertex AI."""
from __future__ import annotations

import json
from typing import Type, TypeVar

from pydantic import BaseModel

from src.config.settings import GCP_LOCATION, GCP_PROJECT_ID, GEMINI_MODEL

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None


StructuredResponse = TypeVar("StructuredResponse", bound=BaseModel)


class GeminiClient:
    """Small Vertex AI client with structured-output support."""

    def __init__(
        self,
        project_id: str = GCP_PROJECT_ID,
        location: str = GCP_LOCATION,
        model: str = GEMINI_MODEL,
    ):
        if genai is None or types is None:
            raise RuntimeError("Install google-genai to enable Gemini RAG explanations.")
        self.model = model
        self.client = genai.Client(
            vertexai=True,
            project=project_id,
            location=location,
            http_options=types.HttpOptions(api_version="v1"),
        )
        self.mock_mode = False

    def generate_structured(
        self,
        system_prompt: str,
        user_message: str,
        response_schema: Type[StructuredResponse],
        max_tokens: int = 1200,
    ) -> StructuredResponse:
        response = self.client.models.generate_content(
            model=self.model,
            contents=user_message,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                max_output_tokens=max_tokens,
                temperature=0.1,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
                response_mime_type="application/json",
                response_schema=response_schema,
            ),
        )
        if getattr(response, "parsed", None) is not None:
            parsed = response.parsed
            return parsed if isinstance(parsed, response_schema) else response_schema.model_validate(parsed)
        return response_schema.model_validate(json.loads(response.text))

    def _call(self, system_prompt: str, user_message: str, max_tokens: int = 2000) -> str:
        response = self.client.models.generate_content(
            model=self.model,
            contents=user_message,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                max_output_tokens=max_tokens,
                temperature=0.1,
            ),
        )
        return response.text

    def call_haiku(self, system_prompt: str, user_message: str, max_tokens: int = 2000) -> str:
        return self._call(system_prompt, user_message, max_tokens)

    def call_sonnet(self, system_prompt: str, user_message: str, max_tokens: int = 2000) -> str:
        return self._call(system_prompt, user_message, max_tokens)


ClaudeClient = GeminiClient

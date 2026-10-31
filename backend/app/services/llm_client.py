from __future__ import annotations

from typing import Protocol

import httpx

from app.config import settings


class LLMError(Exception):
    """Base exception for LLM client failures."""


class MissingAPIKeyError(LLMError):
    pass


class LLMProviderError(LLMError):
    pass


class LLMTimeoutError(LLMError):
    pass


class LLMClient(Protocol):
    async def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        ...


class GeminiClient:
    """Google Gemini client using the REST generateContent API."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
        max_tokens: int | None = None,
    ) -> None:
        self.api_key = (
            settings.gemini_api_key if api_key is None else api_key
        )

        self.base_url = (
            settings.gemini_base_url if base_url is None else base_url
        ).rstrip("/")

        self.model = (
            settings.gemini_model if model is None else model
        )

        self.timeout = (
            settings.llm_timeout_seconds if timeout is None else timeout
        )

        self.max_tokens = (
            settings.llm_max_tokens if max_tokens is None else max_tokens
        )

    async def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> str:

        if not self.api_key or not self.api_key.strip():
            raise MissingAPIKeyError(
                "GEMINI_API_KEY is not configured."
            )

        url = (
            f"{self.base_url}/models/"
            f"{self.model}:generateContent"
        )

        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.api_key,
        }

        payload = {
            "system_instruction": {
                "parts": [
                    {"text": system_prompt}
                ]
            },
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": user_prompt}
                    ],
                }
            ],
            "generationConfig": {
                "temperature": 0,
                "maxOutputTokens": self.max_tokens,
                "responseMimeType": "application/json",
            },
        }

        try:
            async with httpx.AsyncClient(
                timeout=self.timeout
            ) as client:
                response = await client.post(
                    url,
                    headers=headers,
                    json=payload,
                )

                response.raise_for_status()

        except httpx.TimeoutException as exc:
            raise LLMTimeoutError(
                "The Gemini request timed out."
            ) from exc

        except httpx.HTTPStatusError as exc:
            raise LLMProviderError(
                f"Gemini returned HTTP {exc.response.status_code}."
            ) from exc

        except httpx.RequestError as exc:
            raise LLMProviderError(
                "Unable to connect to the Gemini API."
            ) from exc

        try:
            data = response.json()

            candidates = data.get("candidates", [])

            if not candidates:
                raise ValueError("Gemini returned no candidates.")

            content = candidates[0].get("content", {})
            parts = content.get("parts", [])

            text_parts = [
                part["text"]
                for part in parts
                if isinstance(part.get("text"), str)
            ]

            generated_text = "\n".join(text_parts).strip()

            if not generated_text:
                raise ValueError("Gemini returned empty content.")

            return generated_text

        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise LLMProviderError(
                "Gemini returned an invalid response."
            ) from exc
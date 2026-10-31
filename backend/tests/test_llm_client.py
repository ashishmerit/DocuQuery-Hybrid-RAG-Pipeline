import httpx
import pytest

from app.services.llm_client import (
    GeminiClient,
    LLMProviderError,
    LLMTimeoutError,
    MissingAPIKeyError,
)


def test_missing_api_key_is_reported():
    client = GeminiClient(api_key="")

    with pytest.raises(MissingAPIKeyError):
        import asyncio

        asyncio.run(
            client.generate(
                system_prompt="system",
                user_prompt="user",
            )
        )


@pytest.mark.asyncio
async def test_provider_error_is_mapped(monkeypatch):

    class FakeResponse:
        def raise_for_status(self):
            raise httpx.HTTPStatusError(
                "server error",
                request=httpx.Request(
                    "POST",
                    "https://example.com",
                ),
                response=httpx.Response(500),
            )

    class FakeAsyncClient:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def post(self, *args, **kwargs):
            return FakeResponse()

    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        FakeAsyncClient,
    )

    client = GeminiClient(api_key="test-key")

    with pytest.raises(LLMProviderError):
        await client.generate(
            system_prompt="system",
            user_prompt="user",
        )


@pytest.mark.asyncio
async def test_timeout_is_mapped(monkeypatch):

    class FakeAsyncClient:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def post(self, *args, **kwargs):
            raise httpx.TimeoutException("timeout")

    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        FakeAsyncClient,
    )

    client = GeminiClient(api_key="test-key")

    with pytest.raises(LLMTimeoutError):
        await client.generate(
            system_prompt="system",
            user_prompt="user",
        )


@pytest.mark.asyncio
async def test_successful_gemini_response(monkeypatch):

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "text": (
                                        '{"answer":"ok",'
                                        '"citation_ids":["S1"],'
                                        '"insufficient_evidence":false}'
                                    )
                                }
                            ]
                        }
                    }
                ]
            }

    class FakeAsyncClient:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def post(self, url, **kwargs):
            assert ":generateContent" in url
            assert kwargs["headers"]["x-goog-api-key"] == "test-key"

            payload = kwargs["json"]
            assert payload["generationConfig"]["responseMimeType"] == (
                "application/json"
            )

            return FakeResponse()

    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        FakeAsyncClient,
    )

    client = GeminiClient(
        api_key="test-key",
        model="gemini-2.5-flash",
    )

    result = await client.generate(
        system_prompt="system",
        user_prompt="user",
    )

    assert '"answer":"ok"' in result
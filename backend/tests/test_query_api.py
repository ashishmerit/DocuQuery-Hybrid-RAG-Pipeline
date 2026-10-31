import json

import pytest

from app.models import (
    Chunk,
    IngestionResponse,
    RetrievalResult,
)
from app.services.document_store import document_store
from app.services.llm_client import (
    LLMProviderError,
    LLMTimeoutError,
)
from app.services.llm_client import GeminiClient

@pytest.fixture(autouse=True)
def reset_documents():
    document_store.clear()
    yield
    document_store.clear()


def add_sample_document():
    chunk = Chunk(
        document_id="doc-query-test",
        chunk_id="chunk-query-test",
        page_number=1,
        text="DocuQuery uses hybrid retrieval and cross-encoder reranking.",
        start_char=0,
        end_char=60,
    )

    document_store.add(
        IngestionResponse(
            document_id="doc-query-test",
            filename="sample.pdf",
            page_count=1,
            chunk_count=1,
            chunks=[chunk],
        )
    )

    return RetrievalResult(
        chunk=chunk,
        score=0.95,
        rank=1,
        retriever="reranked",
    )


class FakeLLMClient:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error

    async def generate(self, *, system_prompt, user_prompt):
        if self.error:
            raise self.error
        return json.dumps(self.response)


def patch_retrieval(monkeypatch, result):
    from app.api import query

    monkeypatch.setattr(
        query,
        "GeminiClient",
        lambda: FakeLLMClient(...),
    )


def test_query_returns_answer_and_citation(client, monkeypatch):
    from app.api import query

    result = add_sample_document()
    patch_retrieval(monkeypatch, result)

    monkeypatch.setattr(
        query,
        "GeminiClient",
        lambda: FakeLLMClient(...),
    )
    response = client.post(
        "/api/query",
        json={
            "question": "What retrieval approach is used?",
            "document_id": "doc-query-test",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "answered"
    assert payload["citations"][0]["page_number"] == 1
    assert payload["citations"][0]["document_name"] == "sample.pdf"


def test_query_rejects_unknown_document(client):
    response = client.post(
        "/api/query",
        json={
            "question": "What is this document about?",
            "document_id": "missing-document",
        },
    )

    assert response.status_code == 404


def test_query_requires_uploaded_document(client):
    response = client.post(
        "/api/query",
        json={"question": "What is this document about?"},
    )

    assert response.status_code == 400


def test_query_maps_missing_api_key_to_503(client, monkeypatch):
    from app.api import query
    from app.services.llm_client import GeminiClient

    result = add_sample_document()
    patch_retrieval(monkeypatch, result)

    monkeypatch.setattr(
        query,
        "GeminiClient",
        lambda: FakeLLMClient(...),
)

    response = client.post(
        "/api/query",
        json={
            "question": "What retrieval approach is used?",
            "document_id": "doc-query-test",
        },
    )

    assert response.status_code == 503


@pytest.mark.asyncio
async def test_provider_errors_are_mapped_by_api(client, monkeypatch):
    from app.api import query

    result = add_sample_document()
    patch_retrieval(monkeypatch, result)

    monkeypatch.setattr(
        query,
        "GeminiClient",
        lambda: FakeLLMClient(...),
    )

    response = client.post(
        "/api/query",
        json={
            "question": "What retrieval approach is used?",
            "document_id": "doc-query-test",
        },
    )

    assert response.status_code == 502


def test_timeout_is_mapped_by_api(client, monkeypatch):
    from app.api import query

    result = add_sample_document()
    patch_retrieval(monkeypatch, result)

    monkeypatch.setattr(
        query,
        "GeminiClient",
        lambda: FakeLLMClient(...),
    )

    response = client.post(
        "/api/query",
        json={
            "question": "What retrieval approach is used?",
            "document_id": "doc-query-test",
        },
    )

    assert response.status_code == 504
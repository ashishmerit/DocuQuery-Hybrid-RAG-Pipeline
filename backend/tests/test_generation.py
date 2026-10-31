import json

import pytest

from app.models import (
    Chunk,
    IngestionResponse,
    RetrievalResult,
)
from app.services.document_store import document_store
from app.services.generation import AnswerGenerationService


class FakeLLMClient:
    def __init__(self, response):
        self.response = response
        self.calls = 0

    async def generate(self, *, system_prompt, user_prompt):
        self.calls += 1
        return self.response


@pytest.fixture(autouse=True)
def clear_store():
    document_store.clear()
    yield
    document_store.clear()


def make_result():
    chunk = Chunk(
        document_id="doc-1",
        chunk_id="chunk-1",
        page_number=2,
        text="RAG combines retrieval with language model generation.",
        start_char=0,
        end_char=55,
    )

    document_store.add(
        IngestionResponse(
            document_id="doc-1",
            filename="rag-guide.pdf",
            page_count=3,
            chunk_count=1,
            chunks=[chunk],
        )
    )

    return RetrievalResult(
        chunk=chunk,
        score=0.9,
        rank=1,
        retriever="reranked",
    )


@pytest.mark.asyncio
async def test_supported_question_returns_answer_and_citation():
    result = make_result()

    client = FakeLLMClient(json.dumps({
        "answer": "RAG combines retrieval with language generation.",
        "citation_ids": ["S1"],
        "insufficient_evidence": False,
    }))

    service = AnswerGenerationService(client)
    response = await service.generate("What is RAG?", [result])

    assert response.status == "answered"
    assert len(response.citations) == 1
    assert response.citations[0].citation_id == "S1"
    assert response.citations[0].document_name == "rag-guide.pdf"
    assert response.citations[0].page_number == 2
    assert response.citations[0].chunk_id == "chunk-1"


@pytest.mark.asyncio
async def test_unsupported_question_returns_insufficient_evidence():
    result = make_result()

    client = FakeLLMClient(json.dumps({
        "answer": "The document does not provide this information.",
        "citation_ids": [],
        "insufficient_evidence": True,
    }))

    response = await AnswerGenerationService(client).generate(
        "What is the author's phone number?",
        [result],
    )

    assert response.status == "insufficient_evidence"
    assert response.citations == []


@pytest.mark.asyncio
async def test_no_retrieved_evidence_skips_llm_call():
    client = FakeLLMClient("{}")

    response = await AnswerGenerationService(client).generate(
        "What is RAG?",
        [],
    )

    assert response.status == "insufficient_evidence"
    assert client.calls == 0


@pytest.mark.asyncio
async def test_unknown_citation_ids_are_not_returned():
    result = make_result()

    client = FakeLLMClient(json.dumps({
        "answer": "RAG uses retrieval.",
        "citation_ids": ["S1", "S99"],
        "insufficient_evidence": False,
    }))

    response = await AnswerGenerationService(client).generate(
        "What is RAG?",
        [result],
    )

    assert response.status == "answered"
    assert [item.citation_id for item in response.citations] == ["S1"]


@pytest.mark.asyncio
async def test_only_unknown_citations_causes_abstention():
    result = make_result()

    client = FakeLLMClient(json.dumps({
        "answer": "Unsupported claim.",
        "citation_ids": ["S99"],
        "insufficient_evidence": False,
    }))

    response = await AnswerGenerationService(client).generate(
        "What is RAG?",
        [result],
    )

    assert response.status == "insufficient_evidence"
    assert response.citations == []


@pytest.mark.asyncio
async def test_malformed_json_raises_provider_error():
    from app.services.llm_client import LLMProviderError

    result = make_result()
    client = FakeLLMClient("not-json")

    with pytest.raises(LLMProviderError):
        await AnswerGenerationService(client).generate(
            "What is RAG?",
            [result],
        )
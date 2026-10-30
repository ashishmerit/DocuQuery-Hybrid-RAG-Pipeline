import pytest

from app.models import Chunk, RetrievalResult
from app.services.reranker import CrossEncoderReranker


class FakeCrossEncoder:
    def predict(self, pairs):
        # Deliberately rank the second candidate above the first.
        return [0.1, 0.9, 0.4][:len(pairs)]


def make_result(chunk_id, document_id="doc-1"):
    text = f"Passage {chunk_id}"
    chunk = Chunk(
        document_id=document_id,
        chunk_id=chunk_id,
        page_number=2,
        text=text,
        start_char=0,
        end_char=len(text),
    )
    return RetrievalResult(
        chunk=chunk,
        score=0.01,
        rank=1,
        retriever="hybrid",
    )


def test_reranker_reorders_candidates():
    reranker = CrossEncoderReranker(model=FakeCrossEncoder())

    results = reranker.rerank(
        "test query",
        [
            make_result("A"),
            make_result("B"),
            make_result("C"),
        ],
    )

    assert [item.chunk.chunk_id for item in results] == [
        "B", "C", "A"
    ]
    assert [item.rank for item in results] == [1, 2, 3]
    assert all(item.retriever == "reranked" for item in results)


def test_reranker_preserves_metadata():
    reranker = CrossEncoderReranker(model=FakeCrossEncoder())

    results = reranker.rerank(
        "test query",
        [make_result("A")],
    )

    assert results[0].chunk.document_id == "doc-1"
    assert results[0].chunk.page_number == 2
    assert results[0].chunk.text == "Passage A"


def test_reranker_filters_document_id():
    reranker = CrossEncoderReranker(model=FakeCrossEncoder())

    results = reranker.rerank(
        "test query",
        [
            make_result("A", "doc-1"),
            make_result("B", "doc-2"),
        ],
        document_id="doc-1",
    )

    assert len(results) == 1
    assert results[0].chunk.document_id == "doc-1"


def test_reranker_empty_query():
    reranker = CrossEncoderReranker(model=FakeCrossEncoder())

    assert reranker.rerank(" ", [make_result("A")]) == []


def test_reranker_empty_candidates():
    reranker = CrossEncoderReranker(model=FakeCrossEncoder())

    assert reranker.rerank("query", []) == []


def test_reranker_rejects_invalid_top_k():
    reranker = CrossEncoderReranker(model=FakeCrossEncoder())

    with pytest.raises(ValueError):
        reranker.rerank(
            "query",
            [make_result("A")],
            top_k=0,
        )
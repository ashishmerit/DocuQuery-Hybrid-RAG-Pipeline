import numpy as np
import pytest

from app.models import Chunk
from app.services.bm25_retrieval import (
    BM25Retriever,
    tokenize,
)
from app.services.dense_retrieval import DenseRetriever


class FakeEmbedder:
    """Deterministic test embedder; no model download required."""

    def encode(self, texts):
        vectors = []

        for text in texts:
            normalized = text.casefold()

            if "apple" in normalized or "fruit" in normalized:
                vectors.append([1.0, 0.0])
            elif "car" in normalized or "vehicle" in normalized:
                vectors.append([0.0, 1.0])
            else:
                vectors.append([0.5, 0.5])

        return np.asarray(vectors, dtype=np.float32)


def make_chunk(
    chunk_id,
    text,
    document_id="doc-1",
    page_number=1,
):
    return Chunk(
        document_id=document_id,
        chunk_id=chunk_id,
        page_number=page_number,
        text=text,
        start_char=0,
        end_char=len(text),
    )


def test_dense_retrieval_ranks_nearest_chunk_and_preserves_metadata():
    chunks = [
        make_chunk(
            "fruit-1",
            "An apple is a fruit.",
            page_number=2,
        ),
        make_chunk(
            "vehicle-1",
            "A car is a vehicle.",
            document_id="doc-2",
            page_number=4,
        ),
    ]

    retriever = DenseRetriever(
        chunks,
        embedder=FakeEmbedder(),
    )

    results = retriever.retrieve("apple fruit", top_k=1)

    assert len(results) == 1
    assert results[0].chunk.chunk_id == "fruit-1"
    assert results[0].chunk.document_id == "doc-1"
    assert results[0].chunk.page_number == 2
    assert results[0].rank == 1
    assert results[0].retriever == "dense"
    assert results[0].score == pytest.approx(1.0)


def test_dense_document_filter_is_applied_before_top_k():
    chunks = [
        make_chunk(
            "fruit-a",
            "apple fruit",
            document_id="doc-a",
        ),
        make_chunk(
            "fruit-b",
            "apple fruit",
            document_id="doc-b",
        ),
    ]

    retriever = DenseRetriever(
        chunks,
        embedder=FakeEmbedder(),
    )

    results = retriever.retrieve(
        "apple",
        top_k=1,
        document_id="doc-b",
    )

    assert len(results) == 1
    assert results[0].chunk.document_id == "doc-b"


def test_dense_empty_index_returns_no_results():
    retriever = DenseRetriever([], embedder=FakeEmbedder())

    assert retriever.retrieve("apple") == []


def test_bm25_retrieves_keyword_match_and_preserves_metadata():
    chunks = [
        make_chunk(
            "api-1",
            "FastAPI validates HTTP requests.",
        ),
        make_chunk(
            "bm25-1",
            "BM25 ranks keyword documents using term frequency.",
        ),
    ]

    retriever = BM25Retriever(chunks)

    results = retriever.retrieve(
        "BM25 term frequency",
        top_k=2,
    )

    assert results
    assert results[0].chunk.chunk_id == "bm25-1"
    assert results[0].chunk.page_number == 1
    assert results[0].rank == 1
    assert results[0].retriever == "bm25"
    assert isinstance(results[0].score, float)

    assert tokenize(
        "BM25, Term-frequency!"
    ) == ["bm25", "term-frequency"]


def test_bm25_returns_empty_when_no_query_terms_match():
    retriever = BM25Retriever([
        make_chunk(
            "api-1",
            "FastAPI validates HTTP requests.",
        )
    ])

    assert retriever.retrieve("unrelated zebras") == []


def test_bm25_document_filter_is_respected():
    chunks = [
        make_chunk(
            "bm25-a",
            "BM25 keyword ranking",
            document_id="doc-a",
        ),
        make_chunk(
            "bm25-b",
            "BM25 keyword ranking",
            document_id="doc-b",
        ),
    ]

    results = BM25Retriever(chunks).retrieve(
        "BM25 keyword",
        document_id="doc-b",
    )

    assert results
    assert all(
        result.chunk.document_id == "doc-b"
        for result in results
    )


@pytest.mark.parametrize("retriever_kind", ["dense", "bm25"])
def test_retrievers_reject_non_positive_top_k(retriever_kind):
    chunks = [make_chunk("fruit-1", "apple fruit")]

    if retriever_kind == "dense":
        retriever = DenseRetriever(
            chunks,
            embedder=FakeEmbedder(),
        )
    else:
        retriever = BM25Retriever(chunks)

    with pytest.raises(ValueError, match="top_k"):
        retriever.retrieve("apple", top_k=0)
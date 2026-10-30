import pytest

from app.models import Chunk, RetrievalResult
from app.services.rrf import reciprocal_rank_fusion


def result(chunk_id, rank, document_id="doc-1"):
    chunk = Chunk(
        document_id=document_id,
        chunk_id=chunk_id,
        page_number=rank,
        text=f"Text for {chunk_id}",
        start_char=0,
        end_char=len(f"Text for {chunk_id}"),
    )
    return RetrievalResult(
        chunk=chunk,
        score=1.0 / rank,
        rank=rank,
        retriever="dense",
    )


def test_rrf_combines_rankings_and_deduplicates():
    dense = [result("A", 1), result("B", 2)]
    bm25 = [result("B", 1), result("C", 2)]

    fused = reciprocal_rank_fusion(
        [dense, bm25],
        k=60,
    )

    ids = [item.chunk.chunk_id for item in fused]

    assert len(ids) == 3
    assert ids[0] == "B"
    assert len(set(ids)) == len(ids)
    assert all(item.retriever == "hybrid" for item in fused)


def test_rrf_preserves_source_metadata():
    dense = [result("A", 1)]

    fused = reciprocal_rank_fusion([dense])

    assert fused[0].chunk.document_id == "doc-1"
    assert fused[0].chunk.page_number == 1
    assert fused[0].chunk.text == "Text for A"
    assert fused[0].chunk.start_char == 0


def test_rrf_filters_document_id():
    rankings = [
        [
            result("A", 1, "doc-1"),
            result("B", 2, "doc-2"),
        ]
    ]

    fused = reciprocal_rank_fusion(
        rankings,
        document_id="doc-1",
    )

    assert [item.chunk.chunk_id for item in fused] == ["A"]


def test_rrf_top_k():
    fused = reciprocal_rank_fusion(
        [[result("A", 1), result("B", 2), result("C", 3)]],
        top_k=2,
    )

    assert len(fused) == 2


@pytest.mark.parametrize("value", [0, -1])
def test_rrf_rejects_invalid_k(value):
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([], k=value)
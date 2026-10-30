from app.models import Chunk, RetrievalResult
from app.services.hybrid_retrieval import HybridRetriever


def make_result(chunk_id, rank, retriever, document_id="doc-1"):
    text = f"Passage {chunk_id}"
    chunk = Chunk(
        document_id=document_id,
        chunk_id=chunk_id,
        page_number=rank,
        text=text,
        start_char=0,
        end_char=len(text),
    )
    return RetrievalResult(
        chunk=chunk,
        score=1.0 / rank,
        rank=rank,
        retriever=retriever,
    )


class FakeRetriever:
    def __init__(self, results):
        self.results = results
        self.calls = []

    def retrieve(self, query, top_k=None, document_id=None):
        self.calls.append((query, top_k, document_id))

        results = self.results
        if document_id:
            results = [
                result for result in results
                if result.chunk.document_id == document_id
            ]

        return results[:top_k]


class FakeReranker:
    def rerank(
        self,
        query,
        candidates,
        top_k=None,
        document_id=None,
    ):
        candidates = list(reversed(candidates))

        if document_id:
            candidates = [
                item for item in candidates
                if item.chunk.document_id == document_id
            ]

        candidates = candidates[:top_k]

        return [
            RetrievalResult(
                chunk=item.chunk,
                score=float(len(candidates) - index),
                rank=index + 1,
                retriever="reranked",
            )
            for index, item in enumerate(candidates)
        ]


def build_pipeline():
    dense = FakeRetriever([
        make_result("A", 1, "dense"),
        make_result("B", 2, "dense"),
        make_result("X", 3, "dense", "doc-2"),
    ])

    bm25 = FakeRetriever([
        make_result("B", 1, "bm25"),
        make_result("C", 2, "bm25"),
        make_result("X", 3, "bm25", "doc-2"),
    ])

    pipeline = HybridRetriever(
        dense,
        bm25,
        FakeReranker(),
        rrf_k=60,
        candidate_k=5,
    )

    return pipeline, dense, bm25


def test_hybrid_combines_both_retrievers():
    pipeline, dense, bm25 = build_pipeline()

    results = pipeline.retrieve_hybrid("query", top_k=5)

    assert len(results) == 4
    assert dense.calls
    assert bm25.calls
    assert all(item.retriever == "hybrid" for item in results)


def test_hybrid_document_filtering():
    pipeline, _, _ = build_pipeline()

    results = pipeline.retrieve_hybrid(
        "query",
        top_k=5,
        document_id="doc-1",
    )

    assert results
    assert all(
        item.chunk.document_id == "doc-1"
        for item in results
    )


def test_reranked_pipeline_preserves_metadata():
    pipeline, _, _ = build_pipeline()

    results = pipeline.retrieve_reranked(
        "query",
        top_k=2,
        document_id="doc-1",
    )

    assert len(results) == 2
    assert all(item.retriever == "reranked" for item in results)
    assert all(item.chunk.document_id == "doc-1" for item in results)
    assert all(item.chunk.page_number >= 1 for item in results)


def test_empty_query_returns_empty_results():
    pipeline, _, _ = build_pipeline()

    assert pipeline.retrieve_hybrid(" ") == []
    assert pipeline.retrieve_reranked(" ") == []
from app.config import settings
from app.models import RetrievalResult
from app.services.rrf import reciprocal_rank_fusion


class HybridRetriever:
    """
    Runs dense and BM25 retrieval, fuses their rankings,
    and optionally reranks the fused candidates.
    """

    def __init__(
        self,
        dense_retriever,
        bm25_retriever,
        reranker,
        *,
        rrf_k: int | None = None,
        candidate_k: int | None = None,
    ):
        self.dense_retriever = dense_retriever
        self.bm25_retriever = bm25_retriever
        self.reranker = reranker

        self.rrf_k = rrf_k if rrf_k is not None else settings.rrf_k
        self.candidate_k = (
            candidate_k
            if candidate_k is not None
            else settings.rerank_candidate_k
        )

        if self.rrf_k < 1:
            raise ValueError("rrf_k must be >= 1")

        if self.candidate_k < 1:
            raise ValueError("candidate_k must be >= 1")

    def retrieve_dense(
        self,
        query: str,
        *,
        top_k: int = 5,
        document_id: str | None = None,
    ) -> list[RetrievalResult]:
        return self.dense_retriever.retrieve(
            query,
            top_k=top_k,
            document_id=document_id,
        )

    def retrieve_bm25(
        self,
        query: str,
        *,
        top_k: int = 5,
        document_id: str | None = None,
    ) -> list[RetrievalResult]:
        return self.bm25_retriever.retrieve(
            query,
            top_k=top_k,
            document_id=document_id,
        )

    def retrieve_hybrid(
        self,
        query: str,
        *,
        top_k: int = 5,
        candidate_k: int | None = None,
        document_id: str | None = None,
    ) -> list[RetrievalResult]:
        if not query or not query.strip():
            return []

        if top_k < 1:
            raise ValueError("top_k must be >= 1")

        fetch_k = candidate_k or max(self.candidate_k, top_k)

        dense_results = self.retrieve_dense(
            query,
            top_k=fetch_k,
            document_id=document_id,
        )

        bm25_results = self.retrieve_bm25(
            query,
            top_k=fetch_k,
            document_id=document_id,
        )

        return reciprocal_rank_fusion(
            [dense_results, bm25_results],
            k=self.rrf_k,
            top_k=top_k,
            document_id=document_id,
        )

    def retrieve_reranked(
        self,
        query: str,
        *,
        top_k: int = 5,
        candidate_k: int | None = None,
        document_id: str | None = None,
    ) -> list[RetrievalResult]:
        if not query or not query.strip():
            return []

        fetch_k = candidate_k or max(self.candidate_k, top_k)

        candidates = self.retrieve_hybrid(
            query,
            top_k=fetch_k,
            candidate_k=fetch_k,
            document_id=document_id,
        )

        return self.reranker.rerank(
            query,
            candidates,
            top_k=top_k,
            document_id=document_id,
        )
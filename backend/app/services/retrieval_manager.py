from app.config import settings
from app.services.bm25_retrieval import BM25Retriever
from app.services.dense_retrieval import DenseRetriever
from app.services.document_store import document_store
from app.services.embeddings import SentenceTransformerEmbedder
from app.services.hybrid_retrieval import HybridRetriever
from app.services.reranker import CrossEncoderReranker


class RetrievalPipelineManager:
    """Build and cache the existing hybrid retrieval pipeline."""

    def __init__(self) -> None:
        self._pipeline = None
        self._built_version = -1

    def get_pipeline(self) -> HybridRetriever:
        current_version = document_store.version

        if (
            self._pipeline is not None
            and self._built_version == current_version
        ):
            return self._pipeline

        chunks = document_store.get_all_chunks()

        if not chunks:
            raise ValueError("No documents have been uploaded yet.")

        embedder = SentenceTransformerEmbedder(
            model_name=settings.embedding_model_name
        )

        dense_retriever = DenseRetriever(
            chunks,
            embedder=embedder,
        )

        bm25_retriever = BM25Retriever(chunks)

        reranker = CrossEncoderReranker(
            model_name=settings.reranker_model_name
        )

        self._pipeline = HybridRetriever(
            dense_retriever,
            bm25_retriever,
            reranker,
            rrf_k=settings.rrf_k,
            candidate_k=settings.rerank_candidate_k,
        )

        self._built_version = current_version

        return self._pipeline

    def retrieve(
        self,
        question: str,
        *,
        top_k: int = 5,
        document_id: str | None = None,
    ):
        pipeline = self.get_pipeline()

        return pipeline.retrieve_reranked(
            question,
            top_k=top_k,
            document_id=document_id,
        )


retrieval_manager = RetrievalPipelineManager()
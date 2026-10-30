from typing import Protocol

from app.models import RetrievalResult


class CrossEncoderModel(Protocol):
    def predict(self, sentences, **kwargs):
        ...


class CrossEncoderReranker:
    """Reorders retrieved chunks using query-passage relevance scores."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        model: CrossEncoderModel | None = None,
    ):
        self.model_name = model_name
        self._model = model

    @property
    def model(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:
                raise RuntimeError(
                    "Install sentence-transformers to use reranking."
                ) from exc

            self._model = CrossEncoder(self.model_name)

        return self._model

    def rerank(
        self,
        query: str,
        candidates: list[RetrievalResult],
        *,
        top_k: int | None = None,
        document_id: str | None = None,
    ) -> list[RetrievalResult]:
        if not query or not query.strip():
            return []

        if top_k is not None and top_k < 1:
            raise ValueError("top_k must be >= 1")

        filtered = [
            candidate
            for candidate in candidates
            if document_id is None
            or candidate.chunk.document_id == document_id
        ]

        if not filtered:
            return []

        pairs = [
            (query, candidate.chunk.text)
            for candidate in filtered
        ]

        scores = self.model.predict(pairs)

        if len(scores) != len(filtered):
            raise ValueError(
                "Cross-encoder returned an unexpected number of scores."
            )

        scored = list(zip(filtered, scores))

        # Stable sorting preserves original RRF ordering for equal scores.
        scored.sort(
            key=lambda item: float(item[1]),
            reverse=True,
        )

        if top_k is not None:
            scored = scored[:top_k]

        return [
            RetrievalResult(
                chunk=result.chunk,
                score=float(score),
                rank=rank,
                retriever="reranked",
            )
            for rank, (result, score) in enumerate(scored, start=1)
        ]
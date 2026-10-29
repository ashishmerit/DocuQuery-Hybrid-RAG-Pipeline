from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import faiss
import numpy as np

from app.config import settings
from app.models import Chunk, RetrievalResult
from app.services.embeddings import SentenceTransformerEmbedder


class Embedder(Protocol):
    def encode(self, texts: Sequence[str]) -> np.ndarray:
        ...


class DenseRetriever:
    """Search page-aware chunks using normalized vectors and FAISS."""

    def __init__(
        self,
        chunks: Sequence[Chunk],
        *,
        embedder: Embedder | None = None,
        model_name: str | None = None,
    ) -> None:
        self.chunks = list(chunks)
        self.embedder = embedder
        self.index: faiss.IndexFlatIP | None = None
        self._vectors: np.ndarray | None = None

        if not self.chunks:
            return

        if self.embedder is None:
            self.embedder = SentenceTransformerEmbedder(
                model_name or settings.embedding_model_name
            )

        texts = [chunk.text for chunk in self.chunks]
        vectors = self._validated_vectors(
            self.embedder.encode(texts),
            expected_rows=len(texts),
        )

        vectors = self._normalize(vectors)
        self._vectors = vectors

        # Exact inner-product search.
        # With normalized vectors, inner product equals cosine similarity.
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)

    @staticmethod
    def _validated_vectors(
        vectors: np.ndarray,
        expected_rows: int,
    ) -> np.ndarray:
        array = np.asarray(vectors, dtype=np.float32)

        if (
            array.ndim != 2
            or array.shape[0] != expected_rows
            or array.shape[1] == 0
        ):
            raise ValueError(
                "Embedder must return a non-empty 2D matrix "
                "with one row per text."
            )

        if not np.isfinite(array).all():
            raise ValueError(
                "Embedding vectors must contain only finite values."
            )

        return np.ascontiguousarray(array, dtype=np.float32)

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0

        return np.ascontiguousarray(
            vectors / norms,
            dtype=np.float32,
        )

    def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        document_id: str | None = None,
    ) -> list[RetrievalResult]:
        limit = (
            settings.retrieval_top_k
            if top_k is None
            else top_k
        )

        if limit < 1:
            raise ValueError("top_k must be at least 1")

        if (
            not query.strip()
            or self.index is None
            or self.embedder is None
        ):
            return []

        query_vectors = self._validated_vectors(
            self.embedder.encode([query]),
            expected_rows=1,
        )

        if query_vectors.shape[1] != self.index.d:
            raise ValueError(
                "Query embedding dimension does not match "
                "the FAISS index."
            )

        query_vector = self._normalize(query_vectors)

        # Search all indexed chunks before applying document scope.
        # This ensures filtering doesn't unnecessarily reduce top_k.
        scores, indices = self.index.search(
            query_vector,
            len(self.chunks),
        )

        results: list[RetrievalResult] = []

        for score, index in zip(
            scores[0],
            indices[0],
            strict=True,
        ):
            if index < 0:
                continue

            chunk = self.chunks[int(index)]

            if (
                document_id is not None
                and chunk.document_id != document_id
            ):
                continue

            results.append(
                RetrievalResult(
                    chunk=chunk,
                    score=float(score),
                    rank=len(results) + 1,
                    retriever="dense",
                )
            )

            if len(results) == limit:
                break

        return results
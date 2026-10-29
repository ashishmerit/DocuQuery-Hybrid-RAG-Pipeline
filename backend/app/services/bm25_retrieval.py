from __future__ import annotations

import re
from collections.abc import Sequence

from rank_bm25 import BM25Okapi

from app.config import settings
from app.models import Chunk, RetrievalResult


_TOKEN_PATTERN = re.compile(
    r"[\w]+(?:['-][\w]+)*",
    flags=re.UNICODE,
)


def tokenize(text: str) -> list[str]:
    """Apply deterministic, case-insensitive tokenization."""
    return _TOKEN_PATTERN.findall(text.casefold())


class BM25Retriever:
    """Keyword retriever using rank-bm25's BM25Okapi."""

    def __init__(self, chunks: Sequence[Chunk]) -> None:
        self.chunks = list(chunks)

        self._tokens_by_chunk = [
            tokenize(chunk.text)
            for chunk in self.chunks
        ]

        self._bm25 = (
            BM25Okapi(self._tokens_by_chunk)
            if self.chunks and any(self._tokens_by_chunk)
            else None
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

        query_tokens = tokenize(query)

        if not query_tokens or self._bm25 is None:
            return []

        scores = self._bm25.get_scores(query_tokens)
        query_token_set = set(query_tokens)

        candidates: list[tuple[Chunk, float]] = []

        for chunk, tokens, score in zip(
            self.chunks,
            self._tokens_by_chunk,
            scores,
            strict=True,
        ):
            if (
                document_id is not None
                and chunk.document_id != document_id
            ):
                continue

            # Only include passages with actual lexical overlap.
            if not query_token_set.intersection(tokens):
                continue

            # Keep negative BM25 scores if a lexical match exists.
            # A negative score can occur for common terms.
            candidates.append((chunk, float(score)))

        candidates.sort(
            key=lambda item: item[1],
            reverse=True,
        )

        return [
            RetrievalResult(
                chunk=chunk,
                score=score,
                rank=rank,
                retriever="bm25",
            )
            for rank, (chunk, score) in enumerate(
                candidates[:limit],
                start=1,
            )
        ]
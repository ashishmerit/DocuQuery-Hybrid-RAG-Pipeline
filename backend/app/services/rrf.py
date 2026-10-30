from collections import defaultdict
from typing import Iterable

from app.models import RetrievalResult


def reciprocal_rank_fusion(
    rankings: Iterable[list[RetrievalResult]],
    *,
    k: int = 60,
    top_k: int | None = None,
    document_id: str | None = None,
) -> list[RetrievalResult]:
    """
    Combine multiple ranked lists using Reciprocal Rank Fusion.

    RRF score:
        score(d) = sum(1 / (k + rank(d)))

    Ranks are 1-based. A chunk appearing in multiple lists
    accumulates contributions from each list.
    """
    if k < 1:
        raise ValueError("RRF k must be >= 1")

    if top_k is not None and top_k < 1:
        raise ValueError("top_k must be >= 1")

    scores: dict[str, float] = defaultdict(float)
    chunks = {}
    first_seen = {}
    position = 0

    for ranking in rankings:
        seen_in_ranking = set()

        for result in ranking:
            chunk = result.chunk

            if document_id and chunk.document_id != document_id:
                continue

            chunk_id = chunk.chunk_id

            # Avoid counting duplicate appearances in one ranking.
            if chunk_id in seen_in_ranking:
                continue

            seen_in_ranking.add(chunk_id)

            scores[chunk_id] += 1.0 / (k + result.rank)

            if chunk_id not in chunks:
                chunks[chunk_id] = chunk
                first_seen[chunk_id] = position
                position += 1

    ordered_ids = sorted(
        scores,
        key=lambda chunk_id: (
            -scores[chunk_id],
            first_seen[chunk_id],
        ),
    )

    if top_k is not None:
        ordered_ids = ordered_ids[:top_k]

    return [
        RetrievalResult(
            chunk=chunks[chunk_id],
            score=float(scores[chunk_id]),
            rank=rank,
            retriever="hybrid",
        )
        for rank, chunk_id in enumerate(ordered_ids, start=1)
    ]
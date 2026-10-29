"""Evaluate Day 2 dense and BM25 retrieval on a fixed sample dataset.

Run from backend/:
    python ..\\evaluation\\evaluate_retrieval.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import mean
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = ROOT / "backend"

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.models import Chunk  # noqa: E402
from app.services.bm25_retrieval import BM25Retriever  # noqa: E402
from app.services.dense_retrieval import DenseRetriever  # noqa: E402


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def calculate_metrics(
    queries,
    retriever,
    top_k: int = 5,
) -> dict:
    recalls: list[float] = []
    reciprocal_ranks: list[float] = []
    latencies_ms: list[float] = []

    for item in queries:
        relevant_ids = set(item["relevant_chunk_ids"])

        started = perf_counter()
        results = retriever.retrieve(
            item["query"],
            top_k=top_k,
        )
        elapsed_ms = (perf_counter() - started) * 1000
        latencies_ms.append(elapsed_ms)

        ranked_ids = [
            result.chunk.chunk_id
            for result in results
        ]

        recall = (
            len(set(ranked_ids) & relevant_ids)
            / len(relevant_ids)
        )
        recalls.append(recall)

        reciprocal_rank = 0.0

        for rank, chunk_id in enumerate(ranked_ids, start=1):
            if chunk_id in relevant_ids:
                reciprocal_rank = 1.0 / rank
                break

        reciprocal_ranks.append(reciprocal_rank)

    return {
        "queries": len(queries),
        f"recall@{top_k}": round(mean(recalls), 4),
        "mrr": round(mean(reciprocal_ranks), 4),
        "mean_query_latency_ms": round(
            mean(latencies_ms),
            3,
        ),
    }


def main() -> None:
    corpus_data = load_json(
        ROOT / "evaluation" / "sample_corpus.json"
    )
    queries = load_json(
        ROOT / "evaluation" / "test_queries.json"
    )

    chunks = [
        Chunk.model_validate(item)
        for item in corpus_data
    ]

    corpus_ids = {chunk.chunk_id for chunk in chunks}

    missing_ids = {
        chunk_id
        for item in queries
        for chunk_id in item["relevant_chunk_ids"]
        if chunk_id not in corpus_ids
    }

    if missing_ids:
        raise ValueError(
            "Evaluation labels reference unknown chunks: "
            f"{sorted(missing_ids)}"
        )

    # Construct each retriever once. Model loading and dense index
    # construction happen before per-query latency measurements.
    bm25 = BM25Retriever(chunks)
    dense = DenseRetriever(chunks)

    report = {
        "dataset": (
            "Small manually curated sample corpus; "
            "not a general benchmark"
        ),
        "corpus_chunks": len(chunks),
        "top_k": 5,
        "bm25": calculate_metrics(queries, bm25),
        "dense": calculate_metrics(queries, dense),
    }

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
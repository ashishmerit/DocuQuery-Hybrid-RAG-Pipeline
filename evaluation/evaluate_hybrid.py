import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"

sys.path.insert(0, str(BACKEND))

from app.config import settings
from app.models import Chunk
from app.services.embeddings import SentenceTransformerEmbedder
from app.services.dense_retrieval import DenseRetriever
from app.services.bm25_retrieval import BM25Retriever
from app.services.reranker import CrossEncoderReranker
from app.services.hybrid_retrieval import HybridRetriever


TOP_K = 5


def load_json(path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def recall_at_k(retrieved_ids, relevant_ids, k=5):
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0
    return len(set(retrieved_ids[:k]) & relevant) / len(relevant)


def reciprocal_rank(retrieved_ids, relevant_ids):
    relevant = set(relevant_ids)
    for rank, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids, relevant_ids, k=5):
    relevant = set(relevant_ids)

    dcg = sum(
        1.0 / __import__("math").log2(rank + 1)
        for rank, chunk_id in enumerate(retrieved_ids[:k], start=1)
        if chunk_id in relevant
    )

    ideal_hits = min(len(relevant), k)
    idcg = sum(
        1.0 / __import__("math").log2(rank + 1)
        for rank in range(1, ideal_hits + 1)
    )

    return dcg / idcg if idcg else 0.0


def evaluate(name, retrieve, queries):
    recalls = []
    mrrs = []
    ndcgs = []
    latencies = []

    per_query = []

    for item in queries:
        query = item["query"]
        relevant = item["relevant_chunk_ids"]

        start = time.perf_counter()
        results = retrieve(query)
        elapsed_ms = (time.perf_counter() - start) * 1000

        retrieved_ids = [
            result.chunk.chunk_id
            for result in results
        ]

        recall = recall_at_k(retrieved_ids, relevant, TOP_K)
        mrr = reciprocal_rank(retrieved_ids, relevant)
        ndcg = ndcg_at_k(retrieved_ids, relevant, TOP_K)

        recalls.append(recall)
        mrrs.append(mrr)
        ndcgs.append(ndcg)
        latencies.append(elapsed_ms)

        per_query.append({
            "query_id": item["query_id"],
            "retrieved_chunk_ids": retrieved_ids,
            "relevant_chunk_ids": relevant,
            "recall_at_5": recall,
            "mrr": mrr,
            "ndcg_at_5": ndcg,
            "latency_ms": elapsed_ms,
        })

    return {
        "configuration": name,
        "query_count": len(queries),
        "recall_at_5": statistics.mean(recalls),
        "mrr": statistics.mean(mrrs),
        "ndcg_at_5": statistics.mean(ndcgs),
        "mean_latency_ms": statistics.mean(latencies),
        "per_query": per_query,
    }


def main():
    corpus_data = load_json(ROOT / "evaluation" / "sample_corpus.json")
    queries = load_json(ROOT / "evaluation" / "test_queries.json")

    chunks = [Chunk(**row) for row in corpus_data]

    embedder = SentenceTransformerEmbedder(
        settings.embedding_model_name
    )

    dense = DenseRetriever(chunks)
    bm25 = BM25Retriever(chunks)

    reranker = CrossEncoderReranker(
        settings.reranker_model_name
    )

    hybrid = HybridRetriever(
        dense,
        bm25,
        reranker,
    )

    configurations = [
        (
            "dense_only",
            lambda query: dense.retrieve(query, top_k=TOP_K),
        ),
        (
            "bm25_only",
            lambda query: bm25.retrieve(query, top_k=TOP_K),
        ),
        (
            "hybrid_rrf",
            lambda query: hybrid.retrieve_hybrid(
                query,
                top_k=TOP_K,
                candidate_k=20,
            ),
        ),
        (
            "hybrid_rrf_reranked",
            lambda query: hybrid.retrieve_reranked(
                query,
                top_k=TOP_K,
                candidate_k=20,
            ),
        ),
    ]

    reports = []

    for name, retrieve in configurations:
        print(f"Evaluating {name}...")
        reports.append(evaluate(name, retrieve, queries))

    summary = {
        "dataset": {
            "corpus_chunks": len(chunks),
            "queries": len(queries),
            "top_k": TOP_K,
        },
        "results": [
            {
                key: value
                for key, value in report.items()
                if key != "per_query"
            }
            for report in reports
        ],
        "per_query": {
            report["configuration"]: report["per_query"]
            for report in reports
        },
    }

    output = ROOT / "evaluation" / "Complete_results.json"
    output.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\nComplete evaluation summary")
    print("-" * 78)

    for report in reports:
        print(
            f"{report['configuration']:<26}"
            f"Recall@5: {report['recall_at_5']:.4f}  "
            f"MRR: {report['mrr']:.4f}  "
            f"nDCG@5: {report['ndcg_at_5']:.4f}  "
            f"Latency: {report['mean_latency_ms']:.2f} ms"
        )

    print(f"\nDetailed results saved to: {output}")


if __name__ == "__main__":
    main()
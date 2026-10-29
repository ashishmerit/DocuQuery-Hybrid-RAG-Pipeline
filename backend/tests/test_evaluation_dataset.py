import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_evaluation_queries_reference_existing_chunks_and_have_enough_examples():
    corpus = json.loads(
        (
            ROOT / "evaluation" / "sample_corpus.json"
        ).read_text(encoding="utf-8")
    )

    queries = json.loads(
        (
            ROOT / "evaluation" / "test_queries.json"
        ).read_text(encoding="utf-8")
    )

    chunk_ids = {chunk["chunk_id"] for chunk in corpus}

    assert len(corpus) >= 10
    assert len(queries) >= 15
    assert len(chunk_ids) == len(corpus)
    assert all(chunk["page_number"] >= 1 for chunk in corpus)
    assert all(query["relevant_chunk_ids"] for query in queries)

    assert all(
        relevant_id in chunk_ids
        for query in queries
        for relevant_id in query["relevant_chunk_ids"]
    )
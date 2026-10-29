import pytest

from app.services.ingestion import chunk_page_text


def test_chunking_preserves_page_and_overlaps():
    text = "abcdefghijklmnopqrstuvwxyz"
    chunks = chunk_page_text(text, document_id="doc-1", page_number=3, chunk_size=10, overlap=3)
    assert [chunk.text for chunk in chunks] == ["abcdefghij", "hijklmnopq", "opqrstuvwx", "vwxyz"]
    assert all(chunk.page_number == 3 for chunk in chunks)
    assert len({chunk.chunk_id for chunk in chunks}) == len(chunks)


def test_chunking_rejects_invalid_overlap():
    with pytest.raises(ValueError):
        chunk_page_text("hello", document_id="doc-1", page_number=1, chunk_size=5, overlap=5)


def test_empty_page_has_no_chunks():
    assert chunk_page_text("  \n  ", document_id="doc-1", page_number=1) == []

def test_chunking_rejects_non_positive_chunk_size():
    with pytest.raises(
        ValueError,
        match="chunk_size must be positive",
    ):
        chunk_page_text(
            "hello",
            document_id="doc-1",
            page_number=1,
            chunk_size=0,
            overlap=0,
        )

def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["app"] == "DocuQuery API"


def test_upload_pdf_returns_page_metadata_and_chunks(client, sample_pdf_bytes):
    response = client.post(
        "/api/documents",
        files={"file": ("sample.pdf", sample_pdf_bytes, "application/pdf")},
    )
    assert response.status_code == 201, response.text
    payload = response.json()
    assert payload["filename"] == "sample.pdf"
    assert payload["page_count"] == 2
    assert payload["chunk_count"] >= 2
    assert all(chunk["document_id"] == payload["document_id"] for chunk in payload["chunks"])
    assert {chunk["page_number"] for chunk in payload["chunks"]} == {1, 2}
    assert any("searchable text" in chunk["text"] for chunk in payload["chunks"])


def test_rejects_non_pdf_extension(client):
    response = client.post(
        "/api/documents",
        files={"file": ("notes.txt", b"hello", "text/plain")},
    )
    assert response.status_code == 415


def test_rejects_invalid_pdf_header(client):
    response = client.post(
        "/api/documents",
        files={"file": ("fake.pdf", b"not a pdf", "application/pdf")},
    )
    assert response.status_code == 400


def test_rejects_empty_pdf(client):
    response = client.post(
        "/api/documents",
        files={"file": ("empty.pdf", b"", "application/pdf")},
    )
    assert response.status_code == 400

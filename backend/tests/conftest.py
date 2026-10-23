import fitz
import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def sample_pdf_bytes():
    doc = fitz.open()
    page1 = doc.new_page()
    page1.insert_text((72, 72), "DocuQuery extracts searchable text from PDF pages.")
    page2 = doc.new_page()
    page2.insert_text((72, 72), "Page two contains additional source evidence.")
    data = doc.tobytes()
    doc.close()
    return data

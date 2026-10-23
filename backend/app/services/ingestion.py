from __future__ import annotations

import re
import uuid
from pathlib import Path

import fitz
from fastapi import HTTPException, UploadFile, status

from app.config import settings
from app.models import Chunk, IngestionResponse

PDF_MIME_TYPES = {"application/pdf", "application/x-pdf"}


def normalize_text(text: str) -> str:
    """Normalize whitespace without joining words from separate lines."""
    text = text.replace("\x00", " ")
    text = re.sub(r"[\t\r\f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def chunk_page_text(
    text: str,
    *,
    document_id: str,
    page_number: int,
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> list[Chunk]:
    """Split one page into overlapping character chunks with source metadata."""
    size = chunk_size or settings.chunk_size
    overlap_size = settings.chunk_overlap if overlap is None else overlap
    if size < 1:
        raise ValueError("chunk_size must be positive")
    if overlap_size < 0 or overlap_size >= size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    text = normalize_text(text)
    if not text:
        return []

    chunks: list[Chunk] = []
    step = size - overlap_size
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        piece = text[start:end].strip()
        if piece:
            # IDs are unique within this ingestion request and stable in ordering.
            chunks.append(Chunk(
                document_id=document_id,
                chunk_id=f"{document_id}-p{page_number}-c{len(chunks) + 1}",
                page_number=page_number,
                text=piece,
                start_char=start,
                end_char=end,
            ))
        if end >= len(text):
            break
        start += step
    return chunks


async def ingest_pdf(file: UploadFile) -> IngestionResponse:
    filename = Path(file.filename or "").name
    if not filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="A filename is required.")
    if Path(filename).suffix.lower() != ".pdf":
        raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail="Only PDF files are accepted.")
    if file.content_type and file.content_type not in PDF_MIME_TYPES:
        raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail="The uploaded file must have a PDF content type.")

    max_bytes = settings.max_upload_mb * 1024 * 1024
    data = await file.read(max_bytes + 1)
    if not data:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="The uploaded file is empty.")
    if len(data) > max_bytes:
        raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail=f"PDF exceeds the {settings.max_upload_mb} MB upload limit.")
    if not data.startswith(b"%PDF-"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="The uploaded file does not have a valid PDF header.")

    document_id = str(uuid.uuid4())
    try:
        with fitz.open(stream=data, filetype="pdf") as document:
            if document.is_encrypted:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Password-protected PDFs are not supported.")
            if document.page_count == 0:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="The PDF contains no pages.")
            chunks: list[Chunk] = []
            for page_index, page in enumerate(document, start=1):
                page_text = page.get_text("text")
                chunks.extend(chunk_page_text(page_text, document_id=document_id, page_number=page_index))
            if not chunks:
                raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="No extractable text was found. Scanned PDFs require OCR, which is not included in this MVP.")
            return IngestionResponse(
                document_id=document_id,
                filename=filename,
                page_count=document.page_count,
                chunk_count=len(chunks),
                chunks=chunks,
            )
    except HTTPException:
        raise
    except (fitz.FileDataError, RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="The PDF is malformed, unreadable, or unsupported.") from exc
    finally:
        await file.close()

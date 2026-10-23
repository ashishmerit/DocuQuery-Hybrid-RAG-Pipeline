from fastapi import APIRouter, File, UploadFile

from app.models import IngestionResponse
from app.services.ingestion import ingest_pdf

router = APIRouter(prefix="/api/documents", tags=["documents"])


@router.post("", response_model=IngestionResponse, status_code=201)
async def upload_document(file: UploadFile = File(...)) -> IngestionResponse:
    """Validate and parse a PDF, returning extracted chunks for Day 1 verification."""
    return await ingest_pdf(file)

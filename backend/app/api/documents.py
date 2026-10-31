from fastapi import APIRouter, File, UploadFile

from app.models import IngestionResponse
from app.services.document_store import document_store
from app.services.ingestion import ingest_pdf

router = APIRouter(prefix="/api/documents", tags=["documents"])


@router.post("", response_model=IngestionResponse, status_code=201)
async def upload_document(file: UploadFile = File(...)) -> IngestionResponse:
    """Validate, parse, and register a PDF for subsequent querying."""

    document = await ingest_pdf(file)
    document_store.add(document)

    return document
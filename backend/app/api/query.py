from fastapi import APIRouter, HTTPException, status

from app.config import settings
from app.models import QueryRequest, QueryResponse
from app.services.document_store import document_store
from app.services.generation import AnswerGenerationService
from app.services.llm_client import (
    LLMProviderError,
    LLMTimeoutError,
    MissingAPIKeyError,
    GeminiClient,
)
from app.services.retrieval_manager import retrieval_manager


router = APIRouter(prefix="/api/query", tags=["query"])


@router.post("", response_model=QueryResponse)
async def query_documents(
    request: QueryRequest,
) -> QueryResponse:

    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Question cannot be empty.",
        )

    if request.document_id:
        document = document_store.get(request.document_id)

        if document is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found.",
            )

    elif not document_store.list_documents():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Upload a document before asking a question.",
        )

    try:
        results = retrieval_manager.retrieve(
            question,
            top_k=request.top_k or settings.generation_top_k,
            document_id=request.document_id,
        )

        generation_service = AnswerGenerationService(
            GeminiClient()
        )

        return await generation_service.generate(
            question,
            results,
        )

    except MissingAPIKeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    except LLMTimeoutError as exc:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=str(exc),
        ) from exc

    except LLMProviderError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc
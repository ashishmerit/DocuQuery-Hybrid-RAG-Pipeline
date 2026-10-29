from typing import Literal

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    document_id: str
    chunk_id: str
    page_number: int = Field(ge=1)
    text: str
    start_char: int = Field(ge=0)
    end_char: int = Field(ge=0)


class IngestionResponse(BaseModel):
    document_id: str
    filename: str
    page_count: int
    chunk_count: int
    chunks: list[Chunk]


class HealthResponse(BaseModel):
    status: str
    app: str


class RetrievalResult(BaseModel):
    """One ranked passage returned by a retrieval method."""

    chunk: Chunk
    score: float
    rank: int = Field(ge=1)
    retriever: Literal["dense", "bm25"]
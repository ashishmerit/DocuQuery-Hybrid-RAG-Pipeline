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


from typing import Literal

class RetrievalResult(BaseModel):
    """A ranked retrieval result with original source metadata."""

    chunk: Chunk
    score: float
    rank: int = Field(ge=1)
    retriever: Literal["dense", "bm25", "hybrid", "reranked"]

class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    document_id: str | None = None
    top_k: int = Field(default=5, ge=1, le=20)


class SourceCitation(BaseModel):
    citation_id: str
    document_id: str
    document_name: str
    chunk_id: str
    page_number: int
    excerpt: str


class QueryResponse(BaseModel):
    status: Literal["answered", "insufficient_evidence"]
    question: str
    answer: str
    citations: list[SourceCitation] = Field(default_factory=list)
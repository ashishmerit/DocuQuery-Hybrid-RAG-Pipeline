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

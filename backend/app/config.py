from pydantic_settings import BaseSettings, SettingsConfigDict



class Settings(BaseSettings):
    app_name: str = "DocuQuery API"
    max_upload_mb: int = 15
    chunk_size: int = 900
    chunk_overlap: int = 150

    embedding_model_name: str = (
        "sentence-transformers/all-MiniLM-L6-v2"
    )
    retrieval_top_k: int = 5

    cors_origins: str = "http://localhost:5173"

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )

    @property
    def allowed_origins(self) -> list[str]:
        return [
            origin.strip()
            for origin in self.cors_origins.split(",")
            if origin.strip()
        ]

    reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rrf_k: int = 60
    rerank_candidate_k: int = 20



    # Google Gemini configuration
    gemini_api_key: str = ""
    gemini_base_url: str = (
        "https://generativelanguage.googleapis.com/v1beta"
    )
    gemini_model: str = "gemini-2.5-flash"

    # LLM generation configuration
    llm_timeout_seconds: float = 30.0
    llm_max_tokens: int = 900
    generation_top_k: int = 5

settings = Settings()
from __future__ import annotations

from threading import RLock

from app.models import Chunk, IngestionResponse


class DocumentStore:
    """Process-local storage for uploaded documents and their chunks."""

    def __init__(self) -> None:
        self._documents: dict[str, IngestionResponse] = {}
        self._lock = RLock()
        self._version = 0

    def add(self, document: IngestionResponse) -> None:
        with self._lock:
            self._documents[document.document_id] = document
            self._version += 1

    def get(self, document_id: str) -> IngestionResponse | None:
        with self._lock:
            return self._documents.get(document_id)

    def get_all_chunks(self) -> list[Chunk]:
        with self._lock:
            return [
                chunk
                for document in self._documents.values()
                for chunk in document.chunks
            ]

    def list_documents(self) -> list[IngestionResponse]:
        with self._lock:
            return list(self._documents.values())

    def clear(self) -> None:
        """Clear stored documents; primarily useful for isolated tests."""
        with self._lock:
            self._documents.clear()
            self._version += 1

    @property
    def version(self) -> int:
        with self._lock:
            return self._version


document_store = DocumentStore()
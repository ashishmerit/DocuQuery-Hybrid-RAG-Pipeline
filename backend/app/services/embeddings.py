from __future__ import annotations

from collections.abc import Sequence

import numpy as np


class SentenceTransformerEmbedder:
    """Generate dense vectors for text using Sentence Transformers."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Sentence Transformers is not installed. "
                "Install backend requirements."
            ) from exc

        self.model_name = model_name
        self._model = SentenceTransformer(model_name)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Return one normalized embedding vector per input text."""
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        vectors = self._model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        return np.asarray(vectors, dtype=np.float32)
"""Wraps sentence-transformers: batched encoding + L2 normalization.

Vectors are always L2-normalized so that FAISS's inner-product index
(`IndexFlatIP`) computes cosine similarity - see vectorstore.py.
"""

from __future__ import annotations

import hashlib

import numpy as np
from sentence_transformers import SentenceTransformer

from codesearch.config import Settings


class Embedder:
    def __init__(self, model_name: str, batch_size: int = 32) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = SentenceTransformer(model_name)

    @property
    def dimension(self) -> int:
        dim = self._model.get_embedding_dimension()
        if dim is None:
            raise RuntimeError(f"could not determine embedding dimension for {self.model_name}")
        return dim

    def encode(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        vectors = self._model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return vectors.astype(np.float32)

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]

    @property
    def fingerprint(self) -> str:
        """Short id capturing model name + dimension, stored in manifest.json
        so a stale/mismatched index can be detected before returning bad
        results."""
        raw = f"{self.model_name}:{self.dimension}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]


def build_embedder(settings: Settings) -> Embedder:
    return Embedder(settings.embedding_model, settings.embedding_batch_size)

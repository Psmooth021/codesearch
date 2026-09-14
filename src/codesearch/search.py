"""Embeds a query and searches an already-built index."""

from __future__ import annotations

from pathlib import Path

from codesearch.embedding import Embedder
from codesearch.indexer import default_index_dir
from codesearch.vectorstore import SearchHit, VectorStore


class IndexNotFoundError(RuntimeError):
    pass


def open_store(repo_path: Path, embedder: Embedder, index_dir: Path | None = None) -> VectorStore:
    index_dir = index_dir or default_index_dir(repo_path)
    manifest_path = index_dir / "manifest.json"
    if not manifest_path.exists():
        raise IndexNotFoundError(
            f"No index found at {index_dir}. Run `codesearch index {repo_path}` first."
        )
    store = VectorStore(index_dir, embedder.dimension)
    manifest = store.load_manifest()
    if manifest and manifest.get("embedding_fingerprint") != embedder.fingerprint:
        store.close()
        raise IndexNotFoundError(
            f"Index at {index_dir} was built with a different embedding model "
            f"({manifest.get('embedding_model')}); pass --embedding-model to match it "
            "or rebuild with the current one."
        )
    return store


def query(
    store: VectorStore,
    embedder: Embedder,
    question: str,
    k: int = 5,
    language: str | None = None,
) -> list[SearchHit]:
    vector = embedder.encode_one(question)
    # Over-fetch when filtering by language so post-filtering still leaves
    # up to k results instead of silently returning fewer.
    fetch_k = k * 4 if language else k
    hits = store.search(vector, fetch_k)
    if language:
        hits = [h for h in hits if h.language == language]
    return hits[:k]

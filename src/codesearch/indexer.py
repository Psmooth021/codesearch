"""Orchestrates walk -> chunk -> embed -> vectorstore, with incremental
re-indexing: unchanged files are skipped entirely (no re-chunk/re-embed),
changed files have their old chunks purged and replaced, and files that
disappeared from disk have their chunks pruned.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from codesearch.chunking.base import Chunker
from codesearch.chunking.registry import get_chunker_for_extension
from codesearch.config import CHUNKER_VERSION, INDEX_DIR_NAME, MANIFEST_FILE_NAME, Settings
from codesearch.embedding import Embedder, build_embedder
from codesearch.utils.hashing import file_hash
from codesearch.vectorstore import VectorStore
from codesearch.walker import iter_source_files


@dataclass
class IndexStats:
    files_scanned: int = 0
    files_indexed: int = 0
    files_skipped_unchanged: int = 0
    files_removed: int = 0
    chunks_written: int = 0
    chunker_upgraded: bool = False


def default_index_dir(repo_path: Path) -> Path:
    return repo_path / INDEX_DIR_NAME


def build_manifest(embedder: Embedder, settings: Settings) -> dict:
    return {
        "embedding_model": embedder.model_name,
        "embedding_fingerprint": embedder.fingerprint,
        "chunker_version": CHUNKER_VERSION,
        "max_chunk_tokens": settings.max_chunk_tokens,
    }


def index_repo(
    repo_path: Path,
    index_dir: Path | None = None,
    settings: Settings | None = None,
    embedder: Embedder | None = None,
    rebuild: bool = False,
    chunker_factory: Callable[[str, Settings], Chunker] | None = None,
) -> IndexStats:
    """`chunker_factory(extension, settings) -> Chunker` overrides the
    default per-extension dispatch - used by the eval harness's
    `--baseline naive` mode to force every file through FallbackChunker
    instead of the AST-aware chunkers, for an apples-to-apples comparison."""
    settings = settings or Settings()
    repo_path = repo_path.resolve()
    index_dir = index_dir or default_index_dir(repo_path)
    embedder = embedder or build_embedder(settings)
    chunker_factory = chunker_factory or get_chunker_for_extension

    stats = IndexStats()

    manifest_path = index_dir / MANIFEST_FILE_NAME
    existing_manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None

    existing_chunker_version = (
        existing_manifest.get("chunker_version") if existing_manifest is not None else None
    )
    if existing_manifest is not None and existing_chunker_version != CHUNKER_VERSION:
        # Chunking logic changed since this index was built (e.g. a new
        # language-specific chunker was added). Incremental indexing only
        # re-chunks a file when its *content* hash changes, so a plain
        # re-run of `codesearch index` would never pick this up on its own
        # - silently continuing to serve chunks produced by outdated
        # chunking logic forever. Force a full rebuild instead. Unlike an
        # embedding-model change (an explicit, deliberate --embedding-model
        # flag), this happens passively just from upgrading the package, so
        # it auto-rebuilds rather than requiring the user to remember
        # --rebuild themselves.
        rebuild = True
        stats.chunker_upgraded = True

    if rebuild and index_dir.exists():
        shutil.rmtree(index_dir)

    store = VectorStore(index_dir, embedder.dimension)

    manifest = store.load_manifest()
    if manifest is not None and manifest.get("embedding_fingerprint") != embedder.fingerprint:
        store.close()
        raise ValueError(
            f"Index at {index_dir} was built with a different embedding model "
            f"({manifest.get('embedding_model')}). Re-run with --rebuild to switch models."
        )

    seen_files: set[str] = set()
    try:
        for rel_path in iter_source_files(repo_path, settings):
            stats.files_scanned += 1
            file_key = str(rel_path).replace("\\", "/")
            seen_files.add(file_key)
            abs_path = repo_path / rel_path

            current_hash = file_hash(abs_path)
            if store.get_file_hash(file_key) == current_hash:
                stats.files_skipped_unchanged += 1
                continue

            store.purge_file(file_key)

            try:
                source = abs_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue

            chunker = chunker_factory(abs_path.suffix, settings)
            chunks = chunker.chunk_file(rel_path, source)
            # Normalize to the same forward-slash key used for file_hash
            # bookkeeping below - on Windows, str(Path) would otherwise
            # produce backslashes here, desyncing purge/removed-file lookups
            # from the sqlite-stored file_path values.
            for c in chunks:
                c.file_path = file_key
            if chunks:
                vectors = embedder.encode([c.embed_text for c in chunks])
                store.add_chunks(chunks, vectors)
                stats.chunks_written += len(chunks)

            store.record_file_hash(file_key, current_hash)
            stats.files_indexed += 1

        removed = store.indexed_file_paths() - seen_files
        for file_key in removed:
            store.purge_file(file_key)
            stats.files_removed += 1

        store.save(build_manifest(embedder, settings))
    finally:
        store.close()

    return stats

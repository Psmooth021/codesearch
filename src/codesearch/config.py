"""Central configuration for indexing, chunking, and embedding.

A single Settings instance is threaded through the indexer/search/eval
code paths instead of scattering magic numbers across modules.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

# Chosen empirically, not by assumption: `codesearch eval` was run against
# both this general-purpose sentence embedding model and a model fine-tuned
# on CodeSearchNet specifically for code search (BASELINE_EMBEDDING_MODEL,
# below). The code-specific model measurably lost - it's tuned for
# code<->docstring similarity, not for the "how does X work"-style natural-
# language questions this tool's queries actually look like. See README
# "Design Decisions" for the numbers. Also loads as a plain
# sentence-transformers model with no trust_remote_code (jina-embeddings-v2
# -base-code was evaluated first but its custom modeling code breaks under
# transformers>=5).
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BASELINE_EMBEDDING_MODEL = "flax-sentence-embeddings/st-codesearch-distilroberta-base"

INDEX_DIR_NAME = ".codesearch"
INDEX_FILE_NAME = "index.faiss"
METADATA_DB_NAME = "metadata.db"
MANIFEST_FILE_NAME = "manifest.json"

BUILTIN_EXCLUDE_DIRS = {
    ".git",
    ".codesearch",
    "node_modules",
    "venv",
    ".venv",
    "__pycache__",
    "dist",
    "build",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "eval_repos",
    ".tox",
    "egg-info",
}

BUILTIN_EXCLUDE_EXTENSIONS = {
    ".pyc",
    ".pyo",
    ".so",
    ".dll",
    ".dylib",
    ".exe",
    ".bin",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".ico",
    ".pdf",
    ".zip",
    ".tar",
    ".gz",
    ".lock",
}

# extension -> chunker language key, consumed by chunking/registry.py
LANGUAGE_EXTENSIONS = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
}


class Settings(BaseModel):
    model_config = ConfigDict(frozen=True)

    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    embedding_batch_size: int = 32

    # Chunking caps, in whitespace-delimited "tokens" (a cheap proxy for
    # model tokens - good enough to bound chunk size without pulling in
    # a full tokenizer for chunking decisions).
    max_chunk_tokens: int = 2000
    module_level_window_tokens: int = 400
    module_level_window_overlap_tokens: int = 50
    fallback_window_lines: int = 60
    fallback_window_overlap_lines: int = 10
    context_header_import_lines: int = 10

    default_k: int = 5

    excluded_dirs: set[str] = Field(default_factory=lambda: set(BUILTIN_EXCLUDE_DIRS))
    excluded_extensions: set[str] = Field(default_factory=lambda: set(BUILTIN_EXCLUDE_EXTENSIONS))

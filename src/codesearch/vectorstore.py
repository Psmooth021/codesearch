"""FAISS IndexFlatIP + a sqlite metadata sidecar, keyed by a shared int id.

Vectors are L2-normalized before insertion (see embedding.py), so inner
product on IndexFlatIP is equivalent to cosine similarity. Wrapped in
IndexIDMap2 so chunks can be removed by id (needed for incremental
indexing: a changed/deleted file's old chunks get purged and, if changed,
replaced - see indexer.py) rather than requiring a full rebuild.

Rejected chromadb: it hides the exact indexing decisions this project
exists to showcase. Rejected sqlite-vec: smaller ecosystem, adds SQL
extension-loading complexity for no benefit at this project's scale.
Rejected an approximate index (IVF/HNSW): at a few thousand files -> tens
of thousands of chunks, exact brute-force search is tens of milliseconds,
so approximate search would trade exactness for speed this tool doesn't
need. At ~1M+ vectors, IndexHNSWFlat or IndexIVFPQ would be the right call.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import faiss
import numpy as np

from codesearch.chunking.base import Chunk
from codesearch.config import INDEX_FILE_NAME, MANIFEST_FILE_NAME, METADATA_DB_NAME

_SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY,
    file_path TEXT NOT NULL,
    start_line INTEGER NOT NULL,
    end_line INTEGER NOT NULL,
    symbol_name TEXT NOT NULL,
    qualified_name TEXT NOT NULL,
    kind TEXT NOT NULL,
    language TEXT NOT NULL,
    code_text TEXT NOT NULL,
    truncated INTEGER NOT NULL,
    content_hash TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_file_path ON chunks(file_path);

CREATE TABLE IF NOT EXISTS files (
    file_path TEXT PRIMARY KEY,
    file_hash TEXT NOT NULL
);
"""


@dataclass
class SearchHit:
    chunk_id: int
    score: float
    file_path: str
    start_line: int
    end_line: int
    symbol_name: str
    qualified_name: str
    kind: str
    language: str
    code_text: str
    truncated: bool


class VectorStore:
    """Owns the FAISS index + sqlite metadata db for one indexed repo,
    both rooted at `index_dir` (normally `<repo>/.codesearch/`)."""

    def __init__(self, index_dir: Path, dimension: int) -> None:
        self.index_dir = index_dir
        self.dimension = dimension
        self.index_path = index_dir / INDEX_FILE_NAME
        self.db_path = index_dir / METADATA_DB_NAME
        self.manifest_path = index_dir / MANIFEST_FILE_NAME

        index_dir.mkdir(parents=True, exist_ok=True)
        self._index = self._load_or_create_index()
        self._conn = sqlite3.connect(self.db_path)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def _load_or_create_index(self) -> faiss.IndexIDMap2:
        if self.index_path.exists():
            return cast(faiss.IndexIDMap2, faiss.read_index(str(self.index_path)))
        base = faiss.IndexFlatIP(self.dimension)
        return faiss.IndexIDMap2(base)

    def add_chunks(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if not chunks:
            return
        assert vectors.shape[0] == len(chunks)
        cursor = self._conn.cursor()
        ids = []
        for chunk in chunks:
            cursor.execute(
                """INSERT INTO chunks
                   (file_path, start_line, end_line, symbol_name, qualified_name,
                    kind, language, code_text, truncated, content_hash)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    chunk.file_path,
                    chunk.start_line,
                    chunk.end_line,
                    chunk.symbol_name,
                    chunk.qualified_name,
                    chunk.kind,
                    chunk.language,
                    chunk.code_text,
                    int(chunk.truncated),
                    chunk.content_hash,
                ),
            )
            ids.append(cursor.lastrowid)
        self._conn.commit()
        self._index.add_with_ids(vectors, np.array(ids, dtype="int64"))

    def purge_file(self, file_path: str) -> None:
        """Remove all chunks (sqlite rows + FAISS vectors) for one file -
        used before re-indexing a changed file, or dropping a deleted one."""
        cursor = self._conn.cursor()
        cursor.execute("SELECT id FROM chunks WHERE file_path = ?", (file_path,))
        ids = [row[0] for row in cursor.fetchall()]
        if ids:
            # faiss-cpu's type stubs only declare IDSelector here, but a
            # 1D int64 array is accepted and auto-wrapped at runtime.
            self._index.remove_ids(np.array(ids, dtype="int64"))  # type: ignore[arg-type]
            cursor.execute("DELETE FROM chunks WHERE file_path = ?", (file_path,))
            self._conn.commit()
        cursor.execute("DELETE FROM files WHERE file_path = ?", (file_path,))
        self._conn.commit()

    def record_file_hash(self, file_path: str, file_hash: str) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO files (file_path, file_hash) VALUES (?, ?)",
            (file_path, file_hash),
        )
        self._conn.commit()

    def get_file_hash(self, file_path: str) -> str | None:
        row = self._conn.execute(
            "SELECT file_hash FROM files WHERE file_path = ?", (file_path,)
        ).fetchone()
        return row[0] if row else None

    def indexed_file_paths(self) -> set[str]:
        rows = self._conn.execute("SELECT file_path FROM files").fetchall()
        return {row[0] for row in rows}

    def chunk_count(self) -> int:
        return self._index.ntotal

    def search(self, query_vector: np.ndarray, k: int) -> list[SearchHit]:
        if self._index.ntotal == 0:
            return []
        query = query_vector.reshape(1, -1)
        scores, ids = self._index.search(query, min(k, self._index.ntotal))
        hits: list[SearchHit] = []
        cursor = self._conn.cursor()
        for score, chunk_id in zip(scores[0], ids[0], strict=True):
            if chunk_id == -1:
                continue
            row = cursor.execute(
                """SELECT file_path, start_line, end_line, symbol_name, qualified_name,
                          kind, language, code_text, truncated
                   FROM chunks WHERE id = ?""",
                (int(chunk_id),),
            ).fetchone()
            if row is None:
                continue
            hits.append(
                SearchHit(
                    chunk_id=int(chunk_id),
                    score=float(score),
                    file_path=row[0],
                    start_line=row[1],
                    end_line=row[2],
                    symbol_name=row[3],
                    qualified_name=row[4],
                    kind=row[5],
                    language=row[6],
                    code_text=row[7],
                    truncated=bool(row[8]),
                )
            )
        return hits

    def save(self, manifest: dict) -> None:
        faiss.write_index(self._index, str(self.index_path))
        self.manifest_path.write_text(json.dumps(manifest, indent=2))

    def load_manifest(self) -> dict | None:
        if not self.manifest_path.exists():
            return None
        return json.loads(self.manifest_path.read_text())

    def close(self) -> None:
        self._conn.close()

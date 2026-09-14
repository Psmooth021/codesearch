"""Naive fixed-window chunker.

Used for two purposes:
  1. Any file type without a dedicated AST-aware chunker (markdown, json,
     yaml, etc.) still gets indexed.
  2. The eval harness's `--baseline naive` mode reruns AST-supported
     languages through this exact chunker to produce an apples-to-apples
     comparison against the AST-aware chunkers.
"""

from __future__ import annotations

from pathlib import Path

from codesearch.chunking.base import Chunk
from codesearch.chunking.text_utils import line_windows
from codesearch.config import Settings
from codesearch.utils.hashing import content_hash


class FallbackChunker:
    language = "text"

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()

    def chunk_file(self, file_path: Path, source: str) -> list[Chunk]:
        windows = line_windows(
            source,
            1,
            self.settings.fallback_window_lines,
            self.settings.fallback_window_overlap_lines,
        )
        chunks: list[Chunk] = []
        for chunk_no, (start_line, end_line, text) in enumerate(windows, start=1):
            if not text.strip():
                continue
            symbol_name = f"{file_path.name}:window{chunk_no}"
            chunks.append(
                Chunk(
                    file_path=str(file_path),
                    start_line=start_line,
                    end_line=end_line,
                    symbol_name=symbol_name,
                    qualified_name=symbol_name,
                    kind="fallback_window",
                    code_text=text,
                    embed_text=f"# {file_path}\n{text}",
                    language=self.language,
                    content_hash=content_hash(text),
                )
            )
        return chunks

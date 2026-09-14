"""Shared chunk representation and chunker interface.

Every language-specific chunker (and the naive fallback) produces a list
of `Chunk` objects. A chunk's `embed_text` is what actually gets embedded
(it may carry a synthesized context header); `start_line`/`end_line` always
point at the real source range so results can be cited precisely.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol


@dataclass
class Chunk:
    file_path: str
    start_line: int
    end_line: int
    symbol_name: str
    qualified_name: str
    kind: str  # "function" | "method" | "class" | "module_block" | "fallback_window"
    code_text: str
    embed_text: str
    language: str
    truncated: bool = False
    content_hash: str = ""
    extra: dict = field(default_factory=dict)


class Chunker(Protocol):
    """A Chunker turns one file's source text into a list of Chunks."""

    language: str

    def chunk_file(self, file_path: Path, source: str) -> list[Chunk]: ...

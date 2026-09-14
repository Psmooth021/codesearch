"""AST-aware chunker for Markdown, built on tree-sitter's markdown grammar.

Documentation files (READMEs, docs/*.md) were previously falling through to
FallbackChunker's naive 60-line windows - the exact failure mode the rest of
this project argues against for code. This chunker instead splits on the
document's actual heading structure, so each chunk is one complete section
(heading + its own content, not its subsections' content), with a breadcrumb
of ancestor headings for context - e.g. a chunk under "## Installation"
nested under "# Getting Started" carries that path, so the embedding isn't
just "here are some install steps" with no idea what project it's for.

Deliberately not regex/line-scanning: tree-sitter's markdown grammar already
distinguishes a `#` that starts a heading from a `#` inside a fenced code
block (e.g. a Python comment in an example snippet), which a naive line
scanner would have to reimplement and would be easy to get wrong.
"""

from __future__ import annotations

from pathlib import Path

from tree_sitter import Node
from tree_sitter_language_pack import get_parser

from codesearch.chunking.base import Chunk
from codesearch.chunking.text_utils import line_windows, node_text, token_count
from codesearch.config import Settings
from codesearch.utils.hashing import content_hash

_PARSER = get_parser("markdown")

_BREADCRUMB_SEP = " > "


def _heading_level(atx_heading: Node) -> int:
    marker = atx_heading.named_children[0]
    # marker.type is "atx_h1_marker".."atx_h6_marker"
    return int(marker.type[5])


def _heading_text(atx_heading: Node, source: bytes) -> str:
    for child in atx_heading.named_children:
        if child.type == "inline":
            return node_text(child, source).strip()
    return ""


class MarkdownChunker:
    language = "markdown"

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()

    def chunk_file(self, file_path: Path, source: str) -> list[Chunk]:
        source_bytes = source.encode("utf-8")
        tree = _PARSER.parse(source_bytes)
        root = tree.root_node

        chunks: list[Chunk] = []
        for section in root.named_children:
            if section.type == "section":
                self._walk_section(section, source_bytes, file_path, [], chunks)
        return chunks

    def _walk_section(
        self,
        section: Node,
        source: bytes,
        file_path: Path,
        heading_stack: list[str],
        chunks: list[Chunk],
    ) -> None:
        named = list(section.named_children)
        heading_node: Node | None = None
        rest = named
        if named and named[0].type == "atx_heading":
            heading_node = named[0]
            rest = named[1:]

        if heading_node is not None:
            level = _heading_level(heading_node)
            title = _heading_text(heading_node, source)
            new_stack = [*heading_stack, title]
        else:
            level = 0
            title = None
            new_stack = heading_stack

        own_content: list[Node] = []
        nested_sections: list[Node] = []
        for child in rest:
            if child.type == "section":
                nested_sections.append(child)
            else:
                own_content.append(child)

        if heading_node is not None or own_content:
            chunks.extend(
                self._build_chunks(
                    section, heading_node, own_content, source, file_path, new_stack, title, level
                )
            )

        for nested in nested_sections:
            self._walk_section(nested, source, file_path, new_stack, chunks)

    def _build_chunks(
        self,
        section: Node,
        heading_node: Node | None,
        own_content: list[Node],
        source: bytes,
        file_path: Path,
        heading_stack: list[str],
        title: str | None,
        level: int,
    ) -> list[Chunk]:
        start_byte = section.start_byte
        start_line = section.start_point.row + 1
        if own_content:
            end_byte = own_content[-1].end_byte
            end_line = own_content[-1].end_point.row + 1
        elif heading_node is not None:
            end_byte = heading_node.end_byte
            end_line = heading_node.end_point.row + 1
        else:
            return []

        text = source[start_byte:end_byte].decode("utf-8", errors="replace")
        if not text.strip():
            return []

        breadcrumb = _BREADCRUMB_SEP.join(heading_stack)
        symbol_name = title or "intro"
        qualified_name = breadcrumb or file_path.name
        header = f"# file: {file_path}\n" + (f"# section: {breadcrumb}\n" if breadcrumb else "")

        if token_count(text) <= self.settings.max_chunk_tokens:
            return [
                Chunk(
                    file_path=str(file_path),
                    start_line=start_line,
                    end_line=end_line,
                    symbol_name=symbol_name,
                    qualified_name=qualified_name,
                    kind="doc_section",
                    code_text=text,
                    embed_text=header + text,
                    language=self.language,
                    content_hash=content_hash(text),
                    extra={"heading_level": level},
                )
            ]

        # Oversized section (a long prose block with no subheadings to
        # split on): fall back to line-windows, each tagged truncated=True
        # and still carrying the section's breadcrumb, so nothing in a long
        # section is silently dropped from the index.
        windows = line_windows(
            text,
            start_line,
            self.settings.fallback_window_lines,
            self.settings.fallback_window_overlap_lines,
        )
        return [
            Chunk(
                file_path=str(file_path),
                start_line=w_start,
                end_line=w_end,
                symbol_name=symbol_name,
                qualified_name=qualified_name,
                kind="doc_section",
                code_text=w_text,
                embed_text=header + w_text,
                language=self.language,
                truncated=True,
                content_hash=content_hash(w_text),
                extra={"heading_level": level},
            )
            for w_start, w_end, w_text in windows
        ]

"""AST-aware chunker for Markdown, built on tree-sitter's markdown grammar.

Documentation files (READMEs, docs/*.md) were previously falling through to
FallbackChunker's naive 60-line windows - the exact failure mode the rest of
this project argues against for code. This chunker instead splits on the
document's actual heading structure, so each chunk is one complete section
(heading + its own content, not its subsections' content), with a breadcrumb
of ancestor headings for context - e.g. a chunk under "## Installation"
nested under "# Getting Started" carries that path, so the embedding isn't
just "here are some install steps" with no idea what project it's for.

Deliberately not regex/line-scanning for the *structural* split: tree-sitter's
markdown grammar already distinguishes a `#` that starts a heading from a `#`
inside a fenced code block (e.g. a Python comment in an example snippet),
which a naive line scanner would have to reimplement and would be easy to
get wrong. Regex is used for one narrower, secondary job: recognizing raw
HTML heading tags (`<h1>...</h1>`) inside a block the grammar treats as
opaque HTML - see `_HTML_HEADING_RE` below.
"""

from __future__ import annotations

import re
from pathlib import Path

from tree_sitter import Node
from tree_sitter_language_pack import get_parser

from codesearch.chunking.base import Chunk
from codesearch.chunking.text_utils import line_windows, node_text, token_count
from codesearch.config import Settings
from codesearch.utils.hashing import content_hash

_PARSER = get_parser("markdown")

_BREADCRUMB_SEP = " > "

# Matches a single-block raw HTML heading like `<h1 align="center">Title</h1>`
# or `<h2>Title with <strong>bold</strong></h2>` - common in READMEs that use
# a centered logo/title block instead of a markdown `#` heading. Tree-sitter's
# markdown grammar treats this as an opaque html_block, not a heading, so it
# never becomes a `section` boundary on its own; this regex is what lets such
# a block still act as one.
_HTML_HEADING_RE = re.compile(r"<h([1-6])\b[^>]*>(.*?)</h\1\s*>", re.IGNORECASE | re.DOTALL)
_HTML_TAG_RE = re.compile(r"<[^>]+>")


def _heading_text(atx_heading: Node, source: bytes) -> str:
    for child in atx_heading.named_children:
        if child.type == "inline":
            return node_text(child, source).strip()
    return ""


def _html_heading_title(node: Node, source: bytes) -> str | None:
    """A node counts as an HTML pseudo-heading only if its *entire* text is
    one heading tag (open, content, close) - not a node that merely
    contains one alongside other prose. A single-line `<h1>Title</h1>`
    doesn't qualify as tree-sitter's own `html_block` node type under
    CommonMark's HTML block rules (its "type 7" rule requires an opening or
    closing tag alone on the line, not open+content+close together) - it
    parses as a plain `paragraph` instead - so this matches by content
    across any node type rather than gating on `html_block`."""
    text = node_text(node, source).strip()
    match = _HTML_HEADING_RE.fullmatch(text)
    if not match:
        return None
    title = _HTML_TAG_RE.sub("", match.group(2)).strip()
    return title or None


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
            title = _heading_text(heading_node, source)
            new_stack = [*heading_stack, title]
        else:
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
                self._build_section_chunks(
                    section, heading_node, own_content, source, file_path, new_stack, title
                )
            )

        for nested in nested_sections:
            self._walk_section(nested, source, file_path, new_stack, chunks)

    def _build_section_chunks(
        self,
        section: Node,
        heading_node: Node | None,
        own_content: list[Node],
        source: bytes,
        file_path: Path,
        heading_stack: list[str],
        title: str | None,
    ) -> list[Chunk]:
        """Emit one chunk for the section's ATX-heading region, plus one
        more per raw HTML heading tag found within its own content (so a
        README's HTML `<h1>` title block still gets its own clean chunk
        instead of being buried in an undifferentiated preamble blob)."""
        html_breaks = [
            (i, htitle)
            for i, node in enumerate(own_content)
            if (htitle := _html_heading_title(node, source)) is not None
        ]

        if not html_breaks:
            end_byte, end_line = self._span_end(heading_node, own_content)
            return self._chunks_for_span(
                section.start_byte,
                section.start_point.row + 1,
                end_byte,
                end_line,
                source,
                file_path,
                heading_stack,
                title,
            )

        chunks: list[Chunk] = []

        # Region before the first HTML heading: the ATX heading (if any)
        # plus any leading content, same as the no-html-heading case.
        first_idx = html_breaks[0][0]
        pre_nodes = own_content[:first_idx]
        if heading_node is not None or pre_nodes:
            pre_end_byte, pre_end_line = self._span_end(heading_node, pre_nodes)
            chunks.extend(
                self._chunks_for_span(
                    section.start_byte,
                    section.start_point.row + 1,
                    pre_end_byte,
                    pre_end_line,
                    source,
                    file_path,
                    heading_stack,
                    title,
                )
            )

        # One chunk per HTML heading, spanning from that heading's own node
        # to just before the next HTML heading (or the end of own_content).
        for j, (idx, htitle) in enumerate(html_breaks):
            next_idx = html_breaks[j + 1][0] if j + 1 < len(html_breaks) else len(own_content)
            segment_nodes = own_content[idx:next_idx]
            seg_start_byte = segment_nodes[0].start_byte
            seg_start_line = segment_nodes[0].start_point.row + 1
            seg_end_byte = segment_nodes[-1].end_byte
            seg_end_line = segment_nodes[-1].end_point.row + 1
            chunks.extend(
                self._chunks_for_span(
                    seg_start_byte,
                    seg_start_line,
                    seg_end_byte,
                    seg_end_line,
                    source,
                    file_path,
                    [*heading_stack, htitle],
                    htitle,
                )
            )

        return chunks

    @staticmethod
    def _span_end(
        heading_node: Node | None, content_nodes: list[Node]
    ) -> tuple[int | None, int | None]:
        if content_nodes:
            return content_nodes[-1].end_byte, content_nodes[-1].end_point.row + 1
        if heading_node is not None:
            return heading_node.end_byte, heading_node.end_point.row + 1
        return None, None

    def _chunks_for_span(
        self,
        start_byte: int,
        start_line: int,
        end_byte: int | None,
        end_line: int | None,
        source: bytes,
        file_path: Path,
        heading_stack: list[str],
        title: str | None,
    ) -> list[Chunk]:
        if end_byte is None or end_line is None:
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
            )
            for w_start, w_end, w_text in windows
        ]

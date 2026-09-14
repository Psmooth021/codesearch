"""AST-aware chunker for Python source, built on tree-sitter.

Produces one chunk per top-level function, one chunk per method, and one
summary chunk per class (signature + docstring + method list) so a query
about "what does class X do" matches at class granularity instead of being
buried inside one arbitrary method. Anything not captured by those (bare
module-level statements) is grouped into sliding-window chunks so nothing
in the file is silently dropped.
"""

from __future__ import annotations

from pathlib import Path

from tree_sitter import Node
from tree_sitter_language_pack import get_parser

from codesearch.chunking.base import Chunk
from codesearch.chunking.text_utils import line_windows, token_count
from codesearch.config import Settings
from codesearch.utils.hashing import content_hash

_PARSER = get_parser("python")


def _node_text(node: Node, source: bytes) -> str:
    return source[node.start_byte : node.end_byte].decode("utf-8", errors="replace")


def _line_range(node: Node) -> tuple[int, int]:
    return node.start_point.row + 1, node.end_point.row + 1


def _docstring(body: Node, source: bytes) -> str:
    if body.named_child_count == 0:
        return ""
    first = body.named_children[0]
    if first.type != "string":
        return ""
    text = _node_text(first, source)
    return text.strip("\"' \t\n")


def _unwrap_decorated(node: Node) -> tuple[Node, list[Node]]:
    """decorated_definition -> (function_definition|class_definition, [decorator, ...])."""
    if node.type != "decorated_definition":
        return node, []
    decorators = [c for c in node.named_children if c.type == "decorator"]
    inner = next(
        c for c in node.named_children if c.type in ("function_definition", "class_definition")
    )
    return inner, decorators


def _collect_imports(root: Node, source: bytes, limit: int) -> list[str]:
    imports: list[str] = []
    for child in root.named_children:
        if child.type in ("import_statement", "import_from_statement"):
            imports.append(_node_text(child, source).strip())
        if len(imports) >= limit:
            break
    return imports


def _context_header(file_path: Path, imports: list[str], enclosing_class: str | None) -> str:
    lines = [f"# file: {file_path}"]
    if enclosing_class:
        lines.append(f"# class: {enclosing_class}")
    if imports:
        lines.append("# imports:")
        lines.extend(f"#   {imp}" for imp in imports)
    return "\n".join(lines) + "\n"


def _split_oversized(node: Node, source: bytes, settings: Settings) -> list[Node]:
    """Split an oversized node at nested function/class boundaries, if any exist."""
    body = node.child_by_field_name("body")
    if body is None:
        return [node]
    nested = [
        c for c in body.named_children if c.type in ("function_definition", "class_definition")
    ]
    if not nested:
        return [node]
    return nested


class PythonChunker:
    language = "python"

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()

    def chunk_file(self, file_path: Path, source: str) -> list[Chunk]:
        source_bytes = source.encode("utf-8")
        tree = _PARSER.parse(source_bytes)
        root = tree.root_node

        imports = _collect_imports(root, source_bytes, self.settings.context_header_import_lines)
        chunks: list[Chunk] = []
        leftover_nodes: list[Node] = []

        for node in root.named_children:
            actual, decorators = _unwrap_decorated(node)
            if actual.type == "function_definition":
                chunks.extend(
                    self._function_chunks(
                        actual, decorators, source_bytes, file_path, imports, None
                    )
                )
            elif actual.type == "class_definition":
                chunks.extend(
                    self._class_chunks(actual, decorators, source_bytes, file_path, imports)
                )
            elif node.type in ("import_statement", "import_from_statement", "comment"):
                continue
            else:
                leftover_nodes.append(node)

        chunks.extend(self._module_level_chunks(leftover_nodes, source_bytes, file_path))
        return chunks

    def _function_chunks(
        self,
        fn_node: Node,
        decorators: list[Node],
        source: bytes,
        file_path: Path,
        imports: list[str],
        enclosing_class: str | None,
    ) -> list[Chunk]:
        name_node = fn_node.child_by_field_name("name")
        name = _node_text(name_node, source) if name_node else "<anonymous>"
        qualified = f"{enclosing_class}.{name}" if enclosing_class else name
        kind = "method" if enclosing_class else "function"
        header = _context_header(file_path, imports, enclosing_class)
        decorator_text = "\n".join(_node_text(d, source) for d in decorators)
        full_text = _node_text(fn_node, source)
        start_line, end_line = _line_range(fn_node)

        if token_count(full_text) <= self.settings.max_chunk_tokens:
            body_text = (decorator_text + "\n" + full_text) if decorator_text else full_text
            return [
                Chunk(
                    file_path=str(file_path),
                    start_line=start_line,
                    end_line=end_line,
                    symbol_name=name,
                    qualified_name=qualified,
                    kind=kind,
                    code_text=body_text,
                    embed_text=header + body_text,
                    language=self.language,
                    content_hash=content_hash(body_text),
                )
            ]

        # Oversized: prefer splitting at nested function/class boundaries.
        nested = _split_oversized(fn_node, source, self.settings)
        if len(nested) > 1:
            chunks: list[Chunk] = []
            for part in nested:
                p_start, p_end = _line_range(part)
                part_text = _node_text(part, source)
                chunks.append(
                    Chunk(
                        file_path=str(file_path),
                        start_line=p_start,
                        end_line=p_end,
                        symbol_name=name,
                        qualified_name=qualified,
                        kind=kind,
                        code_text=part_text,
                        embed_text=header + part_text,
                        language=self.language,
                        content_hash=content_hash(part_text),
                    )
                )
            return chunks

        # No nested boundaries to split on: fall back to a line-based
        # sliding window over the function body, each window tagged
        # truncated=True since no single chunk represents the whole function.
        return self._window_lines(
            full_text, start_line, file_path, header, name, qualified, kind
        )

    def _window_lines(
        self,
        text: str,
        base_start_line: int,
        file_path: Path,
        header: str,
        name: str,
        qualified: str,
        kind: str,
    ) -> list[Chunk]:
        windows = line_windows(
            text,
            base_start_line,
            self.settings.fallback_window_lines,
            self.settings.fallback_window_overlap_lines,
        )
        return [
            Chunk(
                file_path=str(file_path),
                start_line=start_line,
                end_line=end_line,
                symbol_name=name,
                qualified_name=qualified,
                kind=kind,
                code_text=window_text,
                embed_text=header + window_text,
                language=self.language,
                truncated=True,
                content_hash=content_hash(window_text),
            )
            for start_line, end_line, window_text in windows
        ]

    def _class_chunks(
        self,
        cls_node: Node,
        decorators: list[Node],
        source: bytes,
        file_path: Path,
        imports: list[str],
    ) -> list[Chunk]:
        name_node = cls_node.child_by_field_name("name")
        class_name = _node_text(name_node, source) if name_node else "<anonymous>"
        body = cls_node.child_by_field_name("body")
        docstring = _docstring(body, source) if body else ""

        method_summaries: list[str] = []
        chunks: list[Chunk] = []
        if body:
            for member in body.named_children:
                actual, method_decorators = _unwrap_decorated(member)
                if actual.type != "function_definition":
                    continue
                mname_node = actual.child_by_field_name("name")
                mname = _node_text(mname_node, source) if mname_node else "<anonymous>"
                params_node = actual.child_by_field_name("parameters")
                params = _node_text(params_node, source) if params_node else "()"
                method_body = actual.child_by_field_name("body")
                mdoc = _docstring(method_body, source) if method_body else ""
                summary = f"{mname}{params}"
                if mdoc:
                    summary += f" - {mdoc.splitlines()[0]}"
                method_summaries.append(summary)
                chunks.extend(
                    self._function_chunks(
                        actual, method_decorators, source, file_path, imports, class_name
                    )
                )

        class_start, class_end = _line_range(cls_node)
        bases_node = cls_node.child_by_field_name("superclasses")
        bases = _node_text(bases_node, source) if bases_node else ""
        signature = f"class {class_name}{bases}:"
        summary_text = signature
        if docstring:
            summary_text += f"\n    \"\"\"{docstring}\"\"\""
        if method_summaries:
            summary_text += "\nMethods:\n" + "\n".join(f"  - {m}" for m in method_summaries)

        header = _context_header(file_path, imports, None)
        chunks.append(
            Chunk(
                file_path=str(file_path),
                start_line=class_start,
                end_line=class_end,
                symbol_name=class_name,
                qualified_name=class_name,
                kind="class",
                code_text=summary_text,
                embed_text=header + summary_text,
                language=self.language,
                content_hash=content_hash(summary_text),
            )
        )
        return chunks

    def _module_level_chunks(
        self, nodes: list[Node], source: bytes, file_path: Path
    ) -> list[Chunk]:
        if not nodes:
            return []
        chunks: list[Chunk] = []
        window_tokens = self.settings.module_level_window_tokens
        overlap_tokens = self.settings.module_level_window_overlap_tokens

        buffer_nodes: list[Node] = []
        buffer_tokens = 0

        def flush() -> None:
            nonlocal buffer_nodes, buffer_tokens
            if not buffer_nodes:
                return
            start_line = buffer_nodes[0].start_point.row + 1
            end_line = buffer_nodes[-1].end_point.row + 1
            text = "\n".join(_node_text(n, source) for n in buffer_nodes)
            symbol = f"{file_path.name}:module:{start_line}-{end_line}"
            chunks.append(
                Chunk(
                    file_path=str(file_path),
                    start_line=start_line,
                    end_line=end_line,
                    symbol_name=symbol,
                    qualified_name=symbol,
                    kind="module_block",
                    code_text=text,
                    embed_text=f"# file: {file_path}\n{text}",
                    language=self.language,
                    content_hash=content_hash(text),
                )
            )

        for node in nodes:
            text = _node_text(node, source)
            buffer_nodes.append(node)
            buffer_tokens += token_count(text)
            if buffer_tokens >= window_tokens:
                flush()
                # keep a small overlap tail for continuity across windows
                overlap_nodes: list[Node] = []
                overlap_count = 0
                for n in reversed(buffer_nodes):
                    overlap_count += token_count(_node_text(n, source))
                    overlap_nodes.insert(0, n)
                    if overlap_count >= overlap_tokens:
                        break
                buffer_nodes = overlap_nodes
                buffer_tokens = overlap_count if buffer_nodes else 0
        flush()
        return chunks

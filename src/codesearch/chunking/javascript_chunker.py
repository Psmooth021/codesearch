"""AST-aware chunker for JavaScript/TypeScript/TSX, built on tree-sitter.

Shares the same chunk granularity as the Python chunker (one chunk per
function/method, one summary chunk per class), adapted to JS/TS syntax:
function declarations, const/let-bound arrow functions, class declarations
with method_definition members, and (TS/TSX only) interface declarations,
which get the same "signature + member list" summary treatment as classes.
"""

from __future__ import annotations

from pathlib import Path

from tree_sitter import Node
from tree_sitter_language_pack import get_parser

from codesearch.chunking.base import Chunk
from codesearch.chunking.text_utils import line_windows, token_count
from codesearch.config import Settings
from codesearch.utils.hashing import content_hash

_PARSERS = {
    "javascript": get_parser("javascript"),
    "typescript": get_parser("typescript"),
    "tsx": get_parser("tsx"),
}

_FUNCTION_VALUE_TYPES = ("arrow_function", "function_expression")
_DECLARATION_TYPES = (
    "function_declaration",
    "class_declaration",
    "lexical_declaration",
    "variable_declaration",
    "interface_declaration",
    "type_alias_declaration",
)


def _node_text(node: Node, source: bytes) -> str:
    return source[node.start_byte : node.end_byte].decode("utf-8", errors="replace")


def _line_range(node: Node) -> tuple[int, int]:
    return node.start_point.row + 1, node.end_point.row + 1


def _leading_comment(node: Node, source: bytes) -> str:
    prev = node.prev_sibling
    is_adjacent = prev is not None and prev.end_point.row == node.start_point.row - 1
    if prev is not None and prev.type == "comment" and is_adjacent:
        return _node_text(prev, source)
    return ""


def _unwrap_export(node: Node) -> Node:
    """export_statement -> its inner declaration, if it wraps one we handle."""
    if node.type != "export_statement":
        return node
    for child in node.named_children:
        if child.type in _DECLARATION_TYPES:
            return child
    return node


def _collect_imports(root: Node, source: bytes, limit: int) -> list[str]:
    imports: list[str] = []
    for child in root.named_children:
        if child.type == "import_statement":
            imports.append(_node_text(child, source).strip())
        if len(imports) >= limit:
            break
    return imports


def _context_header(file_path: Path, imports: list[str], enclosing_class: str | None) -> str:
    lines = [f"// file: {file_path}"]
    if enclosing_class:
        lines.append(f"// class: {enclosing_class}")
    if imports:
        lines.append("// imports:")
        lines.extend(f"//   {imp}" for imp in imports)
    return "\n".join(lines) + "\n"


class JavaScriptChunker:
    """Handles .js/.jsx/.mjs (language="javascript"), .ts (language="typescript"),
    and .tsx (language="tsx") - the grammars differ slightly but share this
    same traversal logic."""

    def __init__(self, language: str = "javascript", settings: Settings | None = None) -> None:
        if language not in _PARSERS:
            raise ValueError(f"unsupported language: {language}")
        self.language = language
        self.parser = _PARSERS[language]
        self.settings = settings or Settings()

    def chunk_file(self, file_path: Path, source: str) -> list[Chunk]:
        source_bytes = source.encode("utf-8")
        tree = self.parser.parse(source_bytes)
        root = tree.root_node

        imports = _collect_imports(root, source_bytes, self.settings.context_header_import_lines)
        chunks: list[Chunk] = []
        leftover_nodes: list[Node] = []

        for node in root.named_children:
            comment = _leading_comment(node, source_bytes)
            actual = _unwrap_export(node)

            if actual.type == "function_declaration":
                chunks.extend(
                    self._function_chunk(actual, source_bytes, file_path, imports, None, comment)
                )
            elif actual.type == "class_declaration":
                chunks.extend(
                    self._class_chunks(actual, source_bytes, file_path, imports, comment)
                )
            elif actual.type == "interface_declaration":
                chunks.append(
                    self._interface_chunk(actual, source_bytes, file_path, imports, comment)
                )
            elif actual.type in ("lexical_declaration", "variable_declaration"):
                fn_chunks, remainder = self._declarator_chunks(
                    actual, source_bytes, file_path, imports, comment
                )
                chunks.extend(fn_chunks)
                if remainder is not None:
                    leftover_nodes.append(remainder)
            elif (assignment_fn := self._assignment_function_chunk(
                actual, source_bytes, file_path, imports, comment
            )) is not None:
                chunks.extend(assignment_fn)
            elif node.type in ("import_statement", "comment"):
                continue
            else:
                leftover_nodes.append(node)

        chunks.extend(self._module_level_chunks(leftover_nodes, source_bytes, file_path))
        return chunks

    def _function_chunk(
        self,
        fn_node: Node,
        source: bytes,
        file_path: Path,
        imports: list[str],
        enclosing_class: str | None,
        comment: str,
        override_name: str | None = None,
    ) -> list[Chunk]:
        name_node = fn_node.child_by_field_name("name")
        name = override_name or (_node_text(name_node, source) if name_node else "<anonymous>")
        qualified = f"{enclosing_class}.{name}" if enclosing_class else name
        kind = "method" if enclosing_class else "function"
        header = _context_header(file_path, imports, enclosing_class)
        code_text = _node_text(fn_node, source)
        if comment:
            code_text = comment + "\n" + code_text
        start_line, end_line = _line_range(fn_node)

        if token_count(code_text) <= self.settings.max_chunk_tokens:
            return [
                Chunk(
                    file_path=str(file_path),
                    start_line=start_line,
                    end_line=end_line,
                    symbol_name=name,
                    qualified_name=qualified,
                    kind=kind,
                    code_text=code_text,
                    embed_text=header + code_text,
                    language=self.language,
                    content_hash=content_hash(code_text),
                )
            ]

        # Oversized: prefer splitting at nested function/method/class boundaries.
        nested = self._nested_boundaries(fn_node)
        if nested:
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

        # No nested boundaries: fall back to a line-based sliding window,
        # each window tagged truncated=True since no single chunk
        # represents the whole function.
        windows = line_windows(
            code_text,
            start_line,
            self.settings.fallback_window_lines,
            self.settings.fallback_window_overlap_lines,
        )
        return [
            Chunk(
                file_path=str(file_path),
                start_line=w_start,
                end_line=w_end,
                symbol_name=name,
                qualified_name=qualified,
                kind=kind,
                code_text=w_text,
                embed_text=header + w_text,
                language=self.language,
                truncated=True,
                content_hash=content_hash(w_text),
            )
            for w_start, w_end, w_text in windows
        ]

    @staticmethod
    def _nested_boundaries(fn_node: Node) -> list[Node]:
        body = fn_node.child_by_field_name("body")
        if body is None:
            return []
        return [
            c
            for c in body.named_children
            if c.type in ("function_declaration", "class_declaration", "method_definition")
        ]

    def _assignment_function_chunk(
        self,
        node: Node,
        source: bytes,
        file_path: Path,
        imports: list[str],
        comment: str,
    ) -> list[Chunk] | None:
        """Handles the common CommonJS idiom `obj.method = function () {...}`
        (e.g. `app.set = function set(...) {...}`, `exports.foo = ...`,
        `Klass.prototype.bar = ...`) - a plain assignment_expression, not a
        function/class/const declaration, but just as much "a function" as
        any of those for search purposes."""
        if node.type != "expression_statement" or node.named_child_count != 1:
            return None
        assignment = node.named_children[0]
        if assignment.type != "assignment_expression":
            return None
        left = assignment.child_by_field_name("left")
        right = assignment.child_by_field_name("right")
        if left is None or right is None:
            return None
        if left.type != "member_expression" or right.type not in _FUNCTION_VALUE_TYPES:
            return None
        property_node = left.child_by_field_name("property")
        name = _node_text(property_node, source) if property_node else "<anonymous>"
        return self._function_chunk(
            node, source, file_path, imports, None, comment, override_name=name
        )

    def _declarator_chunks(
        self,
        decl_node: Node,
        source: bytes,
        file_path: Path,
        imports: list[str],
        comment: str,
    ) -> tuple[list[Chunk], Node | None]:
        """Split `const foo = () => {...}` (and siblings in the same
        statement) into function chunks; non-function declarators in the
        same statement are returned as a leftover node for module-level
        windowing."""
        function_chunks: list[Chunk] = []
        has_non_function = False
        for declarator in decl_node.named_children:
            if declarator.type != "variable_declarator":
                continue
            name_node = declarator.child_by_field_name("name")
            value_node = declarator.child_by_field_name("value")
            if value_node is not None and value_node.type in _FUNCTION_VALUE_TYPES:
                name = _node_text(name_node, source) if name_node else "<anonymous>"
                single_declarator = decl_node.named_child_count == 1
                function_chunks.extend(
                    self._function_chunk(
                        decl_node if single_declarator else value_node,
                        source,
                        file_path,
                        imports,
                        None,
                        comment,
                        override_name=name,
                    )
                )
            else:
                has_non_function = True
        remainder = decl_node if (has_non_function and not function_chunks) else None
        return function_chunks, remainder

    def _class_chunks(
        self,
        cls_node: Node,
        source: bytes,
        file_path: Path,
        imports: list[str],
        comment: str,
    ) -> list[Chunk]:
        name_node = cls_node.child_by_field_name("name")
        class_name = _node_text(name_node, source) if name_node else "<anonymous>"
        body = cls_node.child_by_field_name("body")

        method_summaries: list[str] = []
        chunks: list[Chunk] = []
        if body:
            for member in body.named_children:
                if member.type != "method_definition":
                    continue
                mname_node = member.child_by_field_name("name")
                mname = _node_text(mname_node, source) if mname_node else "<anonymous>"
                params_node = member.child_by_field_name("parameters")
                params = _node_text(params_node, source) if params_node else "()"
                method_summaries.append(f"{mname}{params}")
                member_comment = _leading_comment(member, source)
                chunks.extend(
                    self._function_chunk(
                        member, source, file_path, imports, class_name, member_comment
                    )
                )

        class_start, class_end = _line_range(cls_node)
        heritage = cls_node.child_by_field_name("heritage")
        heritage_text = f" {_node_text(heritage, source)}" if heritage else ""
        summary_text = f"class {class_name}{heritage_text} {{"
        if method_summaries:
            summary_text += "\n  // methods:\n" + "\n".join(f"  //   {m}" for m in method_summaries)
        summary_text += "\n}"
        if comment:
            summary_text = comment + "\n" + summary_text

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

    def _interface_chunk(
        self,
        iface_node: Node,
        source: bytes,
        file_path: Path,
        imports: list[str],
        comment: str,
    ) -> Chunk:
        name_node = iface_node.child_by_field_name("name")
        iface_name = _node_text(name_node, source) if name_node else "<anonymous>"
        body = iface_node.child_by_field_name("body")
        member_summaries = []
        if body:
            for member in body.named_children:
                member_summaries.append(_node_text(member, source).strip())

        start_line, end_line = _line_range(iface_node)
        summary_text = f"interface {iface_name} {{"
        if member_summaries:
            summary_text += "\n" + "\n".join(f"  {m}" for m in member_summaries)
        summary_text += "\n}"
        if comment:
            summary_text = comment + "\n" + summary_text

        header = _context_header(file_path, imports, None)
        return Chunk(
            file_path=str(file_path),
            start_line=start_line,
            end_line=end_line,
            symbol_name=iface_name,
            qualified_name=iface_name,
            kind="class",
            code_text=summary_text,
            embed_text=header + summary_text,
            language=self.language,
            content_hash=content_hash(summary_text),
        )

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
                    embed_text=f"// file: {file_path}\n{text}",
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

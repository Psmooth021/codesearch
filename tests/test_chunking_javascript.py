from pathlib import Path

from codesearch.chunking.javascript_chunker import JavaScriptChunker

FILE = Path("sample.js")


def chunk(source: str, language: str = "javascript"):
    return JavaScriptChunker(language).chunk_file(FILE, source)


def test_plain_function_declaration():
    src = "function add(a, b) {\n  return a + b;\n}\n"
    chunks = chunk(src)
    fns = [c for c in chunks if c.kind == "function"]
    assert len(fns) == 1
    assert fns[0].symbol_name == "add"
    assert fns[0].start_line == 1
    assert fns[0].end_line == 3


def test_exported_function_with_leading_comment():
    src = "// adds two numbers\nexport function add(a, b) {\n  return a + b;\n}\n"
    chunks = chunk(src)
    fn = next(c for c in chunks if c.kind == "function")
    assert "adds two numbers" in fn.code_text
    assert fn.symbol_name == "add"


def test_const_arrow_function():
    src = "const double = (x) => {\n  return x * 2;\n};\n"
    chunks = chunk(src)
    fns = [c for c in chunks if c.kind == "function"]
    assert len(fns) == 1
    assert fns[0].symbol_name == "double"


def test_exported_class_with_methods():
    src = (
        "export class Widget {\n"
        "  render() {\n"
        "    return this.name;\n"
        "  }\n\n"
        "  destroy() {\n"
        "    return null;\n"
        "  }\n"
        "}\n"
    )
    chunks = chunk(src)
    cls = next(c for c in chunks if c.kind == "class")
    assert cls.symbol_name == "Widget"
    methods = [c for c in chunks if c.kind == "method"]
    assert {m.qualified_name for m in methods} == {"Widget.render", "Widget.destroy"}


def test_export_default_class():
    src = "export default class Gadget {\n  run() {\n    return 1;\n  }\n}\n"
    chunks = chunk(src)
    cls = next(c for c in chunks if c.kind == "class")
    assert cls.symbol_name == "Gadget"


def test_method_leading_comment_captured():
    src = (
        "class Widget {\n"
        "  // renders the widget\n"
        "  render() {\n"
        "    return this.name;\n"
        "  }\n"
        "}\n"
    )
    chunks = chunk(src)
    method = next(c for c in chunks if c.kind == "method")
    assert "renders the widget" in method.code_text


def test_typescript_interface():
    src = "export interface Shape {\n  area(): number;\n  perimeter(): number;\n}\n"
    chunks = chunk(src, language="typescript")
    iface = next(c for c in chunks if c.kind == "class")
    assert iface.symbol_name == "Shape"
    assert "area(): number" in iface.code_text
    assert "perimeter(): number" in iface.code_text


def test_tsx_arrow_component():
    src = "export const Card = (props) => {\n  return props.title;\n};\n"
    chunks = chunk(src, language="tsx")
    fns = [c for c in chunks if c.kind == "function"]
    assert len(fns) == 1
    assert fns[0].symbol_name == "Card"


def test_commonjs_prototype_style_method_assignment():
    # e.g. Express's `app.set = function set(...) {...}` idiom - a plain
    # assignment, not a function/class/const declaration, but still a
    # function worth its own chunk.
    src = "app.set = function set(setting, val) {\n  return val;\n};\n"
    chunks = chunk(src)
    fns = [c for c in chunks if c.kind == "function"]
    assert len(fns) == 1
    assert fns[0].symbol_name == "set"
    assert "app.set" in fns[0].code_text


def test_exports_dot_assignment_function():
    src = "exports.foo = function foo() {\n  return 1;\n};\n"
    chunks = chunk(src)
    fns = [c for c in chunks if c.kind == "function"]
    assert len(fns) == 1
    assert fns[0].symbol_name == "foo"


def test_module_level_statements_captured():
    src = "const CONFIG = { retries: 3 };\nconst TIMEOUT = 30;\n"
    chunks = chunk(src)
    blocks = [c for c in chunks if c.kind == "module_block"]
    assert len(blocks) == 1
    assert "CONFIG" in blocks[0].code_text and "TIMEOUT" in blocks[0].code_text


def test_context_header_includes_imports():
    src = "import { readFile } from 'fs';\n\nfunction load() {\n  return readFile('x');\n}\n"
    chunks = chunk(src)
    fn = next(c for c in chunks if c.kind == "function")
    assert "readFile" in fn.embed_text

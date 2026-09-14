from pathlib import Path

from codesearch.chunking.python_chunker import PythonChunker

FILE = Path("sample.py")


def chunk(source: str):
    return PythonChunker().chunk_file(FILE, source)


def test_plain_function():
    src = "def add(a, b):\n    return a + b\n"
    chunks = chunk(src)
    fns = [c for c in chunks if c.kind == "function"]
    assert len(fns) == 1
    assert fns[0].symbol_name == "add"
    assert fns[0].start_line == 1
    assert fns[0].end_line == 2


def test_function_docstring_captured():
    src = 'def add(a, b):\n    """Add two numbers."""\n    return a + b\n'
    chunks = chunk(src)
    fn = next(c for c in chunks if c.kind == "function")
    assert "Add two numbers." in fn.code_text


def test_async_function():
    src = "async def fetch(url):\n    return await get(url)\n"
    chunks = chunk(src)
    fns = [c for c in chunks if c.kind == "function"]
    assert len(fns) == 1
    assert fns[0].symbol_name == "fetch"
    assert "async def fetch" in fns[0].code_text


def test_decorated_function():
    src = "@staticmethod\ndef helper(x):\n    return x\n"
    chunks = chunk(src)
    fn = next(c for c in chunks if c.kind == "function")
    assert "@staticmethod" in fn.code_text
    assert fn.symbol_name == "helper"


def test_class_with_multiple_methods():
    src = (
        "class Greeter:\n"
        '    """Greets people."""\n\n'
        "    def hello(self, name):\n"
        "        return f'hi {name}'\n\n"
        "    def bye(self, name):\n"
        "        return f'bye {name}'\n"
    )
    chunks = chunk(src)
    cls = next(c for c in chunks if c.kind == "class")
    assert cls.symbol_name == "Greeter"
    assert "Greets people." in cls.code_text
    assert "hello" in cls.code_text and "bye" in cls.code_text

    methods = [c for c in chunks if c.kind == "method"]
    assert {m.qualified_name for m in methods} == {"Greeter.hello", "Greeter.bye"}


def test_nested_class_in_method_not_double_counted():
    src = (
        "class Outer:\n"
        "    def method_a(self):\n"
        "        def inner():\n"
        "            return 1\n"
        "        return inner()\n"
    )
    chunks = chunk(src)
    methods = [c for c in chunks if c.kind == "method"]
    assert len(methods) == 1
    assert methods[0].qualified_name == "Outer.method_a"


def test_module_level_statements_captured():
    src = "X = 1\nY = 2\n\ndef foo():\n    return X + Y\n"
    chunks = chunk(src)
    blocks = [c for c in chunks if c.kind == "module_block"]
    assert len(blocks) == 1
    assert "X = 1" in blocks[0].code_text and "Y = 2" in blocks[0].code_text


def test_line_numbers_are_one_indexed_and_accurate():
    src = "\n\ndef foo():\n    return 1\n"
    chunks = chunk(src)
    fn = next(c for c in chunks if c.kind == "function")
    assert fn.start_line == 3
    assert fn.end_line == 4


def test_context_header_includes_imports():
    src = "import os\nfrom sys import argv\n\ndef foo():\n    return os.getcwd()\n"
    chunks = chunk(src)
    fn = next(c for c in chunks if c.kind == "function")
    assert "import os" in fn.embed_text
    assert "from sys import argv" in fn.embed_text

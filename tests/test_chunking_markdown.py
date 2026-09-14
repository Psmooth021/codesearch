from pathlib import Path

from codesearch.chunking.markdown_chunker import MarkdownChunker

FILE = Path("README.md")


def chunk(source: str):
    return MarkdownChunker().chunk_file(FILE, source)


def test_single_heading_and_content():
    src = "# Title\n\nSome content.\n"
    chunks = chunk(src)
    assert len(chunks) == 1
    assert chunks[0].symbol_name == "Title"
    assert chunks[0].qualified_name == "Title"
    assert "Some content." in chunks[0].code_text


def test_nested_headings_build_breadcrumb():
    src = "# Top\n\nintro\n\n## Sub\n\nsub content\n"
    chunks = chunk(src)
    top = next(c for c in chunks if c.symbol_name == "Top")
    sub = next(c for c in chunks if c.symbol_name == "Sub")
    assert top.qualified_name == "Top"
    assert sub.qualified_name == "Top > Sub"


def test_parent_section_excludes_child_content():
    src = "# Top\n\nintro only\n\n## Sub\n\nsub content only\n"
    chunks = chunk(src)
    top = next(c for c in chunks if c.symbol_name == "Top")
    sub = next(c for c in chunks if c.symbol_name == "Sub")
    assert "intro only" in top.code_text
    assert "sub content only" not in top.code_text
    assert "sub content only" in sub.code_text


def test_preamble_before_first_heading_becomes_its_own_chunk():
    src = "Preamble text.\n\n# First Heading\n\nBody.\n"
    chunks = chunk(src)
    assert chunks[0].code_text.strip() == "Preamble text."
    assert chunks[0].symbol_name == "intro"
    assert chunks[1].symbol_name == "First Heading"


def test_no_headings_at_all_is_one_intro_chunk():
    src = "Just a paragraph.\n\nAnother paragraph.\n"
    chunks = chunk(src)
    assert len(chunks) == 1
    assert chunks[0].symbol_name == "intro"


def test_hash_inside_fenced_code_block_is_not_a_heading():
    src = "# Title\n\n```python\n# not a heading\ndef foo():\n    pass\n```\n"
    chunks = chunk(src)
    assert len(chunks) == 1
    assert "# not a heading" in chunks[0].code_text
    assert chunks[0].symbol_name == "Title"


def test_three_level_nesting_breadcrumb():
    src = "# A\n\nintro\n\n## B\n\nb text\n\n### C\n\nc text\n"
    chunks = chunk(src)
    c_chunk = next(ch for ch in chunks if ch.symbol_name == "C")
    assert c_chunk.qualified_name == "A > B > C"


def test_sibling_sections_do_not_leak_into_each_other():
    src = "# A\n\ntext a\n\n## B\n\ntext b\n\n## C\n\ntext c\n"
    chunks = chunk(src)
    b_chunk = next(ch for ch in chunks if ch.symbol_name == "B")
    c_chunk = next(ch for ch in chunks if ch.symbol_name == "C")
    assert "text c" not in b_chunk.code_text
    assert "text b" not in c_chunk.code_text
    assert b_chunk.qualified_name == "A > B"
    assert c_chunk.qualified_name == "A > C"


def test_oversized_section_windows_with_breadcrumb_preserved():
    lines = [
        f"This is sentence number {i} with a bit of filler text to pad it out nicely."
        for i in range(150)
    ]
    src = "# Top\n\n## Big\n\n" + "\n".join(lines) + "\n"
    chunks = chunk(src)
    big_windows = [c for c in chunks if c.symbol_name == "Big"]
    assert len(big_windows) > 1
    assert all(c.truncated for c in big_windows)
    assert all(c.qualified_name == "Top > Big" for c in big_windows)


def test_context_header_includes_breadcrumb():
    src = "# Top\n\nintro\n\n## Sub\n\nsub content\n"
    chunks = chunk(src)
    sub = next(c for c in chunks if c.symbol_name == "Sub")
    assert "Top > Sub" in sub.embed_text


def test_raw_html_h1_gets_its_own_chunk():
    # The common "centered logo + HTML title" README style: a single-line
    # <h1>...</h1> doesn't qualify as tree-sitter's own html_block node
    # type (CommonMark's HTML block rules require an opening/closing tag
    # alone on the line, not open+content+close together) - it parses as a
    # plain paragraph, so a naive "only handle atx_heading" chunker would
    # bury the title in an undifferentiated intro blob.
    src = (
        '<p align="center"><img src="logo.png" /></p>\n\n'
        '<h1 align="center">MyProject: Doing Cool Things Fast</h1>\n\n'
        '<p align="center"><strong>Tagline.</strong></p>\n\n'
        "## Installation\n\nRun pip install.\n"
    )
    chunks = chunk(src)
    title_chunk = next(c for c in chunks if c.symbol_name == "MyProject: Doing Cool Things Fast")
    assert "<h1" in title_chunk.code_text
    assert "logo.png" not in title_chunk.code_text

    intro_chunk = next(c for c in chunks if c.symbol_name == "intro")
    assert "logo.png" in intro_chunk.code_text
    assert "MyProject" not in intro_chunk.code_text


def test_inline_html_tag_inside_prose_does_not_split():
    src = (
        "# Docs\n\n"
        "You can embed a heading like <h2>this</h2> inline in a sentence, "
        "it should not split here.\n\n"
        "Normal paragraph continues.\n"
    )
    chunks = chunk(src)
    assert len(chunks) == 1
    assert chunks[0].symbol_name == "Docs"
    assert "<h2>this</h2>" in chunks[0].code_text


def test_html_heading_breadcrumb_nests_under_atx_ancestor():
    src = (
        "# Top\n\nintro\n\n"
        "<h2>Embedded Title</h2>\n\ncontent under it\n\n"
        "## Sub\n\nsub content\n"
    )
    chunks = chunk(src)
    html_chunk = next(c for c in chunks if c.symbol_name == "Embedded Title")
    assert html_chunk.qualified_name == "Top > Embedded Title"
    sub_chunk = next(c for c in chunks if c.symbol_name == "Sub")
    assert sub_chunk.qualified_name == "Top > Sub"


def test_multiple_html_headings_each_get_their_own_chunk():
    src = "<h1>First</h1>\n\ntext a\n\n<h1>Second</h1>\n\ntext b\n"
    chunks = chunk(src)
    names = {c.symbol_name for c in chunks}
    assert names == {"First", "Second"}
    first = next(c for c in chunks if c.symbol_name == "First")
    second = next(c for c in chunks if c.symbol_name == "Second")
    assert "text a" in first.code_text and "text b" not in first.code_text
    assert "text b" in second.code_text and "text a" not in second.code_text

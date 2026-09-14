from pathlib import Path

from codesearch.config import Settings
from codesearch.walker import iter_source_files


def _make_repo(tmp_path: Path) -> Path:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("print('hi')\n")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "lib.js").write_text("module.exports = {};\n")
    (tmp_path / "build").mkdir()
    (tmp_path / "build" / "out.js").write_text("console.log(1);\n")
    (tmp_path / "notes.txt").write_text("hello\n")
    (tmp_path / "logo.png").write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00")
    return tmp_path


def test_builtin_excludes_skip_vendored_and_build_dirs(tmp_path):
    root = _make_repo(tmp_path)
    files = {str(p) for p in iter_source_files(root, Settings())}
    assert not any("node_modules" in f for f in files)
    assert not any("build" in f for f in files)


def test_binary_files_skipped(tmp_path):
    root = _make_repo(tmp_path)
    files = {str(p) for p in iter_source_files(root, Settings())}
    assert not any(f.endswith(".png") for f in files)


def test_regular_source_files_included(tmp_path):
    root = _make_repo(tmp_path)
    files = {str(p) for p in iter_source_files(root, Settings())}
    assert any(f.endswith("main.py") for f in files)
    assert any(f.endswith("notes.txt") for f in files)


def test_gitignore_respected(tmp_path):
    root = _make_repo(tmp_path)
    (root / ".gitignore").write_text("notes.txt\n")
    files = {str(p) for p in iter_source_files(root, Settings())}
    assert not any(f.endswith("notes.txt") for f in files)
    assert any(f.endswith("main.py") for f in files)


def test_gitignore_directory_pattern(tmp_path):
    root = _make_repo(tmp_path)
    (root / "scratch").mkdir()
    (root / "scratch" / "temp.py").write_text("x = 1\n")
    (root / ".gitignore").write_text("scratch/\n")
    files = {str(p) for p in iter_source_files(root, Settings())}
    assert not any("scratch" in f for f in files)

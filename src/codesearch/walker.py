"""Walks a repo yielding indexable file paths.

Respects the repo's own .gitignore (via pathspec), plus a built-in set of
excluded directories/extensions (config.BUILTIN_EXCLUDE_DIRS/EXTENSIONS)
so vendored dependencies, build output, and binaries never get chunked.
"""

from __future__ import annotations

import os
from pathlib import Path

import pathspec

from codesearch.config import Settings

_BINARY_SNIFF_BYTES = 8192


def _load_gitignore(root: Path) -> pathspec.PathSpec | None:
    gitignore = root / ".gitignore"
    if not gitignore.exists():
        return None
    lines = gitignore.read_text(errors="replace").splitlines()
    return pathspec.PathSpec.from_lines("gitignore", lines)


def _looks_binary(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            chunk = f.read(_BINARY_SNIFF_BYTES)
    except OSError:
        return True
    return b"\x00" in chunk


def iter_source_files(root: Path, settings: Settings | None = None):
    """Yield Paths (relative to `root`) for every file that should be indexed."""
    settings = settings or Settings()
    root = root.resolve()
    spec = _load_gitignore(root)

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in settings.excluded_dirs]

        for filename in filenames:
            abs_path = Path(dirpath) / filename
            rel_path = abs_path.relative_to(root)

            if abs_path.suffix in settings.excluded_extensions:
                continue
            if spec is not None and spec.match_file(str(rel_path).replace(os.sep, "/")):
                continue
            if _looks_binary(abs_path):
                continue

            yield rel_path

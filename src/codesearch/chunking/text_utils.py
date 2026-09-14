"""Shared helpers used by every language chunker."""

from __future__ import annotations


def token_count(text: str) -> int:
    """Whitespace-split token count - a cheap proxy for model tokens,
    good enough to bound chunk size without pulling in a full tokenizer
    for chunking decisions."""
    return len(text.split())


def line_windows(
    text: str, base_start_line: int, window_lines: int, overlap_lines: int
) -> list[tuple[int, int, str]]:
    """Split `text` into overlapping line windows.

    Returns a list of (start_line, end_line, window_text), with line
    numbers offset from `base_start_line` (1-indexed, as if `text`'s first
    line were `base_start_line`).
    """
    lines = text.splitlines()
    if not lines:
        return []
    step = max(window_lines - overlap_lines, 1)
    windows: list[tuple[int, int, str]] = []
    idx = 0
    while idx < len(lines):
        chunk_lines = lines[idx : idx + window_lines]
        window_text = "\n".join(chunk_lines)
        start_line = base_start_line + idx
        end_line = base_start_line + idx + len(chunk_lines) - 1
        windows.append((start_line, end_line, window_text))
        if idx + window_lines >= len(lines):
            break
        idx += step
    return windows

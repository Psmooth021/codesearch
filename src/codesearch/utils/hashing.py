from __future__ import annotations

import hashlib


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_hash(path) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

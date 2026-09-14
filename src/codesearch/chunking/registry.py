"""Extension -> Chunker dispatch."""

from __future__ import annotations

from codesearch.chunking.base import Chunker
from codesearch.chunking.fallback import FallbackChunker
from codesearch.chunking.javascript_chunker import JavaScriptChunker
from codesearch.chunking.python_chunker import PythonChunker
from codesearch.config import LANGUAGE_EXTENSIONS, Settings


def get_chunker_for_extension(extension: str, settings: Settings | None = None) -> Chunker:
    settings = settings or Settings()
    language = LANGUAGE_EXTENSIONS.get(extension)
    if language == "python":
        return PythonChunker(settings)
    if language in ("javascript", "typescript", "tsx"):
        return JavaScriptChunker(language, settings)
    return FallbackChunker(settings)

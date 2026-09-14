"""End-to-end smoke test: index a real fixture repo, query it, and confirm
the obviously-relevant chunk comes back near the top. This is the one test
that exercises the whole pipeline (walker -> chunker -> embedder ->
vectorstore -> search) together, not just each piece in isolation."""

import shutil
from pathlib import Path

import pytest

from codesearch.config import Settings
from codesearch.embedding import Embedder
from codesearch.indexer import index_repo
from codesearch.search import open_store
from codesearch.search import query as run_query

FIXTURE_REPO = Path(__file__).parent / "fixtures" / "sample_repo"

# Real model download + encode - slow relative to the rest of the suite,
# and network-dependent, so it's opt-in via `pytest -m integration`.
pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def embedder() -> Embedder:
    settings = Settings()
    return Embedder(settings.embedding_model, settings.embedding_batch_size)


@pytest.fixture()
def indexed_repo(tmp_path, embedder):
    repo = tmp_path / "sample_repo"
    shutil.copytree(FIXTURE_REPO, repo)
    index_repo(repo, settings=Settings(), embedder=embedder)
    return repo


def test_index_produces_artifacts(indexed_repo):
    index_dir = indexed_repo / ".codesearch"
    assert (index_dir / "index.faiss").exists()
    assert (index_dir / "metadata.db").exists()
    assert (index_dir / "manifest.json").exists()


def test_query_returns_relevant_top_hit(indexed_repo, embedder):
    store = open_store(indexed_repo, embedder)
    try:
        hits = run_query(store, embedder, "how does the LRU cache evict old entries", k=3)
    finally:
        store.close()
    assert hits
    assert any(h.file_path.endswith("cache.py") for h in hits)


def test_incremental_reindex_skips_unchanged_files(indexed_repo, embedder):
    stats = index_repo(indexed_repo, settings=Settings(), embedder=embedder)
    assert stats.files_indexed == 0
    assert stats.files_skipped_unchanged == stats.files_scanned


def test_changed_file_gets_reindexed(indexed_repo, embedder):
    target = indexed_repo / "pkg" / "cache.py"
    target.write_text(target.read_text() + "\n\ndef extra():\n    return 1\n")
    stats = index_repo(indexed_repo, settings=Settings(), embedder=embedder)
    assert stats.files_indexed == 1
    assert stats.files_skipped_unchanged == stats.files_scanned - 1

"""Runs every labeled query in queries.jsonl against a built index and
reports precision@k/recall@k/MRR, macro-averaged overall and per-repo.

Two ablation modes let the headline README claims be regenerated on
demand rather than asserted:
  --baseline naive          chunk with FallbackChunker instead of the
                             AST-aware chunkers, same embedding model
  --embedding-model NAME     swap the embedding model, same chunking
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from codesearch.chunking.base import Chunker
from codesearch.chunking.fallback import FallbackChunker
from codesearch.config import Settings
from codesearch.embedding import Embedder
from codesearch.eval.metrics import macro_average, mrr, precision_at_k, recall_at_k
from codesearch.indexer import index_repo
from codesearch.search import open_store
from codesearch.search import query as run_query
from codesearch.vectorstore import SearchHit

EVAL_DATA_DIR = Path(__file__).parent / "data"
DEFAULT_QUERIES_PATH = EVAL_DATA_DIR / "queries.jsonl"
DEFAULT_MANIFEST_PATH = EVAL_DATA_DIR / "repo_manifest.yaml"
EVAL_REPOS_DIR = Path(__file__).parent.parent.parent.parent / "eval_repos"
EVAL_INDEX_ROOT = Path(__file__).parent.parent.parent.parent / ".eval_indexes"

DEFAULT_KS = (1, 3, 5, 10)


@dataclass
class RelevantItem:
    file: str
    symbol: str
    start_line: int
    end_line: int

    def key(self) -> str:
        return f"{self.file}::{self.symbol}"


@dataclass
class EvalQuery:
    id: str
    repo: str
    query: str
    relevant: list[RelevantItem]


@dataclass
class QueryResult:
    query_id: str
    repo: str
    precision: dict[int, float] = field(default_factory=dict)
    recall: dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0


def load_repo_roots(path: Path = DEFAULT_MANIFEST_PATH) -> dict[str, str]:
    """Maps repo name -> subdirectory (relative to eval_repos/<name>/) that
    ground-truth query file paths are relative to - e.g. requests/flask use
    a src/ layout, express doesn't."""
    manifest = yaml.safe_load(path.read_text(encoding="utf-8"))
    return {repo["name"]: repo.get("root", ".") for repo in manifest["repos"]}


def load_queries(path: Path = DEFAULT_QUERIES_PATH) -> list[EvalQuery]:
    queries: list[EvalQuery] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        relevant = [RelevantItem(**item) for item in obj["relevant"]]
        queries.append(
            EvalQuery(id=obj["id"], repo=obj["repo"], query=obj["query"], relevant=relevant)
        )
    return queries


_MIN_OVERLAP_IOU = 0.5


def _line_iou(a_start: int, a_end: int, b_start: int, b_end: int) -> float:
    """Intersection-over-union of two inclusive line ranges."""
    overlap = min(a_end, b_end) - max(a_start, b_start) + 1
    if overlap <= 0:
        return 0.0
    union = max(a_end, b_end) - min(a_start, b_start) + 1
    return overlap / union


def _hit_ids(hits: list[SearchHit], relevant: list[RelevantItem]) -> list[str]:
    """Map each retrieved hit to a stable id string usable against
    `RelevantItem.key()` - matching on exact (file, symbol) OR line-range
    IoU >= _MIN_OVERLAP_IOU, tolerant of minor chunk-boundary drift across
    indexer versions.

    A plain "any overlap counts" rule was tried first and rejected: it lets
    one large fallback-chunker window that happens to span an entire small
    file get free credit for containing every function in it, including
    whichever one a query's ground truth names - inflating the naive
    baseline's numbers without it actually retrieving anything precise.
    Requiring proportional overlap (IoU) means a chunk only counts as a
    match if it's actually *about* the relevant span, not merely
    "large enough to contain it somewhere inside."

    A hit that matches no relevant item gets its own unique id (its
    chunk_id) so it correctly counts as a non-hit rather than accidentally
    colliding with something in `relevant`."""
    ids: list[str] = []
    for hit in hits:
        matched_key = None
        for item in relevant:
            same_symbol = hit.file_path == item.file and hit.qualified_name == item.symbol
            same_file = hit.file_path == item.file
            good_overlap = same_file and (
                _line_iou(hit.start_line, hit.end_line, item.start_line, item.end_line)
                >= _MIN_OVERLAP_IOU
            )
            if same_symbol or good_overlap:
                matched_key = item.key()
                break
        ids.append(matched_key or f"__miss__{hit.chunk_id}")
    return ids


def _index_eval_repo(
    repo_name: str,
    repo_path: Path,
    settings: Settings,
    embedder: Embedder,
    baseline: str | None,
) -> Path:
    if not repo_path.exists():
        raise FileNotFoundError(
            f"{repo_path} not found - run `python scripts/download_eval_repos.py` first."
        )
    suffix = f"{embedder.fingerprint}_{baseline or 'ast'}"
    index_dir = EVAL_INDEX_ROOT / repo_name / suffix

    chunker_factory: Callable[[str, Settings], Chunker] | None = None
    if baseline == "naive":

        def chunker_factory(_extension: str, s: Settings) -> Chunker:
            return FallbackChunker(s)

    index_repo(repo_path, index_dir, settings, embedder, chunker_factory=chunker_factory)
    return index_dir


def run_eval(
    queries_path: Path = DEFAULT_QUERIES_PATH,
    repo_filter: str | None = None,
    ks: tuple[int, ...] = DEFAULT_KS,
    baseline: str | None = None,
    embedding_model: str | None = None,
) -> dict:
    settings = Settings(embedding_model=embedding_model) if embedding_model else Settings()
    embedder = Embedder(settings.embedding_model, settings.embedding_batch_size)

    all_queries = load_queries(queries_path)
    if repo_filter:
        all_queries = [q for q in all_queries if q.repo == repo_filter]
    repos = sorted({q.repo for q in all_queries})

    repo_roots = load_repo_roots()
    repo_paths = {repo: EVAL_REPOS_DIR / repo / repo_roots.get(repo, ".") for repo in repos}
    index_dirs = {
        repo: _index_eval_repo(repo, repo_paths[repo], settings, embedder, baseline)
        for repo in repos
    }

    results: list[QueryResult] = []
    max_k = max(ks)
    for eq in all_queries:
        store = open_store(repo_paths[eq.repo], embedder, index_dirs[eq.repo])
        try:
            hits = run_query(store, embedder, eq.query, k=max_k)
        finally:
            store.close()

        retrieved_ids = _hit_ids(hits, eq.relevant)
        relevant_keys = {item.key() for item in eq.relevant}

        qr = QueryResult(query_id=eq.id, repo=eq.repo)
        for k in ks:
            qr.precision[k] = precision_at_k(retrieved_ids, relevant_keys, k)
            qr.recall[k] = recall_at_k(retrieved_ids, relevant_keys, k)
        qr.mrr = mrr(retrieved_ids, relevant_keys)
        results.append(qr)

    return _summarize(results, ks)


def _summarize(results: list[QueryResult], ks: tuple[int, ...]) -> dict:
    def block(subset: list[QueryResult]) -> dict:
        return {
            "n_queries": len(subset),
            "precision_at_k": {k: macro_average([r.precision[k] for r in subset]) for k in ks},
            "recall_at_k": {k: macro_average([r.recall[k] for r in subset]) for k in ks},
            "mrr": macro_average([r.mrr for r in subset]),
        }

    repos = sorted({r.repo for r in results})
    return {
        "overall": block(results),
        "per_repo": {repo: block([r for r in results if r.repo == repo]) for repo in repos},
    }


def format_markdown_table(summary: dict, ks: tuple[int, ...] = DEFAULT_KS) -> str:
    lines = ["| Scope | n | " + " | ".join(f"P@{k}" for k in ks) + " | " + " | ".join(
        f"R@{k}" for k in ks
    ) + " | MRR |"]
    lines.append("|---" * (2 + 2 * len(ks) + 1) + "|")

    def row(name: str, block: dict) -> str:
        p = " | ".join(f"{block['precision_at_k'][k]:.3f}" for k in ks)
        r = " | ".join(f"{block['recall_at_k'][k]:.3f}" for k in ks)
        return f"| {name} | {block['n_queries']} | {p} | {r} | {block['mrr']:.3f} |"

    lines.append(row("**overall**", summary["overall"]))
    for repo, block_data in summary["per_repo"].items():
        lines.append(row(repo, block_data))
    return "\n".join(lines)

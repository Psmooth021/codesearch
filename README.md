# codesearch

A CLI that indexes a codebase with **AST-aware chunking** (tree-sitter, not naive line-splitting) and lets you search it with natural-language questions, using a local, offline embedding model. No API key required.

What makes this more than a wrapper around an embedding API: every design decision below is backed by a **real evaluation harness** — 45 hand-labeled natural-language queries against three real open-source repos, with precision@k/recall@k/MRR numbers you can regenerate yourself, not just asserted in prose.

```
$ codesearch index ~/code/some-repo
$ codesearch query "how does the retry logic decide when to give up"
requests/adapters.py:634-748 (HTTPAdapter.send, score=0.612)
...
```

## Why this exists

Most "semantic code search" demos are a thin wrapper: chunk a file into fixed-size windows, embed each window, cosine-similarity search. That's fast to build and easy to make *look* like it works on a cherry-picked example. It's much harder to make it actually retrieve the right function for a real question — and to know whether it does, at all, beyond eyeballing a few queries.

This project's differentiator is treating code search as something you evaluate, not just demo:

1. **AST-aware chunking** (tree-sitter) — each chunk is one complete function, method, or class, not an arbitrary slice of lines.
2. **A real eval harness** — a hand-labeled query set against pinned commits of `requests`, `flask`, and `express`, scored with precision@k/recall@k/MRR.
3. **Two ablations with measured numbers**, not assumptions — AST-aware vs naive chunking, and a code-specific embedding model vs a general-purpose one. One of these two ablations did **not** go the way I expected; see below.

## Install

```
pip install -e ".[dev]"
codesearch --help
```

Requires Python 3.10+. First run downloads the embedding model and tree-sitter grammars from Hugging Face/PyPI; after that, indexing and querying make zero network calls — "offline-capable" means no network at index/query time, not zero downloads ever.

## Usage

```
codesearch index <path>                          # build or incrementally update the index
codesearch query "<question>" --path <path>       # search it
codesearch status --path <path>                   # index staleness, chunk count, model used
codesearch eval [--repo NAME] [--baseline naive]   # run the eval harness (see below)
```

`index` is incremental: re-running it hashes each file's content and skips anything unchanged, rather than re-embedding the whole repo every time.

## Architecture

**Indexing**: walk the repo (respecting `.gitignore`, plus built-in excludes for `node_modules`/`venv`/`build`/etc.) → dispatch each file to a language-specific chunker by extension → embed each chunk in batches → write to a FAISS index + sqlite metadata sidecar, stored at `<repo>/.codesearch/`.

**Querying**: embed the question with the same model → cosine-similarity search over the FAISS index → join hits against sqlite for file/line/symbol metadata → render ranked, syntax-highlighted results.

**Incremental indexing**: each file's content hash is tracked in sqlite. An unchanged file is skipped entirely — no re-chunk, no re-embed. A changed file has its old chunks purged and replaced. A deleted file's chunks are pruned. Most portfolio-scale code search tools just re-embed everything on every run; this one doesn't.

## Chunking strategy

Built on [tree-sitter](https://tree-sitter.github.io/) via `tree-sitter-language-pack`, with dedicated chunkers for Python and JS/TS/TSX:

- **One chunk per function/method** — full signature + body + its docstring (Python) or directly-adjacent leading comment (JS/TS), not an arbitrary window of lines.
- **One summary chunk per class** — signature + docstring + a synthesized method list, so "what does the `RetryPolicy` class do" matches at class granularity instead of being buried inside one arbitrary method. TS/TSX interfaces get the same treatment.
- **A context header** prepended before embedding (file path, enclosing class, leading imports) — the citation range (`start_line`/`end_line`) still points at the real code, only the *embedded* text carries the extra context.
- **The CommonJS `obj.method = function () {...}` idiom is handled explicitly** — not just ES6 classes/`const`. Express's entire `lib/application.js` and `lib/response.js` are written this way (`app.set = function set(...) {...}`), and a chunker that only understood `function`/`class`/`const` declarations would silently fall back to windowing this repo's most important files. This was a real bug caught by running the chunker against Express during eval-corpus prep, not a hypothetical edge case.
- **Oversized functions** (> ~2000 whitespace-tokens) split at nested function/class boundaries first, falling back to an overlapping line-window (tagged `truncated=True`) only if there's nothing to split on.
- **Anything not captured by the above** (bare module-level statements, files in languages without a dedicated chunker) is grouped into overlapping sliding-window chunks, so nothing is silently dropped from the index — it's just less precisely chunked.

### Why this matters

Naive fixed-size windows routinely cut a function signature from its body, merge two unrelated functions into one chunk, or split one function's logic across two vectors — diluting the embedding for each half and hurting both precision (irrelevant code pollutes a chunk's embedding) and recall (no single chunk fully represents the relevant function). AST-aware chunking guarantees each chunk is one complete, addressable semantic unit — which is also what makes precise `file:line:symbol` citations possible in the first place.

## Vector index: FAISS `IndexFlatIP`

`faiss-cpu`'s `IndexFlatIP` over L2-normalized vectors (inner product on unit vectors = cosine similarity), wrapped in `IndexIDMap2` so chunks can be removed by id for incremental re-indexing, plus a plain sqlite sidecar for chunk metadata.

- **Target scale**: up to a few thousand files → tens of thousands of chunks (e.g. 25k vectors @ 384 dims). Exact brute-force search over that is tens of milliseconds — well within interactive CLI latency.
- **Rejected an approximate index** (IVF/HNSW): at this scale it would trade exactness for speed the tool doesn't need. The crossover point: at roughly 1M+ vectors, `IndexHNSWFlat` or `IndexIVFPQ` becomes the right call; below that, exact search is strictly better *and* simpler.
- **Rejected `chromadb`**: it hides the exact indexing decisions this project exists to showcase.
- **Rejected `sqlite-vec`**: smaller ecosystem, adds SQL-extension-loading complexity for no benefit at this scale.

## Evaluation harness

**Corpus**: three permissively-licensed OSS repos, pinned to exact commit SHAs (`src/codesearch/eval/data/repo_manifest.yaml`) so results are reproducible, not "whatever HEAD happened to be":

| Repo | Language | Why |
|---|---|---|
| [psf/requests](https://github.com/psf/requests) | Python | small, well-documented, idiomatic |
| [pallets/flask](https://github.com/pallets/flask) | Python | second data point, different code style |
| [expressjs/express](https://github.com/expressjs/express) | JavaScript | proves the JS chunker isn't decorative |

Run `python scripts/download_eval_repos.py` to shallow-clone all three at their pinned commits into a gitignored `eval_repos/`.

**Ground truth** (`src/codesearch/eval/data/queries.jsonl`, 45 queries — 18/15/12 per repo): for each query, I found a documented feature in the repo's own README/docs, located the real implementing function by reading the source, and phrased the query the way a contributor would ask it in natural language — deliberately *not* reusing the function's own identifier names, so the eval tests semantic retrieval rather than lexical keyword overlap. Every line-range in the ground truth was extracted by running the actual chunker against the actual pinned source, not hand-counted.

**Matching rule**: a retrieved chunk counts as a hit if it exactly matches the ground truth's `(file, qualified_name)`, or its line range overlaps the ground truth range with **IoU ≥ 0.5** (intersection over union of the two line ranges). This needed a second pass to get right — see below.

Run it yourself: `codesearch eval [--repo NAME] [--baseline naive] [--embedding-model NAME] [--k 1,3,5,10]`.

### Result 1: AST-aware chunking vs. naive fixed-window chunking

Same embedding model (the default, `all-MiniLM-L6-v2`), same 45 queries — the only variable is chunk boundaries.

| Scope | n | P@1 | P@3 | P@5 | P@10 | R@1 | R@3 | R@5 | R@10 | MRR |
|---|---|---|---|---|---|---|---|---|---|---|
| **AST-aware (default)** | 45 | 0.333 | 0.215 | 0.160 | 0.087 | 0.333 | 0.633 | 0.778 | 0.844 | **0.504** |
| naive fixed-window (`--baseline naive`) | 45 | 0.156 | 0.081 | 0.053 | 0.036 | 0.156 | 0.244 | 0.267 | 0.356 | **0.207** |

AST-aware chunking roughly **2.4x MRR** over naive windowing, and finds the right answer in the top 10 for 84% of queries vs 36%.

### Result 2: a "code-specific" embedding model vs. a general-purpose one — the surprise

Same AST-aware chunking, same 45 queries — the only variable is the embedding model. Going in, I expected `st-codesearch-distilroberta-base` (fine-tuned on CodeSearchNet specifically for code search) to beat `all-MiniLM-L6-v2` (a general-purpose sentence embedding model). It didn't:

| Scope | n | P@1 | P@3 | P@5 | P@10 | R@1 | R@3 | R@5 | R@10 | MRR |
|---|---|---|---|---|---|---|---|---|---|---|
| **`all-MiniLM-L6-v2` (default)** | 45 | 0.333 | 0.215 | 0.160 | 0.087 | 0.333 | 0.633 | 0.778 | 0.844 | **0.504** |
| `st-codesearch-distilroberta-base` (`--embedding-model ...`) | 45 | 0.311 | 0.141 | 0.107 | 0.060 | 0.300 | 0.411 | 0.511 | 0.578 | **0.389** |

My working explanation: CodeSearchNet-style training teaches a model to match code against its *docstring* — a fundamentally different task from matching code against a *natural-language question a contributor would actually ask* ("how does the client decide whether to bypass the configured proxy"), which is closer to what general-purpose sentence embedding models like MiniLM are trained on at much larger scale. A code-specific model isn't automatically the right choice for a natural-language-question interface, and I only know that because I measured it instead of assuming it.

### A methodology bug I found and fixed along the way

The first version of the matching rule counted *any* line-range overlap as a hit. Under that rule, the naive baseline's chunking initially **beat** AST-aware chunking overall — which didn't match what the actual retrieved results looked like on manual inspection. The cause: naive windows are 60 lines with 10-line overlap, so for a small file (e.g. `requests/hooks.py` at 48 lines), a *single* window spans the entire file. That one broad, unfocused chunk would trivially "overlap" whatever function a query's ground truth happened to name — getting free credit for containing the target without actually being a precise, targeted retrieval of it.

I fixed this by requiring proportional overlap (IoU ≥ 0.5) instead of any overlap, which is the methodologically standard way to score span retrieval and closes that loophole without discarding the original goal (tolerance for minor chunk-boundary drift between AST-chunker versions). After the fix, AST-aware chunking clearly wins, as it should. I'm noting this here because reporting numbers without being willing to interrogate a result that looks off is exactly the kind of "it demos well but nobody checked" gap this whole project is trying to avoid.

## Project structure

```
src/codesearch/
  cli.py              Typer CLI - thin, delegates to the modules below
  config.py           Settings: model name, chunk-size caps, excludes
  walker.py           repo walk, .gitignore + built-in exclude filtering
  indexer.py          walk -> chunk -> embed -> vectorstore, incremental
  embedding.py        sentence-transformers wrapper, batching, fingerprinting
  vectorstore.py      FAISS IndexFlatIP + sqlite metadata sidecar
  search.py           query embedding + search + result formatting
  chunking/
    python_chunker.py    tree-sitter Python: function/method/class chunks
    javascript_chunker.py tree-sitter JS/TS/TSX: same, + CommonJS idiom
    fallback.py           naive sliding-window chunker + eval baseline
    text_utils.py         shared token-counting/windowing helpers
  eval/
    harness.py          runs queries.jsonl, computes metrics, writes tables
    metrics.py           precision@k / recall@k / MRR - pure functions
    data/queries.jsonl    45 hand-labeled ground-truth queries
tests/                 unit tests per module + one full-pipeline integration test
scripts/
  download_eval_repos.py  pins/clones the 3 eval repos at fixed commits
```

## Testing

```
pytest                     # full suite
pytest -m "not integration"  # fast unit tests only (what CI runs - no model download)
ruff check .
mypy src
```

The one integration test (`tests/test_index_roundtrip.py`) exercises the whole pipeline — walker → chunker → embedder → vectorstore → search — against a small fixture repo, and is the one test that would catch "each piece works in isolation but the wiring is wrong."

## Limitations & future work

- No cross-file understanding (call graphs, "who calls this function") — each chunk is scored independently.
- Python and JS/TS/TSX only; other languages fall back to naive windowing (still indexed, just less precisely).
- Exact search only; no approximate index. Documented above as the right call at this scale, but would need to change well before 1M+ vectors.
- No reranking step — a cross-encoder reranking FAISS's top-k candidates would likely improve precision@1 further; left out to keep query latency low and the pipeline easy to reason about.

## License

MIT

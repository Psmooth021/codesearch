"""Thin Typer CLI - each command validates flags then calls straight into
indexer.py/search.py/eval/harness.py, all independently unit-testable
without going through this layer."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.syntax import Syntax
from rich.table import Table

from codesearch import __version__
from codesearch.config import DEFAULT_EMBEDDING_MODEL, Settings
from codesearch.embedding import Embedder
from codesearch.eval.harness import DEFAULT_QUERIES_PATH, format_markdown_table, run_eval
from codesearch.indexer import default_index_dir, index_repo
from codesearch.search import IndexNotFoundError, open_store
from codesearch.search import query as run_query

app = typer.Typer(add_completion=False, no_args_is_help=True)
console = Console()


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"codesearch {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False, "--version", callback=_version_callback, is_eager=True, help="Show version and exit."
    ),
) -> None:
    pass


@app.command()
def index(
    path: Path = typer.Argument(..., help="Repo (or subdirectory) to index."),
    index_path: Path | None = typer.Option(None, "--index-path", help="Override index location."),
    rebuild: bool = typer.Option(
        False, "--rebuild", help="Discard and rebuild the index from scratch."
    ),
    embedding_model: str = typer.Option(DEFAULT_EMBEDDING_MODEL, "--embedding-model"),
) -> None:
    """Build or incrementally update the search index for a repo."""
    if not path.exists():
        console.print(f"[red]error:[/red] path does not exist: {path}")
        raise typer.Exit(code=1)

    settings = Settings(embedding_model=embedding_model)
    console.print(f"Loading embedding model [cyan]{embedding_model}[/cyan]...")
    embedder = Embedder(embedding_model, settings.embedding_batch_size)

    with console.status("Indexing..."):
        try:
            stats = index_repo(path, index_path, settings, embedder, rebuild=rebuild)
        except ValueError as e:
            console.print(f"[red]error:[/red] {e}")
            raise typer.Exit(code=1) from e

    table = Table(show_header=False)
    table.add_row("Files scanned", str(stats.files_scanned))
    table.add_row("Files indexed (new/changed)", str(stats.files_indexed))
    table.add_row("Files unchanged (skipped)", str(stats.files_skipped_unchanged))
    table.add_row("Files removed", str(stats.files_removed))
    table.add_row("Chunks written", str(stats.chunks_written))
    console.print(table)


@app.command()
def query(
    question: str = typer.Argument(..., help="Natural-language question about the codebase."),
    path: Path = typer.Option(Path("."), "--path", help="Repo the index was built for."),
    k: int = typer.Option(5, "--k", help="Number of results to return."),
    language: str | None = typer.Option(None, "--language", help="Filter results to one language."),
    as_json: bool = typer.Option(
        False, "--json", help="Print raw JSON instead of formatted output."
    ),
    embedding_model: str = typer.Option(DEFAULT_EMBEDDING_MODEL, "--embedding-model"),
) -> None:
    """Search the index with a natural-language question."""
    settings = Settings(embedding_model=embedding_model)
    embedder = Embedder(embedding_model, settings.embedding_batch_size)

    try:
        store = open_store(path, embedder)
    except IndexNotFoundError as e:
        console.print(f"[red]error:[/red] {e}")
        raise typer.Exit(code=1) from e

    try:
        hits = run_query(store, embedder, question, k=k, language=language)
    finally:
        store.close()

    if as_json:
        console.print_json(
            json.dumps(
                [
                    {
                        "file": h.file_path,
                        "start_line": h.start_line,
                        "end_line": h.end_line,
                        "symbol": h.qualified_name,
                        "score": h.score,
                    }
                    for h in hits
                ]
            )
        )
        return

    if not hits:
        console.print("[yellow]No results.[/yellow]")
        return

    for hit in hits:
        console.print(
            f"[bold cyan]{hit.file_path}:{hit.start_line}-{hit.end_line}[/bold cyan] "
            f"[dim]({hit.qualified_name}, score={hit.score:.3f})[/dim]"
        )
        lexer = "python" if hit.language == "python" else "javascript"
        console.print(Syntax(hit.code_text, lexer, line_numbers=True, start_line=hit.start_line))
        console.print()


@app.command()
def eval(
    repo: str | None = typer.Option(
        None, "--repo", help="Restrict to one eval repo (e.g. requests)."
    ),
    k: str = typer.Option("1,3,5,10", "--k", help="Comma-separated list of k values."),
    baseline: str | None = typer.Option(
        None, "--baseline", help="'naive' to chunk with the fallback windower instead of AST."
    ),
    embedding_model: str | None = typer.Option(
        None, "--embedding-model", help="Override the embedding model for this run."
    ),
    output: Path | None = typer.Option(
        None, "--output", help="Write raw results as JSON to this path."
    ),
) -> None:
    """Run the labeled query set against pinned eval repos and report
    precision@k/recall@k/MRR. Requires `python scripts/download_eval_repos.py`
    to have been run first."""
    ks = tuple(int(x) for x in k.split(","))
    with console.status("Indexing eval repos and running queries..."):
        summary = run_eval(
            DEFAULT_QUERIES_PATH,
            repo_filter=repo,
            ks=ks,
            baseline=baseline,
            embedding_model=embedding_model,
        )

    console.print(format_markdown_table(summary, ks))
    if output:
        output.write_text(json.dumps(summary, indent=2))
        console.print(f"\nWrote raw results to {output}")


@app.command()
def status(path: Path = typer.Option(Path("."), "--path")) -> None:
    """Show index staleness, chunk count, model used, and last-indexed info."""
    index_dir = default_index_dir(path)
    manifest_path = index_dir / "manifest.json"
    if not manifest_path.exists():
        console.print(f"[yellow]No index found at {index_dir}[/yellow]")
        raise typer.Exit(code=1)

    manifest = json.loads(manifest_path.read_text())
    embedder = Embedder(manifest["embedding_model"])
    store = open_store(path, embedder)
    try:
        table = Table(show_header=False)
        table.add_row("Index path", str(index_dir))
        table.add_row("Embedding model", manifest["embedding_model"])
        table.add_row("Chunk count", str(store.chunk_count()))
        table.add_row("Indexed files", str(len(store.indexed_file_paths())))
        console.print(table)
    finally:
        store.close()


if __name__ == "__main__":
    app()

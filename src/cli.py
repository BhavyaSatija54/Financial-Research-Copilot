"""
Financial Research Copilot — CLI

Commands:
  frc ingest file <path>          Ingest a local file
  frc ingest dir <directory>      Ingest all .txt files in a directory
  frc ingest sec --ticker AAPL    Fetch + ingest from SEC EDGAR
  frc ingest sample               Ingest bundled sample data
  frc query "<question>"          Run a query against the index
  frc index stats                 Print index statistics
  frc index clear                 Clear all indexes
  frc serve                       Start the API server
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

console = Console()


def _run(coro):  # type: ignore[no-untyped-def]
    """Run an async coroutine from a sync Click command."""
    return asyncio.run(coro)


# ── Root ─────────────────────────────────────────────────────────────

@click.group()
@click.version_option("1.0.0", prog_name="Financial Research Copilot")
def cli() -> None:
    """📊 Financial Research Copilot — hybrid RAG for SEC filings & macro reports."""


# ── Ingest group ─────────────────────────────────────────────────────

@cli.group()
def ingest() -> None:
    """Ingest documents into the vector index."""


@ingest.command("file")
@click.argument("path", type=click.Path(exists=True, path_type=Path))
@click.option("--ticker", default=None, help="Associate with a ticker symbol")
@click.option("--filing-type", default=None, help="Filing type (10-K, 10-Q, etc.)")
def ingest_file(path: Path, ticker: Optional[str], filing_type: Optional[str]) -> None:
    """Ingest a single file."""
    async def _run_ingest() -> None:
        from src.api.dependencies import initialise_services
        from src.utils.config import get_settings

        settings = get_settings()
        await initialise_services(settings)

        from src.api.dependencies import get_ingestion_pipeline
        pipeline = get_ingestion_pipeline()

        meta = {}
        if ticker:
            meta["ticker"] = ticker.upper()
        if filing_type:
            meta["filing_type"] = filing_type.upper()

        with console.status(f"[bold green]Ingesting {path.name}…"):
            result = await pipeline.ingest_file(path, metadata=meta)

        if result.status == "success":
            console.print(Panel(
                f"✅ [bold green]Success[/bold green]\n"
                f"  Chunks added : {result.chunks_added}\n"
                f"  Tokens indexed: {result.tokens_indexed}",
                title=f"Ingested: {path.name}",
            ))
        else:
            console.print(f"[red]Error:[/red] {result.error}")

    _run(_run_ingest())


@ingest.command("dir")
@click.argument("directory", type=click.Path(exists=True, path_type=Path))
@click.option("--glob", default="**/*.txt", show_default=True)
def ingest_directory(directory: Path, glob: str) -> None:
    """Ingest all matching files in a directory."""
    async def _run_ingest() -> None:
        from src.api.dependencies import initialise_services
        from src.utils.config import get_settings
        settings = get_settings()
        await initialise_services(settings)

        from src.api.dependencies import get_ingestion_pipeline
        pipeline = get_ingestion_pipeline()

        with console.status("[bold green]Ingesting directory…"):
            stats = await pipeline.ingest_directory(directory, glob=glob)

        table = Table(title="Ingestion Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        table.add_row("Documents", str(stats.total_documents))
        table.add_row("Chunks", str(stats.total_chunks))
        table.add_row("Tokens", str(stats.total_tokens))
        table.add_row("Errors", str(len(stats.errors)))
        console.print(table)
        if stats.errors:
            for err in stats.errors:
                console.print(f"  [red]⚠[/red] {err}")

    _run(_run_ingest())


@ingest.command("sec")
@click.option("--ticker", required=True, help="Stock ticker (e.g. AAPL)")
@click.option("--filing-type", default="10-K", show_default=True,
              type=click.Choice(["10-K", "10-Q", "8-K"]))
@click.option("--limit", default=3, show_default=True, type=int)
def ingest_sec(ticker: str, filing_type: str, limit: int) -> None:
    """Fetch and ingest SEC EDGAR filings."""
    async def _run_fetch() -> None:
        from src.api.dependencies import initialise_services
        from src.utils.config import get_settings
        settings = get_settings()
        await initialise_services(settings)

        from src.api.dependencies import get_ingestion_pipeline
        from src.ingestion.sec_fetcher import SECFetcher

        pipeline = get_ingestion_pipeline()

        with console.status(f"[bold green]Fetching {ticker} {filing_type} filings…"):
            async with SECFetcher() as fetcher:
                filings = await fetcher.get_filings(ticker, filing_type=filing_type, limit=limit)  # type: ignore

        console.print(f"  Fetched [bold]{len(filings)}[/bold] filings")

        for filing in filings:
            with console.status(f"Ingesting {filing.doc_id}…"):
                result = await pipeline.ingest_text(
                    doc_id=filing.doc_id,
                    text=filing.text,
                    metadata=filing.metadata,
                )
            status_icon = "✅" if result.status == "success" else "❌"
            console.print(f"  {status_icon} {filing.doc_id} — {result.chunks_added} chunks")

    _run(_run_fetch())


@ingest.command("sample")
def ingest_sample() -> None:
    """Ingest the bundled sample documents in data/sample/."""
    sample_dir = Path("data/sample")
    if not sample_dir.exists() or not any(sample_dir.iterdir()):
        console.print("[yellow]No sample files found in data/sample/[/yellow]")
        console.print("Run [bold]scripts/build_index.sh[/bold] to download sample data first.")
        return

    async def _run() -> None:
        from src.api.dependencies import initialise_services
        from src.utils.config import get_settings
        settings = get_settings()
        await initialise_services(settings)

        from src.api.dependencies import get_ingestion_pipeline
        pipeline = get_ingestion_pipeline()
        stats = await pipeline.ingest_directory(sample_dir, glob="**/*.txt")
        console.print(f"✅ Sample ingestion complete: {stats.total_chunks} chunks")

    _run(_run())


# ── Query ─────────────────────────────────────────────────────────────

@cli.command("query")
@click.argument("question")
@click.option("--ticker", default=None)
@click.option("--filing-type", default=None)
@click.option("--top-k", default=None, type=int)
@click.option("--json-output", is_flag=True, default=False, help="Output raw JSON")
def query_cmd(
    question: str,
    ticker: Optional[str],
    filing_type: Optional[str],
    top_k: Optional[int],
    json_output: bool,
) -> None:
    """Run a financial research query."""
    async def _run_query() -> None:
        from src.api.dependencies import initialise_services
        from src.utils.config import get_settings
        from src.agents.research_agent import QueryFilters

        settings = get_settings()
        await initialise_services(settings)

        from src.api.dependencies import get_research_agent
        agent = get_research_agent()

        filters = QueryFilters(
            ticker=ticker.upper() if ticker else None,
            filing_type=filing_type.upper() if filing_type else None,
        )

        with console.status("[bold green]Researching…"):
            response = await agent.query(question=question, filters=filters, top_k=top_k)

        if json_output:
            click.echo(json.dumps(response.__dict__, indent=2, default=str))
            return

        console.print(Panel(
            Markdown(response.answer),
            title=f"[bold blue]Answer[/bold blue] — {response.latency_ms}ms",
            border_style="blue",
        ))

        if response.citations:
            table = Table(title="Sources", show_lines=True)
            table.add_column("#", style="dim", width=3)
            table.add_column("Company", style="cyan")
            table.add_column("Filing", style="green")
            table.add_column("Period")
            table.add_column("Relevance", justify="right")

            for c in response.citations[:8]:
                table.add_row(
                    str(c["citation_number"]),
                    c["company"],
                    c["filing_type"],
                    c.get("period") or "N/A",
                    f"{c['relevance_score']:.3f}",
                )
            console.print(table)

    _run(_run_query())


# ── Index group ──────────────────────────────────────────────────────

@cli.group()
def index() -> None:
    """Manage the vector index."""


@index.command("stats")
def index_stats() -> None:
    """Print current index statistics."""
    from src.retrieval.faiss_store import FAISSStore
    from src.retrieval.bm25_retriever import BM25Retriever

    faiss = FAISSStore.from_disk()
    bm25 = BM25Retriever.from_disk()

    table = Table(title="Index Statistics")
    table.add_column("Component", style="cyan")
    table.add_column("Count", style="green", justify="right")
    table.add_row("FAISS vectors", str(faiss.total_vectors))
    table.add_row("BM25 chunks", str(bm25.total_chunks))
    console.print(table)


@index.command("clear")
@click.confirmation_option(prompt="This will delete all indexed data. Are you sure?")
def index_clear() -> None:
    """Delete all index files."""
    from src.utils.config import get_settings
    import shutil

    settings = get_settings()
    paths = [settings.faiss_index_path, settings.bm25_index_path, settings.metadata_store_path]
    for p in paths:
        if p.exists():
            if p.is_dir():
                shutil.rmtree(p)
            else:
                p.unlink()
            console.print(f"  [red]Deleted[/red] {p}")
    console.print("✅ Index cleared.")


# ── Serve ─────────────────────────────────────────────────────────────

@cli.command("serve")
@click.option("--host", default="0.0.0.0", show_default=True)
@click.option("--port", default=8000, show_default=True, type=int)
@click.option("--workers", default=1, show_default=True, type=int)
@click.option("--reload", is_flag=True, default=False)
def serve(host: str, port: int, workers: int, reload: bool) -> None:
    """Start the API server."""
    import uvicorn
    console.print(Panel(
        f"[bold green]Starting Financial Research Copilot API[/bold green]\n"
        f"  Host    : {host}:{port}\n"
        f"  Workers : {workers}\n"
        f"  Docs    : http://{host}:{port}/docs",
        border_style="green",
    ))
    uvicorn.run(
        "src.api.main:app",
        host=host,
        port=port,
        workers=workers,
        reload=reload,
        log_level="info",
    )


if __name__ == "__main__":
    cli()

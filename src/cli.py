"""
Financial Research Copilot — CLI
Commands: ingest (file/dir/sec/macro/sample), query, index (stats/clear), serve
"""
from __future__ import annotations
import asyncio, json
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

console = Console()

def _run(coro):
    return asyncio.run(coro)

async def _init():
    from src.api.dependencies import initialise_services
    from src.utils.config import get_settings
    await initialise_services(get_settings())

@click.group()
@click.version_option("1.0.0", prog_name="Financial Research Copilot")
def cli():
    """📊 Financial Research Copilot — hybrid RAG for SEC filings, earnings & macro reports."""

# ── Ingest ─────────────────────────────────────────────────────────
@cli.group()
def ingest():
    """Ingest documents into the vector index."""

@ingest.command("file")
@click.argument("path", type=click.Path(exists=True, path_type=Path))
@click.option("--ticker", default=None)
@click.option("--filing-type", default=None)
def ingest_file(path: Path, ticker: Optional[str], filing_type: Optional[str]):
    """Ingest a single file."""
    async def _go():
        await _init()
        from src.api.dependencies import get_ingestion_pipeline
        meta = {}
        if ticker: meta["ticker"] = ticker.upper()
        if filing_type: meta["filing_type"] = filing_type.upper()
        with console.status(f"Ingesting {path.name}…"):
            r = await get_ingestion_pipeline().ingest_file(path, metadata=meta)
        if r.status == "success":
            console.print(Panel(f"✅ Chunks: {r.chunks_added} | Tokens: {r.tokens_indexed}", title=path.name))
        else:
            console.print(f"[red]Error:[/red] {r.error}")
    _run(_go())

@ingest.command("dir")
@click.argument("directory", type=click.Path(exists=True, path_type=Path))
@click.option("--glob", default="**/*.txt", show_default=True)
def ingest_directory(directory: Path, glob: str):
    """Ingest all matching files in a directory."""
    async def _go():
        await _init()
        from src.api.dependencies import get_ingestion_pipeline
        with console.status("Ingesting directory…"):
            s = await get_ingestion_pipeline().ingest_directory(directory, glob=glob)
        t = Table(title="Ingestion Summary")
        t.add_column("Metric", style="cyan"); t.add_column("Value", style="green")
        t.add_row("Documents", str(s.total_documents))
        t.add_row("Chunks", str(s.total_chunks))
        t.add_row("Tokens", str(s.total_tokens))
        t.add_row("Errors", str(len(s.errors)))
        console.print(t)
        for e in s.errors: console.print(f"  [red]⚠[/red] {e}")
    _run(_go())

@ingest.command("sec")
@click.option("--ticker", required=True)
@click.option("--filing-type", default="10-K", type=click.Choice(["10-K","10-Q","8-K"]))
@click.option("--limit", default=3, type=int)
def ingest_sec(ticker: str, filing_type: str, limit: int):
    """Fetch & ingest SEC EDGAR filings."""
    async def _go():
        await _init()
        from src.api.dependencies import get_ingestion_pipeline
        from src.ingestion.sec_fetcher import SECFetcher
        pipeline = get_ingestion_pipeline()
        with console.status(f"Fetching {ticker} {filing_type}…"):
            async with SECFetcher() as f:
                filings = await f.get_filings(ticker, filing_type=filing_type, limit=limit) # type: ignore
        console.print(f"Fetched [bold]{len(filings)}[/bold] filings")
        for filing in filings:
            with console.status(f"Ingesting {filing.doc_id}…"):
                r = await pipeline.ingest_text(doc_id=filing.doc_id, text=filing.text, metadata=filing.metadata)
            icon = "✅" if r.status == "success" else "❌"
            console.print(f"  {icon} {filing.doc_id} — {r.chunks_added} chunks")
    _run(_go())

@ingest.command("macro")
@click.option("--series", multiple=True, help="FRED series IDs (e.g. GDP CPIAUCSL). Leave empty for all defaults.")
@click.option("--start", default="2018-01-01", show_default=True)
def ingest_macro(series: tuple, start: str):
    """Fetch & ingest FRED macroeconomic data."""
    async def _go():
        await _init()
        from src.api.dependencies import get_ingestion_pipeline
        from src.ingestion.fred_fetcher import FREDFetcher, MACRO_SERIES
        from src.utils.config import get_settings
        pipeline = get_ingestion_pipeline()
        api_key = get_settings().fred_api_key_str
        targets = None
        if series:
            series_map = {sid: (name, cat) for sid, name, cat in MACRO_SERIES}
            targets = [(s, series_map.get(s, (s, "macro"))[0], series_map.get(s, (s, "macro"))[1]) for s in series]
        with console.status("Fetching FRED macro data…"):
            async with FREDFetcher(api_key=api_key) as fetcher:
                docs = await fetcher.fetch_all(series=targets, observation_start=start)
        console.print(f"Fetched [bold]{len(docs)}[/bold] macro series")
        for doc in docs:
            with console.status(f"Ingesting {doc.series_id}…"):
                r = await pipeline.ingest_text(doc_id=doc.doc_id, text=doc.text, metadata=doc.metadata)
            icon = "✅" if r.status == "success" else "❌"
            console.print(f"  {icon} {doc.series_id} ({doc.title[:50]}) — {r.chunks_added} chunks")
    _run(_go())

@ingest.command("sample")
def ingest_sample():
    """Ingest bundled sample documents in data/sample/."""
    sample_dir = Path("data/sample")
    if not sample_dir.exists() or not any(sample_dir.glob("**/*.txt")):
        console.print("[yellow]No sample .txt files found in data/sample/[/yellow]")
        return
    async def _go():
        await _init()
        from src.api.dependencies import get_ingestion_pipeline
        s = await get_ingestion_pipeline().ingest_directory(sample_dir, glob="**/*.txt")
        console.print(f"✅ Sample ingestion complete: {s.total_chunks} chunks from {s.total_documents} docs")
    _run(_go())

# ── Query ─────────────────────────────────────────────────────────────
@cli.command("query")
@click.argument("question")
@click.option("--ticker", default=None)
@click.option("--filing-type", default=None)
@click.option("--doc-type", default=None)
@click.option("--top-k", default=None, type=int)
@click.option("--json-output", is_flag=True, default=False)
def query_cmd(question: str, ticker: Optional[str], filing_type: Optional[str],
              doc_type: Optional[str], top_k: Optional[int], json_output: bool):
    """Run a financial research query against the index."""
    async def _go():
        await _init()
        from src.api.dependencies import get_research_agent
        from src.agents.research_agent import QueryFilters
        filters = QueryFilters(
            ticker=ticker.upper() if ticker else None,
            filing_type=filing_type.upper() if filing_type else None,
            doc_type=doc_type,
        )
        with console.status("[bold green]Researching…"):
            r = await get_research_agent().query(question=question, filters=filters, top_k=top_k)
        if json_output:
            click.echo(json.dumps(r.__dict__, indent=2, default=str)); return
        console.print(Panel(Markdown(r.answer),
                            title=f"[bold blue]Answer[/bold blue] — {r.latency_ms}ms",
                            border_style="blue"))
        if r.citations:
            t = Table(title="Sources", show_lines=True)
            t.add_column("#", style="dim", width=3); t.add_column("Company", style="cyan")
            t.add_column("Filing", style="green"); t.add_column("Period"); t.add_column("Score", justify="right")
            for c in r.citations[:8]:
                t.add_row(str(c["citation_number"]), c["company"],
                          c["filing_type"], c.get("period") or "N/A", f"{c['relevance_score']:.3f}")
            console.print(t)
    _run(_go())

# ── Index ─────────────────────────────────────────────────────────────
@cli.group()
def index():
    """Manage the vector index."""

@index.command("stats")
def index_stats():
    """Print current index statistics."""
    async def _go():
        await _init()
        from src.api.dependencies import get_faiss_store, get_bm25_retriever, get_metadata_db
        faiss = get_faiss_store(); bm25 = get_bm25_retriever(); db = get_metadata_db()
        db_stats = db.stats()
        t = Table(title="Index Statistics")
        t.add_column("Component", style="cyan"); t.add_column("Count", style="green", justify="right")
        t.add_row("FAISS vectors",     str(faiss.total_vectors))
        t.add_row("BM25 chunks",       str(bm25.total_chunks))
        t.add_row("SQLite chunks",     str(db_stats.get("total_chunks", 0)))
        t.add_row("SQLite documents",  str(db_stats.get("total_documents", 0)))
        t.add_row("Unique tickers",    str(db_stats.get("total_tickers", 0)))
        console.print(t)
        by_type = db_stats.get("by_filing_type", {})
        if by_type:
            t2 = Table(title="By Filing Type")
            t2.add_column("Type", style="cyan"); t2.add_column("Chunks", style="green", justify="right")
            for ftype, n in sorted(by_type.items()): t2.add_row(ftype, str(n))
            console.print(t2)
    _run(_go())

@index.command("clear")
@click.confirmation_option(prompt="Delete all indexed data — are you sure?")
def index_clear():
    """Delete all index files."""
    import shutil
    from src.utils.config import get_settings
    s = get_settings()
    for p in [s.faiss_index_path, s.bm25_index_path, s.metadata_db_path]:
        if p.exists():
            if p.is_dir(): shutil.rmtree(p)
            else: p.unlink()
            console.print(f"  [red]Deleted[/red] {p}")
    console.print("✅ Index cleared.")

# ── Serve ─────────────────────────────────────────────────────────────
@cli.command("serve")
@click.option("--host", default="0.0.0.0", show_default=True)
@click.option("--port", default=8000, show_default=True, type=int)
@click.option("--workers", default=1, show_default=True, type=int)
@click.option("--reload", is_flag=True, default=False)
def serve(host: str, port: int, workers: int, reload: bool):
    """Start the API server."""
    import uvicorn
    console.print(Panel(
        f"[bold green]Financial Research Copilot API[/bold green]\n"
        f"  http://{host}:{port}\n"
        f"  Docs: http://{host}:{port}/docs",
        border_style="green",
    ))
    uvicorn.run("src.api.main:app", host=host, port=port, workers=workers, reload=reload)

if __name__ == "__main__":
    cli()

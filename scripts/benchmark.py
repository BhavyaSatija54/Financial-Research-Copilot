"""
scripts/benchmark.py

Retrieval quality benchmarking against a golden QA dataset.

Usage:
    python scripts/benchmark.py --qa-file data/sample/qa_golden.json --top-k 5 8

Output:
    Precision@k, Recall@k, MRR, NDCG@k for each k value.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from rich.console import Console
from rich.table import Table

from src.ingestion.embedder import Embedder
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.faiss_store import FAISSStore
from src.retrieval.hybrid import HybridRetriever, build_filters
from src.utils.config import get_settings
from src.utils.metrics import ndcg_at_k, mean_reciprocal_rank, precision_at_k, recall_at_k

console = Console()


async def run_benchmark(qa_file: Path, top_k_values: list[int]) -> None:
    """
    Run retrieval benchmark against a golden QA dataset.

    Expected format of qa_golden.json:
    [
      {
        "question": "What are Apple's risk factors?",
        "relevant_chunk_ids": ["AAPL_10K_2024_chunk_001", "AAPL_10K_2024_chunk_008"],
        "filters": {"ticker": "AAPL"}
      },
      ...
    ]
    """
    if not qa_file.exists():
        console.print(f"[red]QA file not found: {qa_file}[/red]")
        console.print("Create data/sample/qa_golden.json with your golden QA pairs.")
        return

    with open(qa_file) as f:
        qa_pairs = json.load(f)

    console.print(f"Loaded [bold]{len(qa_pairs)}[/bold] QA pairs")

    # Load indexes
    faiss = FAISSStore.from_disk()
    bm25 = BM25Retriever.from_disk()
    embedder = Embedder()
    retriever = HybridRetriever(faiss, bm25)

    metrics: dict[int, dict[str, list[float]]] = {
        k: {"precision": [], "recall": [], "mrr": [], "ndcg": []}
        for k in top_k_values
    }

    for i, qa in enumerate(qa_pairs, 1):
        question = qa["question"]
        relevant = qa.get("relevant_chunk_ids", [])
        filters = build_filters(**qa.get("filters", {})) if qa.get("filters") else None

        query_vec = await embedder.embed_query(question)
        results = retriever.retrieve(question, query_vec, top_k=max(top_k_values), filters=filters)
        retrieved_ids = [r.chunk_id for r in results]

        for k in top_k_values:
            metrics[k]["precision"].append(precision_at_k(retrieved_ids, relevant, k))
            metrics[k]["recall"].append(recall_at_k(retrieved_ids, relevant, k))
            metrics[k]["mrr"].append(mean_reciprocal_rank(retrieved_ids, relevant))
            metrics[k]["ndcg"].append(ndcg_at_k(retrieved_ids, relevant, k))

        if i % 10 == 0:
            console.print(f"  Evaluated {i}/{len(qa_pairs)}…")

    # Results table
    table = Table(title="Retrieval Benchmark Results", show_lines=True)
    table.add_column("k", style="cyan", justify="center")
    table.add_column("Precision@k", justify="right")
    table.add_column("Recall@k", justify="right")
    table.add_column("MRR", justify="right")
    table.add_column("NDCG@k", justify="right")

    for k in top_k_values:
        m = metrics[k]
        table.add_row(
            str(k),
            f"{np.mean(m['precision']):.4f}",
            f"{np.mean(m['recall']):.4f}",
            f"{np.mean(m['mrr']):.4f}",
            f"{np.mean(m['ndcg']):.4f}",
        )

    console.print(table)
    console.print(f"\n[dim]Evaluated on {len(qa_pairs)} questions[/dim]")


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval quality benchmark")
    parser.add_argument(
        "--qa-file",
        type=Path,
        default=Path("data/sample/qa_golden.json"),
    )
    parser.add_argument(
        "--top-k",
        nargs="+",
        type=int,
        default=[5, 8],
        dest="top_k_values",
    )
    args = parser.parse_args()
    asyncio.run(run_benchmark(args.qa_file, args.top_k_values))


if __name__ == "__main__":
    main()

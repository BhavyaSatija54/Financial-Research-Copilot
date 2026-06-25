"""Document ingestion sub-package."""
from src.ingestion.chunker import DocumentChunker, TextChunk
from src.ingestion.embedder import Embedder
from src.ingestion.pipeline import IngestionPipeline, IngestionResult, PipelineStats
from src.ingestion.sec_fetcher import SECFetcher, SECFiling
from src.ingestion.fred_fetcher import FREDFetcher, MacroDocument

__all__ = [
    "DocumentChunker", "TextChunk",
    "Embedder",
    "IngestionPipeline", "IngestionResult", "PipelineStats",
    "SECFetcher", "SECFiling",
    "FREDFetcher", "MacroDocument",
]

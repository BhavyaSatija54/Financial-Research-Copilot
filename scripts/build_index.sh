#!/usr/bin/env bash
# =============================================================================
# scripts/build_index.sh
#
# One-shot script to download sample SEC filings and build the full index.
# Usage: ./scripts/build_index.sh [--tickers "AAPL MSFT GOOGL"] [--years 2022 2023 2024]
# =============================================================================

set -euo pipefail

TICKERS="${TICKERS:-AAPL MSFT GOOGL AMZN NVDA}"
FILING_TYPES="${FILING_TYPES:-10-K 10-Q}"
LIMIT="${LIMIT:-3}"
SAMPLE_DIR="data/sample"
INDEX_DIR="data/processed"

echo "════════════════════════════════════════════════"
echo "  Financial Research Copilot — Index Builder"
echo "════════════════════════════════════════════════"

# Ensure environment
if [ ! -f ".env" ]; then
    echo "⚠  .env not found. Copying .env.example → .env"
    cp .env.example .env
    echo "   Edit .env with your OPENAI_API_KEY before continuing."
    exit 1
fi

source .env
if [ -z "${OPENAI_API_KEY:-}" ] || [ "$OPENAI_API_KEY" = "sk-your-key-here" ]; then
    echo "❌  OPENAI_API_KEY not set in .env"
    exit 1
fi

mkdir -p "$SAMPLE_DIR" "$INDEX_DIR"

echo ""
echo "Step 1: Fetching SEC filings…"
echo "  Tickers : $TICKERS"
echo "  Types   : $FILING_TYPES"
echo "  Limit   : $LIMIT per ticker/type"
echo ""

for TICKER in $TICKERS; do
    for FILING_TYPE in $FILING_TYPES; do
        echo "  → $TICKER $FILING_TYPE"
        python -m src.cli ingest sec \
            --ticker "$TICKER" \
            --filing-type "$FILING_TYPE" \
            --limit "$LIMIT" || echo "  ⚠ Failed for $TICKER $FILING_TYPE (skipping)"
    done
done

echo ""
echo "Step 2: Ingesting any additional local files in $SAMPLE_DIR…"
if [ "$(ls -A "$SAMPLE_DIR" 2>/dev/null)" ]; then
    python -m src.cli ingest dir "$SAMPLE_DIR" --glob "**/*.txt"
else
    echo "  No additional local files found."
fi

echo ""
echo "Step 3: Index statistics"
python -m src.cli index stats

echo ""
echo "✅  Index build complete!"
echo "   Start the API with: python -m src.cli serve"

# ═══════════════════════════════════════════════════════════════════
#  Financial Research Copilot — Makefile
# ═══════════════════════════════════════════════════════════════════
.DEFAULT_GOAL := help
SHELL         := /bin/bash
PYTHON        := python
PIP           := pip
VENV          := venv
VENV_BIN      := $(VENV)/bin

.PHONY: help install install-dev venv run-api run-ui run \
        ingest-sample ingest-sec ingest-macro \
        test test-unit test-integration \
        lint format type-check \
        docker-build docker-run docker-stop \
        index-stats index-clear clean

# ── Help ─────────────────────────────────────────────────────────────
help: ## Show this help
	@echo ""
	@echo "  📊 Financial Research Copilot"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'
	@echo ""

# ── Setup ────────────────────────────────────────────────────────────
venv: ## Create virtual environment
	$(PYTHON) -m venv $(VENV)
	@echo "✅ Virtual env created. Activate with: source $(VENV)/bin/activate"

install: ## Install production dependencies
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "✅ Dependencies installed"

install-dev: ## Install dev + production dependencies
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements-dev.txt
	pre-commit install
	@echo "✅ Dev dependencies installed"

env: ## Copy .env.example → .env (only if .env doesn't exist)
	@test -f .env || (cp .env.example .env && echo "✅ .env created — add your OPENAI_API_KEY")

# ── Run ──────────────────────────────────────────────────────────────
run-api: ## Start FastAPI server (port 8000)
	@echo "🚀 Starting API server at http://localhost:8000"
	@echo "   Docs: http://localhost:8000/docs"
	uvicorn src.api.main:app \
		--host 0.0.0.0 \
		--port 8000 \
		--reload \
		--log-level info

run-ui: ## Start Streamlit UI (port 8501)
	@echo "🎨 Starting Streamlit UI at http://localhost:8501"
	streamlit run src/ui/app.py \
		--server.port 8501 \
		--server.headless true

run: ## Start both API + UI (requires tmux or two terminals)
	@echo "Starting API in background..."
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload &
	@sleep 3
	@echo "Starting Streamlit UI..."
	streamlit run src/ui/app.py --server.port 8501 --server.headless true

# ── Ingestion ────────────────────────────────────────────────────────
ingest-sample: ## Ingest bundled sample documents
	$(PYTHON) -m src.cli ingest sample
	@echo "✅ Sample data ingested"

ingest-sec: ## Ingest SEC filings (usage: make ingest-sec TICKER=AAPL TYPE=10-K LIMIT=3)
	$(PYTHON) -m src.cli ingest sec \
		--ticker  $(or $(TICKER),AAPL) \
		--filing-type $(or $(TYPE),10-K) \
		--limit   $(or $(LIMIT),3)

ingest-macro: ## Ingest FRED macro data
	$(PYTHON) -m src.cli ingest macro

# ── Index Management ─────────────────────────────────────────────────
index-stats: ## Show index statistics
	$(PYTHON) -m src.cli index stats

index-clear: ## Clear all index data (requires confirmation)
	$(PYTHON) -m src.cli index clear

# ── Testing ──────────────────────────────────────────────────────────
test: ## Run full test suite with coverage
	pytest tests/ -v \
		--cov=src \
		--cov-report=term-missing \
		--cov-report=html:htmlcov \
		-m "not integration"
	@echo "✅ Tests complete. Coverage report: htmlcov/index.html"

test-unit: ## Run unit tests only
	pytest tests/unit/ -v --tb=short

test-integration: ## Run integration tests (requires API keys)
	pytest tests/integration/ -v --tb=short

# ── Code Quality ─────────────────────────────────────────────────────
lint: ## Run ruff linter
	ruff check src/ tests/
	@echo "✅ Linting passed"

format: ## Auto-format with black + ruff
	black src/ tests/
	ruff check --fix src/ tests/
	@echo "✅ Formatting done"

type-check: ## Run mypy type checker
	mypy src/ --ignore-missing-imports
	@echo "✅ Type check done"

check: lint type-check test-unit ## Run all checks (lint + types + unit tests)

# ── Docker ───────────────────────────────────────────────────────────
docker-build: ## Build Docker image
	docker build -t financial-research-copilot:latest .
	@echo "✅ Docker image built"

docker-run: ## Run with Docker Compose (API + Redis)
	docker-compose --profile cache up --build -d
	@echo "✅ Services started"
	@echo "   API: http://localhost:8000"
	@echo "   Docs: http://localhost:8000/docs"

docker-run-ui: ## Run API + Redis + UI via Docker
	docker-compose up --build -d
	streamlit run src/ui/app.py --server.port 8501 &
	@echo "✅ All services started"

docker-stop: ## Stop all Docker services
	docker-compose down
	@echo "✅ Services stopped"

docker-logs: ## Tail Docker logs
	docker-compose logs -f api

# ── Benchmark ────────────────────────────────────────────────────────
benchmark: ## Run retrieval quality benchmark
	$(PYTHON) scripts/benchmark.py \
		--qa-file data/sample/qa_golden.json \
		--top-k 5 8

# ── Clean ─────────────────────────────────────────────────────────────
clean: ## Remove build artifacts and caches
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage coverage.xml
	@echo "✅ Cleaned"

clean-index: ## Remove FAISS/BM25/SQLite index files
	rm -rf data/processed/
	mkdir -p data/processed/
	@echo "✅ Index cleared"

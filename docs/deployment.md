# Deployment Guide

## Local Development

```bash
# 1. Install
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Configure
cp .env.example .env   # add your OPENAI_API_KEY

# 3. Ingest sample data
python -m src.cli ingest sample

# 4. Run API
uvicorn src.api.main:app --reload --port 8000
```

## Docker (Single Container)

```bash
docker build -t frc:latest .
docker run -p 8000:8000 \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  -e API_KEY=your-secret \
  -v $(pwd)/data:/app/data \
  frc:latest
```

## Docker Compose (with Redis cache)

```bash
cp .env.example .env          # fill in secrets
docker-compose --profile cache up --build -d
```

## Environment Variables (Production Checklist)

| Variable | Required | Notes |
|---|---|---|
| `OPENAI_API_KEY` | ✅ | Never commit |
| `API_KEY` | ✅ | Rotate regularly |
| `JWT_SECRET_KEY` | ✅ | Min 32 chars |
| `REDIS_URL` | Recommended | Enables caching |
| `ENVIRONMENT` | ✅ | Set to `production` |
| `LOG_FORMAT` | Optional | `json` for log aggregators |

## Scaling

- **Horizontal:** Run multiple API containers behind a load balancer; mount `data/processed/` from a shared volume (NFS/EFS).
- **GPU reranking:** Install `sentence-transformers` with CUDA; the cross-encoder runs automatically on GPU if available.
- **Large index (>5M vectors):** Replace `IndexFlatIP` with `IndexIVFFlat(nlist=1024)` in `faiss_store.py`.

## Health Monitoring

```bash
curl http://localhost:8000/health
# Returns index stats, cache status, latency percentiles
```

Integrate with Prometheus by scraping `/health` and parsing the JSON response.

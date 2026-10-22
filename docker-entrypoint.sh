#!/bin/sh
# Bootstrap: seed DB (idempotent), build the vector index only when missing/stale, then serve.
set -e
python -m scripts.seed_db
if ! python -m scripts.build_index; then
  echo "WARNING: index build failed (embedding model unavailable?). Service starts degraded; /health will report 503." >&2
fi
exec uvicorn app.main:app --host 0.0.0.0 --port 8000

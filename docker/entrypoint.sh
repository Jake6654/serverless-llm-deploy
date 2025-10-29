#!/usr/bin/env bash
set -e

if [ "$MODE" = "serve" ]; then
  echo "[entrypoint] Starting Ray Serve..."
  
  # [수정됨] 최신 Ray 버전에 맞게 --host 옵션을 명시합니다.
  serve run --host 0.0.0.0 --port 8000 api.serve_app:deployment

else
  echo "[entrypoint] Starting FastAPI (uvicorn)..."
  exec uvicorn api.app:app --host 0.0.0.0 --port 8000
fi
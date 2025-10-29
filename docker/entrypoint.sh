#!/usr/bin/env bash
set -e

if [ "$MODE" = "serve" ]; then
  echo "[entrypoint] Starting Ray Serve..."
  # 대시보드는 필요시만 노출
  ray start --head --dashboard-host 0.0.0.0 || true
  # Ray Serve 배포 시작
  serve run api.serve_app:deployment
else
  echo "[entrypoint] Starting FastAPI (uvicorn)..."
  exec uvicorn api.app:app --host 0.0.0.0 --port 8000
fi

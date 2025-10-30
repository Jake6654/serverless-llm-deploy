#!/usr/bin/env bash
set -e

if [ "$MODE" = "serve" ]; then
  echo "[entrypoint] Starting Ray Serve..."

  # 0) 이전 Ray 프로세스 정리
  ray stop --force || true

  # 1) 로컬 Ray 헤드 기동 (대시보드 외부 접근 원하면 8265 매핑)
  ray start --head --dashboard-host=0.0.0.0 --dashboard-port=8265 --disable-usage-stats || true

  # 2) Serve HTTP를 0.0.0.0:8000에 명시적으로 바인드 (기존 127.0.0.1 이슈 방지)
  python3 - <<'PY'
import ray
from ray import serve
ray.init(address="auto")  # 컨테이너 안의 헤드(로컬 클러스터)에 붙음
serve.start(http_options={"host": "0.0.0.0", "port": 8000})
print("[entrypoint] Serve HTTP bound to 0.0.0.0:8000")
PY

  # 3) 앱 실행 (블로킹). RAY_ADDRESS는 절~대 http로 넘기지 말 것 (auto가 기본)
  exec serve run --route-prefix / api.serve_app:deployment

else
  echo "[entrypoint] Starting FastAPI (uvicorn)..."
  exec uvicorn api.app:app --host 0.0.0.0 --port 8000
fi

#!/usr/bin/env bash
# 로컬망에서 접속 가능하도록 0.0.0.0:8000 으로 실행합니다.
set -e
cd "$(dirname "$0")"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
exec python3 -m uvicorn app.main:app --host "$HOST" --port "$PORT" --timeout-keep-alive 120

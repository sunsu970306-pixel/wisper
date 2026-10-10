#!/usr/bin/env bash
# 설치가 끝난 뒤 서버만 실행합니다. 처음이라면 ./start.sh 를 사용하세요.
set -e
cd "$(dirname "$0")"
PY=python3; [ -x .venv/bin/python ] && PY=.venv/bin/python
exec "$PY" -m uvicorn app.main:app --host "${HOST:-0.0.0.0}" --port "${PORT:-8000}" --timeout-keep-alive 120

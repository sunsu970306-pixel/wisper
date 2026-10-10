#!/usr/bin/env bash
# Whisper 음성 변환 - 자동 설치 및 실행 (Linux / macOS)
set -e
cd "$(dirname "$0")"
PORT="${PORT:-8000}"

echo "================================================"
echo " Whisper 음성 변환 - 자동 설치 및 실행"
echo "================================================"

# ---------------------------------------------- Python
PY=""
for c in python3.13 python3.12 python3.11 python3.10 python3; do
  if command -v "$c" >/dev/null && "$c" -c 'import sys; sys.exit(sys.version_info < (3, 10))'; then PY="$c"; break; fi
done
if [ -z "$PY" ]; then
  echo "[오류] Python 3.10 이상이 필요합니다."
  echo "  Ubuntu: sudo apt install python3 python3-venv   /   macOS: brew install python"
  exit 1
fi

# ---------------------------------------------- ffmpeg
if ! command -v ffmpeg >/dev/null || ! command -v ffprobe >/dev/null; then
  echo "[설치] ffmpeg 가 없어 설치합니다..."
  if command -v apt-get >/dev/null; then sudo apt-get update && sudo apt-get install -y ffmpeg
  elif command -v dnf >/dev/null; then sudo dnf install -y ffmpeg
  elif command -v pacman >/dev/null; then sudo pacman -S --noconfirm ffmpeg
  elif command -v brew >/dev/null; then brew install ffmpeg
  else echo "[오류] ffmpeg 를 직접 설치해 주세요."; exit 1; fi
fi

# ---------------------------------------------- 가상환경 + 패키지
if [ ! -x .venv/bin/python ]; then
  echo "[설치] 가상환경 생성 중..."
  "$PY" -m venv .venv || { echo "[오류] venv 생성 실패. Ubuntu 라면: sudo apt install python3-venv"; exit 1; }
fi
VPY=.venv/bin/python
if [ ! -f .venv/.installed ]; then
  echo "[설치] Python 패키지 설치 중... (처음 한 번만, 몇 분 걸립니다)"
  "$VPY" -m pip install --upgrade pip >/dev/null
  "$VPY" -m pip install -r requirements.txt
  if command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1; then
    echo "[설치] NVIDIA GPU 감지 - CUDA 라이브러리 설치 중... (약 1GB)"
    "$VPY" -m pip install nvidia-cublas-cu12 "nvidia-cudnn-cu12==9.*"
  fi
  touch .venv/.installed
fi

# ---------------------------------------------- 사양 확인 + 모델 미리 받기
echo
"$VPY" -m app.hardware
echo
echo "[준비] 추천 모델 다운로드/로딩 확인 중... (처음에는 수 GB 다운로드)"
"$VPY" -m app.prepare

# ---------------------------------------------- 실행
IP=$( (hostname -I 2>/dev/null || ipconfig getifaddr en0 2>/dev/null) | awk '{print $1}')
echo
echo "================================================"
echo " 서버 실행 중 - Ctrl+C 로 종료"
echo " 이 PC:        http://localhost:$PORT"
[ -n "$IP" ] && echo " 같은 망 기기: http://$IP:$PORT"
echo "================================================"
exec "$VPY" -m uvicorn app.main:app --host 0.0.0.0 --port "$PORT" --timeout-keep-alive 120

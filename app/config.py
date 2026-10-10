"""환경변수 기반 설정."""
import os
from pathlib import Path

from .hardware import recommend


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default).strip() or default


DATA_DIR = Path(_env("WHISPER_DATA_DIR", str(Path(__file__).resolve().parent.parent / "data")))
UPLOAD_DIR = DATA_DIR / "uploads"
JOB_DIR = DATA_DIR / "jobs"

# 아래 값들은 "auto" 이면 실행 중인 컴퓨터 사양(hardware.py)에 맞춰 자동으로 정해진다.
_rec = recommend()

# faster-whisper 모델 이름 또는 로컬 경로 (tiny, base, small, medium, large-v3, turbo ...)
MODEL = _rec["model"] if _env("WHISPER_MODEL", "auto") == "auto" else _env("WHISPER_MODEL", "auto")
DEVICE = _rec["device"] if _env("WHISPER_DEVICE", "auto") == "auto" else _env("WHISPER_DEVICE", "auto")
_compute = _env("WHISPER_COMPUTE_TYPE", "auto")
if _compute == "auto":
    _compute = _rec["compute_type"] if DEVICE == _rec["device"] else ("float16" if DEVICE == "cuda" else "int8")
COMPUTE_TYPE = _compute
MODEL_DIR = os.environ.get("WHISPER_MODEL_DIR") or None  # 모델 다운로드/캐시 위치
CPU_THREADS = int(_env("WHISPER_CPU_THREADS", "0")) or _rec["cpu_threads"]
DEFAULT_LANGUAGE = _env("WHISPER_LANGUAGE", "ko")  # "auto" 이면 자동 감지
BEAM_SIZE = int(_env("WHISPER_BEAM_SIZE", "0")) or _rec["beam_size"]
# 배치 추론 크기 (1 이면 배치 없이 순차 처리). 여러 30초 구간을 한 번에 돌려 처리 속도를 높인다.
BATCH_SIZE = int(_env("WHISPER_BATCH_SIZE", "0")) or _rec["batch_size"]

# 긴 파일은 이 길이(초) 단위로 잘라서 순차 처리 → 메모리 사용량이 파일 길이와 무관하게 일정
SEGMENT_SECONDS = int(_env("WHISPER_SEGMENT_SECONDS", "1200"))

# 업로드 청크 크기(바이트)와 최대 파일 크기(바이트, 0 = 무제한)
CHUNK_SIZE = int(_env("WHISPER_CHUNK_SIZE", str(16 * 1024 * 1024)))
MAX_UPLOAD_BYTES = int(_env("WHISPER_MAX_UPLOAD_BYTES", "0"))

# 변환 완료 후 원본 음성 파일을 지울지 여부
DELETE_SOURCE_AFTER = _env("WHISPER_DELETE_SOURCE", "false").lower() in ("1", "true", "yes")

for d in (UPLOAD_DIR, JOB_DIR):
    d.mkdir(parents=True, exist_ok=True)

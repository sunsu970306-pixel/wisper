"""변환 작업 큐와 백그라운드 워커.

처리 흐름
  1. ffmpeg 로 원본(어떤 포맷이든)을 16kHz mono wav 조각(SEGMENT_SECONDS 단위)으로 분할
  2. 조각을 하나씩 faster-whisper 로 변환하고 결과를 조각별 JSON 으로 저장
  3. 모든 조각이 끝나면 타임스탬프를 이어붙여 txt / srt / vtt / json 생성

조각 단위로 결과를 저장하므로 서버가 중간에 꺼져도 재시작 시 남은 조각부터 이어서 처리한다.
"""
import json
import logging
import os
import queue
import re
import shutil
import subprocess
import threading
import time
import uuid
import wave
from pathlib import Path

from . import config

log = logging.getLogger("whisper.jobs")

ACTIVE = ("queued", "preparing", "transcribing")
MODELS = ["tiny", "base", "small", "medium", "large-v2", "large-v3", "turbo"]


class Cancelled(Exception):
    pass


# --------------------------------------------------------------------------- 상태 저장

_jobs: dict[str, dict] = {}
_live: dict[str, list] = {}  # job_id -> 지금까지 나온 세그먼트 (미리보기용)
_cancel: set[str] = set()
_lock = threading.RLock()
_queue: "queue.Queue[str]" = queue.Queue()


def _job_dir(job_id: str) -> Path:
    if not re.fullmatch(r"[0-9a-f]{32}", job_id):
        raise KeyError(job_id)
    return config.JOB_DIR / job_id


def _persist(job: dict) -> None:
    p = _job_dir(job["id"]) / "job.json"
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(job, ensure_ascii=False))
    os.replace(tmp, p)


def _update(job_id: str, **fields) -> dict:
    with _lock:
        job = _jobs[job_id]
        job.update(fields)
        _persist(job)
        return dict(job)


def public(job: dict) -> dict:
    j = dict(job)
    j.pop("source", None)
    if j["status"] == "transcribing" and j.get("progress", 0) > 0.01 and j.get("transcribe_started"):
        elapsed = time.time() - j["transcribe_started"]
        done_frac = j["progress"] - j.get("progress_at_start", 0)
        if done_frac > 0.005:
            j["eta_seconds"] = elapsed / done_frac * (1 - j["progress"])
    return j


def list_jobs() -> list[dict]:
    with _lock:
        return [public(j) for j in sorted(_jobs.values(), key=lambda j: j["created"], reverse=True)]


def get(job_id: str) -> dict:
    with _lock:
        if job_id not in _jobs:
            raise KeyError(job_id)
        return public(_jobs[job_id])


def segments(job_id: str, offset: int = 0) -> list:
    with _lock:
        if job_id not in _jobs:
            raise KeyError(job_id)
        live = _live.get(job_id)
        if live is None:
            live = _live[job_id] = _load_segments(job_id)
        return live[offset:]


def create(filename: str, move_source, language: str, model: str, timestamps_in_txt: bool = False) -> dict:
    job_id = uuid.uuid4().hex
    d = config.JOB_DIR / job_id
    d.mkdir(parents=True)
    ext = Path(filename).suffix.lower()[:10] or ".bin"
    source = d / f"source{ext}"
    move_source(source)
    job = {
        "id": job_id,
        "filename": filename,
        "source": source.name,
        "size": source.stat().st_size,
        "language": language,
        "model": model,
        "status": "queued",
        "stage": "대기 중",
        "progress": 0.0,
        "duration": None,
        "created": time.time(),
        "finished": None,
        "error": None,
        "detected_language": None,
    }
    with _lock:
        _jobs[job_id] = job
        _persist(job)
    _queue.put(job_id)
    return public(job)


def cancel(job_id: str) -> dict:
    with _lock:
        job = _jobs[job_id]
        if job["status"] in ACTIVE:
            _cancel.add(job_id)
            if job["status"] == "queued":
                _update(job_id, status="cancelled", stage="취소됨")
        return public(_jobs[job_id])


def retry(job_id: str) -> dict:
    """실패/취소된 작업을 다시 큐에 넣는다. 이미 끝난 조각은 건너뛰고 이어서 처리한다."""
    with _lock:
        job = _jobs[job_id]
        if job["status"] not in ("error", "cancelled"):
            raise ValueError("실패하거나 취소된 작업만 다시 시도할 수 있습니다.")
        if not (_job_dir(job_id) / job["source"]).exists():
            raise ValueError("원본 파일이 없어 다시 시도할 수 없습니다.")
        _cancel.discard(job_id)
        _update(job_id, status="queued", stage="대기 중", error=None, finished=None)
    _queue.put(job_id)
    return get(job_id)


def delete(job_id: str) -> None:
    with _lock:
        job = _jobs[job_id]
        if job["status"] in ("preparing", "transcribing"):
            raise ValueError("진행 중인 작업은 먼저 취소하세요.")
        _cancel.add(job_id)
        _jobs.pop(job_id, None)
        _live.pop(job_id, None)
    shutil.rmtree(_job_dir(job_id), ignore_errors=True)


def output_path(job_id: str, fmt: str) -> Path:
    names = {
        "txt": "transcript.txt",
        "ts.txt": "transcript_timestamps.txt",
        "srt": "transcript.srt",
        "vtt": "transcript.vtt",
        "json": "transcript.json",
    }
    if fmt not in names:
        raise KeyError(fmt)
    return _job_dir(job_id) / names[fmt]


def load_existing() -> None:
    """서버 시작 시 디스크에 남은 작업을 복구하고, 미완료 작업은 다시 큐에 넣는다."""
    pending = []
    for d in config.JOB_DIR.iterdir():
        p = d / "job.json"
        if not p.exists():
            continue
        try:
            job = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if job["status"] in ACTIVE:
            job["status"] = "queued"
            job["stage"] = "재시작 후 대기 중"
            pending.append(job)
        _jobs[job["id"]] = job
        _persist(job)
    for job in sorted(pending, key=lambda j: j["created"]):
        _queue.put(job["id"])


# --------------------------------------------------------------------------- 모델

_model = None
_model_key = None


def _resolve_device() -> tuple[str, str]:
    device = config.DEVICE
    if device == "auto":
        try:
            import ctranslate2

            device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
        except Exception:
            device = "cpu"
    compute = config.COMPUTE_TYPE
    if compute == "default":
        compute = "float16" if device == "cuda" else "int8"
    return device, compute


def get_model(name: str):
    global _model, _model_key
    device, compute = _resolve_device()
    key = (name, device, compute)
    if _model is not None and _model_key == key:
        return _model
    from faster_whisper import WhisperModel

    _model = None  # 이전 모델 메모리 해제
    log.info("모델 로딩: %s (device=%s, compute_type=%s)", name, device, compute)
    _model = WhisperModel(
        name,
        device=device,
        compute_type=compute,
        cpu_threads=config.CPU_THREADS,
        download_root=config.MODEL_DIR,
    )
    _model_key = key
    return _model


def device_info() -> dict:
    device, compute = _resolve_device()
    return {"device": device, "compute_type": compute}


# --------------------------------------------------------------------------- 오디오 처리

def _probe_duration(path: Path) -> float | None:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(path)],
            capture_output=True, text=True, timeout=120,
        ).stdout.strip()
        return float(out)
    except (ValueError, subprocess.SubprocessError, OSError):
        return None


def _split(job_id: str, source: Path, parts_dir: Path, duration: float | None) -> None:
    """원본을 16kHz mono PCM wav 조각으로 분할. 진행률은 ffmpeg -progress 로 읽는다."""
    if parts_dir.exists():
        shutil.rmtree(parts_dir)
    parts_dir.mkdir()
    cmd = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(source), "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
        "-f", "segment", "-segment_time", str(config.SEGMENT_SECONDS), "-reset_timestamps", "1",
        str(parts_dir / "part_%05d.wav"),
        "-progress", "pipe:1",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    last = 0.0
    for line in proc.stdout:
        if job_id in _cancel:
            proc.kill()
            proc.wait()
            raise Cancelled()
        if line.startswith("out_time_us=") and duration:
            try:
                t = int(line.split("=", 1)[1]) / 1e6
            except ValueError:
                continue
            if time.time() - last > 1:
                last = time.time()
                _update(job_id, stage=f"오디오 변환 중 ({min(t / duration, 1) * 100:.0f}%)")
    err = proc.stderr.read()
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg 변환 실패: {err.strip()[-500:] or '알 수 없는 오류'}")


def _wav_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / w.getframerate()


def _load_segments(job_id: str) -> list:
    d = _job_dir(job_id) / "results"
    segs: list = []
    if d.exists():
        for p in sorted(d.glob("part_*.json")):
            segs.extend(json.loads(p.read_text())["segments"])
    return segs


# --------------------------------------------------------------------------- 출력 포맷

def _ts(sec: float, sep: str = ",") -> str:
    ms = int(round(sec * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def _write_outputs(job_id: str, segs: list, meta: dict) -> None:
    d = _job_dir(job_id)
    (d / "transcript.txt").write_text("\n".join(s["text"] for s in segs) + "\n", encoding="utf-8")
    (d / "transcript_timestamps.txt").write_text(
        "".join(f"[{_ts(s['start'], '.')[:-4]}] {s['text']}\n" for s in segs), encoding="utf-8"
    )
    (d / "transcript.srt").write_text(
        "".join(f"{i}\n{_ts(s['start'])} --> {_ts(s['end'])}\n{s['text']}\n\n" for i, s in enumerate(segs, 1)),
        encoding="utf-8",
    )
    (d / "transcript.vtt").write_text(
        "WEBVTT\n\n" + "".join(f"{_ts(s['start'], '.')} --> {_ts(s['end'], '.')}\n{s['text']}\n\n" for s in segs),
        encoding="utf-8",
    )
    (d / "transcript.json").write_text(json.dumps({**meta, "segments": segs}, ensure_ascii=False, indent=1), encoding="utf-8")


# --------------------------------------------------------------------------- 워커

def _process(job_id: str) -> None:
    with _lock:
        job = dict(_jobs[job_id])
    d = _job_dir(job_id)
    source = d / job["source"]
    parts_dir = d / "parts"
    results_dir = d / "results"
    results_dir.mkdir(exist_ok=True)

    # 1) 분할 (이미 끝났으면 건너뜀 → 재시작 시 이어하기)
    duration = job.get("duration") or _probe_duration(source)
    if not job.get("split_done") or not parts_dir.exists():
        _update(job_id, status="preparing", stage="오디오 변환 중", duration=duration)
        shutil.rmtree(results_dir, ignore_errors=True)
        results_dir.mkdir()
        _split(job_id, source, parts_dir, duration)
        _update(job_id, split_done=True)

    parts = sorted(parts_dir.glob("part_*.wav"))
    if not parts:
        raise RuntimeError("오디오 스트림을 찾지 못했습니다. 음성 파일이 맞는지 확인하세요.")
    lengths = [_wav_seconds(p) for p in parts]
    total = sum(lengths) or 1.0
    _update(job_id, duration=total, parts_total=len(parts))

    # 2) 모델 로딩
    _update(job_id, status="transcribing", stage=f"모델 로딩 중 ({job['model']})")
    model = get_model(job["model"])

    with _lock:
        _live[job_id] = _load_segments(job_id)
    offset = 0.0
    done_secs = sum(l for p, l in zip(parts, lengths) if (results_dir / (p.stem + ".json")).exists())
    _update(
        job_id,
        progress=done_secs / total,
        progress_at_start=done_secs / total,
        transcribe_started=time.time(),
    )
    language = None if job["language"] in ("", "auto") else job["language"]
    prev_tail = None

    # 3) 조각별 변환
    for idx, (part, length) in enumerate(zip(parts, lengths)):
        result_file = results_dir / (part.stem + ".json")
        if result_file.exists():
            saved = json.loads(result_file.read_text())
            if saved["segments"]:
                prev_tail = saved["segments"][-1]["text"]
            offset += length
            continue
        if job_id in _cancel:
            raise Cancelled()
        _update(job_id, stage=f"음성 인식 중 (조각 {idx + 1}/{len(parts)})", parts_done=idx)

        seg_iter, info = model.transcribe(
            str(part),
            language=language,
            beam_size=config.BEAM_SIZE,
            vad_filter=True,  # 무음 구간 건너뛰기 (통화 녹음의 대기음/정적에서 환각 방지)
            vad_parameters={"min_silence_duration_ms": 500},
            condition_on_previous_text=False,  # 긴 오디오에서 같은 문장 반복 루프 방지
            initial_prompt=prev_tail,  # 앞 조각의 마지막 문장으로 문맥 연결
        )
        if language is None and not job.get("detected_language"):
            _update(job_id, detected_language=info.language)
            language = info.language
        part_segs = []
        last_update = 0.0
        for s in seg_iter:
            if job_id in _cancel:
                raise Cancelled()
            text = s.text.strip()
            if not text:
                continue
            seg = {"start": round(offset + s.start, 3), "end": round(offset + s.end, 3), "text": text}
            part_segs.append(seg)
            with _lock:
                _live[job_id].append(seg)
            if time.time() - last_update > 1:
                last_update = time.time()
                _update(job_id, progress=min((offset + s.end) / total, 0.999))
        tmp = result_file.with_suffix(".tmp")
        tmp.write_text(json.dumps({"segments": part_segs}, ensure_ascii=False))
        os.replace(tmp, result_file)
        if part_segs:
            prev_tail = part_segs[-1]["text"]
        offset += length
        _update(job_id, progress=min(offset / total, 0.999), parts_done=idx + 1)

    # 4) 결과 생성 및 정리
    segs = _load_segments(job_id)
    with _lock:
        _live[job_id] = segs
    _write_outputs(
        job_id,
        segs,
        {"filename": job["filename"], "model": job["model"], "language": language or job["language"], "duration": total},
    )
    shutil.rmtree(parts_dir, ignore_errors=True)
    if config.DELETE_SOURCE_AFTER:
        source.unlink(missing_ok=True)
    _update(job_id, status="done", stage="완료", progress=1.0, finished=time.time(), segment_count=len(segs))


def _worker() -> None:
    while True:
        job_id = _queue.get()
        with _lock:
            job = _jobs.get(job_id)
            if job is None or job["status"] not in ACTIVE:
                _cancel.discard(job_id)
                continue
            if job_id in _cancel:
                _cancel.discard(job_id)
                _update(job_id, status="cancelled", stage="취소됨")
                continue
        try:
            _process(job_id)
        except Cancelled:
            log.info("작업 취소: %s", job_id)
            with _lock:
                if job_id in _jobs:
                    _update(job_id, status="cancelled", stage="취소됨", finished=time.time())
        except Exception as e:  # noqa: BLE001
            log.exception("작업 실패: %s", job_id)
            with _lock:
                if job_id in _jobs:
                    _update(job_id, status="error", stage="실패", error=str(e), finished=time.time())
        finally:
            _cancel.discard(job_id)


def start_worker() -> None:
    threading.Thread(target=_worker, name="whisper-worker", daemon=True).start()

"""Whisper 음성 → 텍스트 변환 웹 서버 (로컬망 전용)."""
import logging
import shutil
from pathlib import Path
from urllib.parse import quote

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import config, hardware, jobs, uploads

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

app = FastAPI(title="Whisper STT")
STATIC = Path(__file__).parent / "static"


@app.on_event("startup")
def _startup() -> None:
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise RuntimeError("ffmpeg/ffprobe 가 설치되어 있어야 합니다.")
    uploads.cleanup_stale()
    hw, rec = hardware.detect(), hardware.recommend()
    logging.info(
        "사양: %s · %d코어 · RAM %.1fGB · GPU %s → 모델=%s device=%s compute=%s beam=%d batch=%d threads=%d",
        hw["cpu"], hw["cpu_cores"], hw["ram_bytes"] / hardware.GB, ", ".join(g["name"] for g in hw["gpus"]) or "없음",
        config.MODEL, config.DEVICE, config.COMPUTE_TYPE, config.BEAM_SIZE, config.BATCH_SIZE, config.CPU_THREADS,
    )
    jobs.load_existing()
    jobs.start_worker()


def _not_found(e: KeyError):
    raise HTTPException(404, "찾을 수 없습니다.") from e


# --------------------------------------------------------------------------- 설정

@app.get("/api/config")
def get_config():
    return {
        "models": jobs.MODELS,
        "default_model": config.MODEL,
        "default_language": config.DEFAULT_LANGUAGE,
        "chunk_size": config.CHUNK_SIZE,
        "max_upload_bytes": config.MAX_UPLOAD_BYTES,
        "free_disk_bytes": shutil.disk_usage(config.DATA_DIR).free,
        **jobs.device_info(),
        "hardware": hardware.detect(),
        "recommended": hardware.recommend(),
        # 이 컴퓨터에서 실측한 모델별 처리 속도 (오디오 1초당 처리 초)
        "measured_rtf": {m: jobs.model_rtf(m) for m in jobs.MODELS if jobs.model_rtf(m)},
    }


# --------------------------------------------------------------------------- 업로드

class UploadInit(BaseModel):
    filename: str
    size: int


@app.post("/api/uploads")
def create_upload(body: UploadInit):
    try:
        return uploads.create(body.filename, body.size)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


@app.get("/api/uploads/{upload_id}")
def upload_status(upload_id: str):
    try:
        return uploads.status(upload_id)
    except KeyError as e:
        _not_found(e)


@app.put("/api/uploads/{upload_id}/chunks/{index}")
async def upload_chunk(upload_id: str, index: int, request: Request):
    try:
        return await uploads.write_chunk(upload_id, index, request.stream())
    except KeyError as e:
        _not_found(e)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


@app.delete("/api/uploads/{upload_id}")
def delete_upload(upload_id: str):
    try:
        uploads.delete(upload_id)
    except KeyError as e:
        _not_found(e)
    return {"ok": True}


class CompleteBody(BaseModel):
    language: str = config.DEFAULT_LANGUAGE
    model: str = config.MODEL


@app.post("/api/uploads/{upload_id}/complete")
def complete_upload(upload_id: str, body: CompleteBody):
    if body.model not in jobs.MODELS and body.model != config.MODEL:
        raise HTTPException(400, "지원하지 않는 모델입니다.")
    try:
        meta = uploads.status(upload_id)
        return jobs.create(
            meta["filename"],
            lambda dest: uploads.finalize(upload_id, dest),
            language=body.language.strip() or "auto",
            model=body.model,
        )
    except KeyError as e:
        _not_found(e)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


# --------------------------------------------------------------------------- 작업

@app.get("/api/jobs")
def list_jobs():
    return jobs.list_jobs()


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    try:
        return jobs.get(job_id)
    except KeyError as e:
        _not_found(e)


@app.get("/api/jobs/{job_id}/segments")
def get_segments(job_id: str, offset: int = 0):
    try:
        return {"offset": offset, "segments": jobs.segments(job_id, max(offset, 0))}
    except KeyError as e:
        _not_found(e)


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    try:
        return jobs.cancel(job_id)
    except KeyError as e:
        _not_found(e)


@app.post("/api/jobs/{job_id}/retry")
def retry_job(job_id: str):
    try:
        return jobs.retry(job_id)
    except KeyError as e:
        _not_found(e)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str):
    try:
        jobs.delete(job_id)
    except KeyError as e:
        _not_found(e)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    return {"ok": True}


@app.get("/api/jobs/{job_id}/download/{fmt}")
def download(job_id: str, fmt: str):
    try:
        job = jobs.get(job_id)
        path = jobs.output_path(job_id, fmt)
    except KeyError as e:
        _not_found(e)
    if not path.exists():
        raise HTTPException(404, "결과 파일이 아직 없습니다.")
    stem = Path(job["filename"]).stem or "transcript"
    name = f"{stem}{'_timestamps.txt' if fmt == 'ts.txt' else '.' + fmt}"
    return FileResponse(
        path,
        media_type="text/plain; charset=utf-8" if fmt != "json" else "application/json",
        headers={"Content-Disposition": f"attachment; filename*=UTF-8''{quote(name)}"},
    )


app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")

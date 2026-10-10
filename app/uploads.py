"""이어받기(resume) 가능한 청크 업로드.

브라우저가 파일을 CHUNK_SIZE 단위로 잘라 여러 번 PUT 하면, 서버는 미리 만들어 둔
파일의 해당 오프셋에 바로 기록한다. 파일 전체를 메모리에 올리지 않으므로 수 GB
파일도 문제없고, 네트워크가 끊기면 받지 못한 청크만 다시 보내면 된다.
"""
import json
import os
import re
import shutil
import threading
import time
import uuid
from pathlib import Path

from . import config

_lock = threading.Lock()


def _safe_name(name: str) -> str:
    name = os.path.basename(name.replace("\\", "/"))
    name = re.sub(r"[^\w.\-가-힣() ]+", "_", name).strip(" .")
    return name[:200] or "audio"


def _dir(upload_id: str) -> Path:
    if not re.fullmatch(r"[0-9a-f]{32}", upload_id):
        raise KeyError(upload_id)
    return config.UPLOAD_DIR / upload_id


def _meta_path(upload_id: str) -> Path:
    return _dir(upload_id) / "meta.json"


def _load(upload_id: str) -> dict:
    p = _meta_path(upload_id)
    if not p.exists():
        raise KeyError(upload_id)
    return json.loads(p.read_text())


def _save(meta: dict) -> None:
    p = _meta_path(meta["id"])
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(meta))
    os.replace(tmp, p)


def create(filename: str, size: int) -> dict:
    if size <= 0:
        raise ValueError("빈 파일입니다.")
    if config.MAX_UPLOAD_BYTES and size > config.MAX_UPLOAD_BYTES:
        raise ValueError("파일이 허용된 최대 크기를 초과합니다.")
    free = shutil.disk_usage(config.DATA_DIR).free
    if size * 2 > free:  # 원본 + 변환용 wav 여유
        raise ValueError(f"디스크 공간이 부족합니다 (남은 공간 {free / 1e9:.1f} GB).")

    upload_id = uuid.uuid4().hex
    d = config.UPLOAD_DIR / upload_id
    d.mkdir(parents=True)
    filename = _safe_name(filename)
    with open(d / "data", "wb") as f:
        f.truncate(size)  # sparse 파일로 미리 크기 확보
    total = (size + config.CHUNK_SIZE - 1) // config.CHUNK_SIZE
    meta = {
        "id": upload_id,
        "filename": filename,
        "size": size,
        "chunk_size": config.CHUNK_SIZE,
        "total_chunks": total,
        "received": [],
        "created": time.time(),
    }
    _save(meta)
    return meta


def status(upload_id: str) -> dict:
    return _load(upload_id)


async def write_chunk(upload_id: str, index: int, stream) -> dict:
    meta = _load(upload_id)
    if not 0 <= index < meta["total_chunks"]:
        raise ValueError("잘못된 청크 번호입니다.")
    offset = index * meta["chunk_size"]
    expected = min(meta["chunk_size"], meta["size"] - offset)

    written = 0
    fd = os.open(_dir(upload_id) / "data", os.O_WRONLY)
    try:
        async for piece in stream:
            if written + len(piece) > expected:
                raise ValueError("청크 크기가 예상보다 큽니다.")
            os.pwrite(fd, piece, offset + written)
            written += len(piece)
        if written != expected:
            raise ValueError(f"청크가 불완전합니다 ({written}/{expected} bytes).")
    except BaseException:
        # 일부만 기록된 청크는 '받지 않음' 으로 되돌려 다시 받도록 한다
        with _lock:
            meta = _load(upload_id)
            if index in meta["received"]:
                meta["received"].remove(index)
                _save(meta)
        raise
    finally:
        os.close(fd)

    with _lock:
        meta = _load(upload_id)
        if index not in meta["received"]:
            meta["received"].append(index)
            meta["received"].sort()
            _save(meta)
    return {"index": index, "received": len(meta["received"]), "total": meta["total_chunks"]}


def finalize(upload_id: str, dest: Path) -> dict:
    """업로드가 끝났는지 확인하고 데이터 파일을 dest 로 옮긴다."""
    meta = _load(upload_id)
    if len(meta["received"]) != meta["total_chunks"]:
        missing = meta["total_chunks"] - len(meta["received"])
        raise ValueError(f"아직 {missing}개 청크가 업로드되지 않았습니다.")
    os.replace(_dir(upload_id) / "data", dest)
    shutil.rmtree(_dir(upload_id), ignore_errors=True)
    return meta


def delete(upload_id: str) -> None:
    shutil.rmtree(_dir(upload_id), ignore_errors=True)


def cleanup_stale(max_age_seconds: float = 7 * 24 * 3600) -> None:
    now = time.time()
    for d in config.UPLOAD_DIR.iterdir():
        try:
            if now - d.stat().st_mtime > max_age_seconds:
                shutil.rmtree(d, ignore_errors=True)
        except OSError:
            pass

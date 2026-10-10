"""실행 중인 컴퓨터 사양을 파악해 Whisper 설정(모델/정밀도/스레드/배치)을 자동으로 고른다.

`python3 -m app.hardware` 로 실행하면 사양과 추천 설정을 출력한다.
"""
import os
import platform
import shutil
import subprocess
from functools import lru_cache


def _cpu_name() -> str:
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    if platform.system() == "Darwin":
        try:
            return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
        except OSError:
            pass
    return platform.processor() or platform.machine()


def _cpu_cores() -> int:
    """컨테이너/cgroup 제한까지 고려한 사용 가능 코어 수."""
    n = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    try:
        quota, period = open("/sys/fs/cgroup/cpu.max").read().split()
        if quota != "max":
            n = min(n, max(1, int(int(quota) / int(period))))
    except (OSError, ValueError):
        pass
    return n


def _cpu_flags() -> set[str]:
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("flags"):
                    return set(line.split(":", 1)[1].split()) & {"avx", "avx2", "avx512f", "fma", "avx_vnni", "avx512_vnni"}
    except OSError:
        pass
    return set()


def _ram_bytes() -> int:
    total = 0
    try:
        total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        pass
    try:
        limit = open("/sys/fs/cgroup/memory.max").read().strip()
        if limit != "max":
            total = min(total, int(limit)) if total else int(limit)
    except (OSError, ValueError):
        pass
    return total


def _gpus() -> list[dict]:
    try:
        from . import cuda_libs

        cuda_libs.load()  # pip 로 설치한 CUDA 라이브러리를 경로 설정 없이 사용
        import ctranslate2

        if ctranslate2.get_cuda_device_count() == 0:
            return []
    except Exception:
        return []
    gpus = []
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10,
            ).stdout
            for line in out.strip().splitlines():
                name, mem = [x.strip() for x in line.rsplit(",", 1)]
                gpus.append({"name": name, "vram_bytes": int(float(mem)) * 1024 * 1024})
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
    return gpus or [{"name": "CUDA GPU", "vram_bytes": 0}]


@lru_cache(maxsize=1)
def detect() -> dict:
    gpus = _gpus()
    return {
        "os": f"{platform.system()} {platform.release()}",
        "arch": platform.machine(),
        "cpu": _cpu_name(),
        "cpu_cores": _cpu_cores(),
        "cpu_flags": sorted(_cpu_flags()),
        "ram_bytes": _ram_bytes(),
        "gpus": gpus,
    }


GB = 1024**3

# 모델별 대략적인 메모리 요구량 (int8 / float16 기준, 여유 포함)
MODEL_MEM = {"tiny": 0.3, "base": 0.4, "small": 0.8, "medium": 2.0, "turbo": 2.0, "large-v2": 3.5, "large-v3": 3.5}


@lru_cache(maxsize=1)
def recommend() -> dict:
    """사양에 맞는 기본 설정과 모델별 사용 가능 여부/안내 문구."""
    hw = detect()
    ram = hw["ram_bytes"] / GB if hw["ram_bytes"] else 8
    cores = hw["cpu_cores"]
    notes: dict[str, str] = {}

    if hw["gpus"]:
        vram = max(g["vram_bytes"] for g in hw["gpus"]) / GB
        device = "cuda"
        if vram == 0 or vram >= 10:
            model, compute, batch = "large-v3", "float16", 16
        elif vram >= 6:
            model, compute, batch = "large-v3", "int8_float16", 8
        elif vram >= 4:
            model, compute, batch = "turbo", "int8_float16", 4
        else:
            model, compute, batch = "small", "int8_float16", 4
        beam = 5
        for m, need in MODEL_MEM.items():
            if vram and need > vram:
                notes[m] = "GPU 메모리 부족 가능"
        reason = f"GPU {hw['gpus'][0]['name']} ({vram:.0f}GB) 사용"
    else:
        device, compute = "cpu", "int8"
        # CPU 에서는 인코더가 큰 large 계열이 매우 느리다. 코어 수 기준으로 현실적인 모델을 고른다.
        if cores >= 12 and ram >= 16:
            model = "turbo"
        elif cores >= 4 and ram >= 8:
            model = "small"
        else:
            model = "base"
        beam = 1  # CPU 에서는 greedy 디코딩이 2배 가까이 빠르고 정확도 차이는 작다
        batch = 8 if ram >= 8 and cores >= 4 else 4
        slow = {"medium": "CPU 에서 느림", "turbo": "CPU 에서 느림", "large-v2": "CPU 에서 매우 느림", "large-v3": "CPU 에서 매우 느림"}
        if cores >= 12:
            slow.pop("turbo")
        notes.update(slow)
        reason = f"GPU 없음 → CPU {cores}코어 int8 연산"

    for m, need in MODEL_MEM.items():
        if need * 1.5 > ram:
            notes[m] = "RAM 부족 가능"

    return {
        "device": device,
        "compute_type": compute,
        "model": model,
        "beam_size": beam,
        "batch_size": batch,
        "cpu_threads": cores if device == "cpu" else min(cores, 4),
        "reason": reason,
        "model_notes": notes,
    }


if __name__ == "__main__":
    hw, rec = detect(), recommend()
    print(f"OS      : {hw['os']} ({hw['arch']})")
    print(f"CPU     : {hw['cpu']} · {hw['cpu_cores']}코어 · {' '.join(hw['cpu_flags']) or '-'}")
    print(f"RAM     : {hw['ram_bytes'] / GB:.1f} GB")
    gpu_desc = ", ".join("%s (%.0fGB)" % (g["name"], g["vram_bytes"] / GB) for g in hw["gpus"])
    print(f"GPU     : {gpu_desc or '없음'}")
    print()
    print(f"추천 설정: model={rec['model']} device={rec['device']} compute_type={rec['compute_type']} "
          f"beam_size={rec['beam_size']} batch_size={rec['batch_size']} cpu_threads={rec['cpu_threads']}")
    print(f"이유     : {rec['reason']}")
    for m, n in rec["model_notes"].items():
        print(f"  - {m}: {n}")

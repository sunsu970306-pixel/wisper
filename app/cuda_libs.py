"""pip 로 설치한 NVIDIA CUDA 라이브러리(nvidia-cublas-cu12, nvidia-cudnn-cu12)를 자동으로 찾아 로드한다.

이렇게 하면 LD_LIBRARY_PATH / PATH 를 직접 설정하지 않아도 GPU 를 쓸 수 있다.
ctranslate2 를 import 하기 전에 load() 를 호출해야 한다.
"""
import ctypes
import glob
import os
import sys

_done = False


def _nvidia_dirs(sub: str) -> list[str]:
    try:
        import nvidia  # namespace package
    except ImportError:
        return []
    dirs = []
    for base in list(getattr(nvidia, "__path__", [])):
        for pkg in ("cublas", "cudnn", "cuda_runtime", "cuda_nvrtc"):
            d = os.path.join(base, pkg, sub)
            if os.path.isdir(d):
                dirs.append(d)
    return dirs


def load() -> None:
    global _done
    if _done:
        return
    _done = True
    if sys.platform == "win32":
        for d in _nvidia_dirs("bin"):
            os.add_dll_directory(d)
            os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")
        return
    if sys.platform != "linux":
        return
    libs = []
    for d in _nvidia_dirs("lib"):
        libs += sorted(glob.glob(os.path.join(d, "lib*.so*")))
    # 서로 의존하므로 더 이상 로드되는 것이 없을 때까지 반복 (이미 로드된 soname 은 dlopen 이 재사용)
    pending = libs
    while pending:
        failed = []
        for lib in pending:
            try:
                ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                failed.append(lib)
        if len(failed) == len(pending):
            break
        pending = failed

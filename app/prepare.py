"""서버 실행 전 점검: 추천 모델을 미리 다운로드/로드하고 짧은 음성으로 실제 동작을 확인한다.

`python -m app.prepare` 로 실행. GPU 로드에 실패하면 원인을 보여주고 CPU 로 대신 동작하는지 확인한다.
"""
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from . import config


def main() -> int:
    for tool in ("ffmpeg", "ffprobe"):
        if not shutil.which(tool):
            print(f"[오류] {tool} 를 찾을 수 없습니다. ffmpeg 를 설치하세요.")
            return 1

    from . import jobs

    print(f"모델: {config.MODEL} · {config.DEVICE} · {config.COMPUTE_TYPE} (처음 실행이면 다운로드합니다)")
    t = time.time()
    try:
        model = jobs.get_model(config.MODEL)
    except Exception as e:  # noqa: BLE001
        print(f"[오류] 모델 로딩 실패: {e}")
        print("  → 처음 실행이면 인터넷(huggingface.co) 연결이 필요합니다. 오프라인이면 README 의 '모델 미리 받기' 참고.")
        if config.DEVICE == "cuda":
            print("  → GPU 라이브러리(CUDA 12 / cuDNN 9) 문제일 수 있습니다. CPU 로 실행하려면 WHISPER_DEVICE=cpu 로 설정하세요.")
        return 1
    print(f"  모델 로딩 완료 ({time.time() - t:.1f}초)")

    # 3초짜리 무음으로 추론 파이프라인 동작 확인 (GPU 연산 라이브러리까지 실제로 로드됨)
    with tempfile.TemporaryDirectory() as d:
        wav = Path(d) / "check.wav"
        subprocess.run(
            ["ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono", "-t", "3", str(wav)],
            check=True,
        )
        try:
            # VAD 를 끄고 직접 돌려야 무음이어도 인코더/디코더 연산이 실제로 실행된다
            base = getattr(model, "model", model)
            segs, _ = base.transcribe(str(wav), language="ko", beam_size=1, vad_filter=False)
            list(segs)
        except Exception as e:  # noqa: BLE001
            print(f"[오류] 추론 테스트 실패: {e}")
            if config.DEVICE == "cuda":
                print("  → cuDNN/cuBLAS 를 찾지 못했을 수 있습니다: pip install nvidia-cublas-cu12 \"nvidia-cudnn-cu12==9.*\"")
            return 1
    print("  추론 테스트 통과")
    return 0


if __name__ == "__main__":
    sys.exit(main())

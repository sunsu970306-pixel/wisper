# Whisper 음성 → 텍스트 변환 웹 (로컬망용)

로컬 PC/서버에서 [faster-whisper](https://github.com/SYSTRAN/faster-whisper)(Whisper 모델의 고속 구현)로
음성·통화 녹음 파일을 텍스트(txt / srt / vtt / json)로 변환하는 웹사이트입니다.
같은 네트워크의 다른 PC·휴대폰 브라우저에서 `http://<서버IP>:8000` 으로 접속해 사용합니다.

## 대용량 통화 파일 대응

| 문제 | 처리 방식 |
|---|---|
| 업로드가 수 GB | 브라우저가 16MB 조각으로 나눠 3개씩 병렬 전송. 서버는 메모리에 올리지 않고 디스크에 바로 기록 |
| 업로드 중 끊김 | 조각별 자동 재시도(지수 백오프). 페이지를 닫았다가 **같은 파일을 다시 선택하면 받은 부분 이후부터 이어서** 업로드 |
| 변환 시 메모리 | ffmpeg 로 16kHz mono wav 를 20분 단위 조각으로 나눠 순차 처리 → 10시간짜리 파일도 메모리 사용량 일정 |
| 변환이 오래 걸림 | 백그라운드 큐에서 처리, 진행률·남은 시간 표시, 인식된 문장 실시간 미리보기 |
| 서버 재시작/실패 | 조각별 결과를 디스크에 저장 → 재시작 시 자동으로, 실패 시 "다시 시도"로 **끝난 조각 이후부터 이어서** 변환 |
| 통화 녹음 특성 | VAD(무음 구간 제거)로 대기음·정적 구간 환각 감소, 반복 루프 방지 옵션 적용 |

지원 포맷: ffmpeg 가 읽을 수 있는 모든 포맷 (mp3, m4a, wav, amr, 3gp, ogg, flac, aac, mp4 영상 등)

## 설치

필요: Python 3.10+, ffmpeg

```bash
# ffmpeg
sudo apt install ffmpeg          # Ubuntu/Debian
# brew install ffmpeg            # macOS
# winget install ffmpeg          # Windows

python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**NVIDIA GPU 사용 시** (강력 추천 — CPU 대비 수십 배 빠름): CUDA 12 + cuDNN 9 가 필요합니다.
```bash
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12==9.*
export LD_LIBRARY_PATH=$(python3 -c 'import os, nvidia.cublas.lib, nvidia.cudnn.lib; print(os.path.dirname(nvidia.cublas.lib.__file__) + ":" + os.path.dirname(nvidia.cudnn.lib.__file__))')
```
GPU 가 있으면 자동으로 사용하며, 화면 상단에 `장치: CUDA` 로 표시됩니다.

## 실행

```bash
./run.sh
# 또는: python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```
- 이 PC: http://localhost:8000
- 같은 망의 다른 기기: http://<이 PC의 IP>:8000 (방화벽에서 8000 포트 허용 필요)

첫 변환 시 선택한 모델이 Hugging Face 에서 자동 다운로드됩니다(large-v3 약 3GB).
인터넷이 없는 망이라면 다른 PC에서 모델 폴더(예: `Systran/faster-whisper-large-v3`)를 받아 두고
`WHISPER_MODEL=/경로/faster-whisper-large-v3` 로 지정하면 됩니다.

## 설정 (환경변수)

| 변수 | 기본값 | 설명 |
|---|---|---|
| `WHISPER_MODEL` | `large-v3` | 기본 모델. `tiny`/`base`/`small`/`medium`/`large-v3`/`turbo` 또는 로컬 경로 |
| `WHISPER_DEVICE` | `auto` | `auto` / `cuda` / `cpu` |
| `WHISPER_COMPUTE_TYPE` | `default` | GPU=`float16`, CPU=`int8` 자동. VRAM 부족 시 `int8_float16` |
| `WHISPER_LANGUAGE` | `ko` | 기본 언어 (`auto` = 자동 감지) |
| `WHISPER_BEAM_SIZE` | `5` | 낮추면 빠르고 정확도 약간 하락 |
| `WHISPER_SEGMENT_SECONDS` | `1200` | 분할 처리 단위(초) |
| `WHISPER_CHUNK_SIZE` | `16777216` | 업로드 조각 크기(바이트) |
| `WHISPER_MAX_UPLOAD_BYTES` | `0` | 최대 업로드 크기(0=무제한) |
| `WHISPER_DATA_DIR` | `./data` | 업로드·작업·결과 저장 위치 (디스크 여유 공간 충분한 곳으로) |
| `WHISPER_MODEL_DIR` | (HF 캐시) | 모델 다운로드 위치 |
| `WHISPER_CPU_THREADS` | `0` | CPU 스레드 수 (0=자동) |
| `WHISPER_DELETE_SOURCE` | `false` | 변환 완료 후 원본 음성 삭제 |

### 모델 선택 가이드
- **GPU 있음**: `large-v3` (가장 정확) 또는 `turbo` (거의 비슷한 정확도로 훨씬 빠름)
- **CPU만**: `small` ~ `medium` 권장. CPU 에서 `large-v3` 는 1시간 녹음에 수 시간 걸릴 수 있습니다.

## 디스크 사용량
- 업로드 원본 + 변환용 wav(1시간당 약 115MB, 완료 후 자동 삭제) + 결과 텍스트
- 작업 목록의 "삭제" 버튼으로 원본·결과를 지울 수 있습니다.

## 구조
```
app/
  main.py        FastAPI 라우트 (업로드 / 작업 / 다운로드)
  uploads.py     이어받기 가능한 청크 업로드
  jobs.py        작업 큐, ffmpeg 분할, faster-whisper 변환, 결과 파일 생성
  config.py      환경변수 설정
  static/index.html  웹 UI
data/            업로드·작업 데이터 (자동 생성)
```

## 참고
- 인증이 없으므로 **로컬망에서만** 사용하세요. 외부에 노출하려면 리버스 프록시(nginx 등)에 인증을 붙이세요.
  nginx 를 앞에 둘 경우 `client_max_body_size 32m;` 이상으로 설정하세요(조각 크기보다 크게).
- 변환은 한 번에 하나씩 순서대로 처리됩니다(GPU 메모리 보호). 여러 파일을 올리면 대기열에 쌓입니다.

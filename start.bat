@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
title Whisper 음성 변환 서버

echo ================================================
echo  Whisper 음성 변환 - 자동 설치 및 실행 (Windows)
echo ================================================
echo.

rem ---------------------------------------------- Python
set "PY="
where py >nul 2>nul && set "PY=py -3"
if not defined PY (
  where python >nul 2>nul && set "PY=python"
)
if not defined PY (
  echo [설치] Python 이 없어 winget 으로 설치합니다...
  winget install -e --id Python.Python.3.12 --accept-package-agreements --accept-source-agreements
  echo.
  echo Python 설치가 끝났습니다. 이 창을 닫고 start.bat 을 다시 실행해 주세요.
  pause
  exit /b 1
)

rem ---------------------------------------------- ffmpeg
where ffmpeg >nul 2>nul
if errorlevel 1 (
  if exist "%~dp0tools\ffmpeg\bin\ffmpeg.exe" (
    set "PATH=%~dp0tools\ffmpeg\bin;%PATH%"
  ) else (
    echo [설치] ffmpeg 가 없어 winget 으로 설치합니다...
    winget install -e --id Gyan.FFmpeg --accept-package-agreements --accept-source-agreements
    echo.
    echo ffmpeg 설치가 끝났습니다. 이 창을 닫고 start.bat 을 다시 실행해 주세요.
    pause
    exit /b 1
  )
)

rem ---------------------------------------------- 가상환경 + 패키지
if not exist ".venv\Scripts\python.exe" (
  echo [설치] 가상환경 생성 중...
  %PY% -m venv .venv || goto :fail
)
set "VPY=.venv\Scripts\python.exe"
if not exist ".venv\.installed" (
  echo [설치] Python 패키지 설치 중... 처음 한 번만, 몇 분 걸립니다
  "%VPY%" -m pip install --upgrade pip >nul
  "%VPY%" -m pip install -r requirements.txt || goto :fail
  where nvidia-smi >nul 2>nul
  if not errorlevel 1 (
    echo [설치] NVIDIA GPU 감지 - CUDA 라이브러리 설치 중... 약 1GB
    "%VPY%" -m pip install nvidia-cublas-cu12 "nvidia-cudnn-cu12==9.*" || goto :fail
  )
  echo ok> ".venv\.installed"
)

rem ---------------------------------------------- 사양 확인 + 모델 미리 받기
echo.
"%VPY%" -m app.hardware
echo.
echo [준비] 추천 모델 다운로드/로딩 확인 중... 처음에는 수 GB 다운로드
"%VPY%" -m app.prepare || goto :fail

rem ---------------------------------------------- 실행
echo.
echo ================================================
echo  서버 실행 중 - 이 창을 닫으면 서버가 꺼집니다
echo  이 PC:        http://localhost:8000
for /f "tokens=2 delims=:" %%a in ('ipconfig ^| findstr /c:"IPv4"') do for /f "tokens=*" %%b in ("%%a") do echo  같은 망 기기: http://%%b:8000
echo ================================================
start "" http://localhost:8000
"%VPY%" -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --timeout-keep-alive 120
pause
exit /b 0

:fail
echo.
echo [오류] 설치/실행 중 문제가 발생했습니다. 위 메시지를 확인해 주세요.
pause
exit /b 1

@echo off
setlocal EnableDelayedExpansion
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  set "PYTHON_CMD="
  where py >nul 2>nul
  if not errorlevel 1 (
    py -3.12 -c "import sys; assert sys.version_info >= (3, 12)" >nul 2>nul
    if not errorlevel 1 set "PYTHON_CMD=py -3.12"
  )
  if not defined PYTHON_CMD (
    python -c "import sys; assert sys.version_info >= (3, 12)" >nul 2>nul
    if errorlevel 1 (
      echo Python 3.12 o superiore e' richiesto.
      exit /b 1
    )
    set "PYTHON_CMD=python"
  )
  !PYTHON_CMD! -m venv .venv
  if errorlevel 1 exit /b 1
  ".venv\Scripts\python.exe" -m pip install --upgrade pip
  if errorlevel 1 exit /b 1
  ".venv\Scripts\python.exe" -m pip install -r requirements.txt
  if errorlevel 1 exit /b 1
)
".venv\Scripts\python.exe" -c "import sys; assert sys.version_info >= (3, 12)" >nul 2>nul
if errorlevel 1 (
  echo La virtualenv esistente non usa Python 3.12 o superiore.
  exit /b 1
)
".venv\Scripts\python.exe" -c "import cv2, mediapipe, numpy, pycaw, comtypes" >nul 2>nul
if errorlevel 1 (
  ".venv\Scripts\python.exe" -m pip install -r requirements.txt
  if errorlevel 1 exit /b 1
)
".venv\Scripts\python.exe" "test.py"

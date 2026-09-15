@echo off
setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" goto :validate_venv

where py >nul 2>nul
if errorlevel 1 goto :try_python
call py -3.12 -c "import platform, struct, sys; raise SystemExit(0 if platform.python_implementation() == 'CPython' and sys.version_info[:2] == (3, 12) and struct.calcsize('P') * 8 == 64 else 1)" >nul 2>nul
if errorlevel 1 goto :try_python
set "PYTHON_CMD=py -3.12"
goto :create_venv

:try_python
where python >nul 2>nul
if errorlevel 1 goto :unsupported_python
call python -c "import platform, struct, sys; raise SystemExit(0 if platform.python_implementation() == 'CPython' and sys.version_info[:2] == (3, 12) and struct.calcsize('P') * 8 == 64 else 1)" >nul 2>nul
if errorlevel 1 goto :unsupported_python
set "PYTHON_CMD=python"

:create_venv
if /i "%PYTHON_CMD%"=="py -3.12" goto :create_with_py
call python -m venv ".venv"
goto :created_venv

:create_with_py
call py -3.12 -m venv ".venv"

:created_venv
if errorlevel 1 goto :venv_failed
if not exist ".venv\Scripts\python.exe" goto :venv_missing_python

:validate_venv
".venv\Scripts\python.exe" -c "import platform, struct, sys; raise SystemExit(0 if platform.python_implementation() == 'CPython' and sys.version_info[:2] == (3, 12) and struct.calcsize('P') * 8 == 64 else 1)" >nul 2>nul
if errorlevel 1 goto :invalid_venv
if not exist "requirements.lock" goto :missing_lock

".venv\Scripts\python.exe" -m pip install --require-hashes --only-binary=:all: -r "requirements.lock"
if errorlevel 1 goto :dependency_failed

".venv\Scripts\python.exe" "main.py"
exit /b %errorlevel%

:unsupported_python
echo CPython 3.12 x64 richiesto per creare l'ambiente.
exit /b 1

:venv_failed
echo Creazione della virtualenv fallita.
exit /b 1

:venv_missing_python
echo Virtualenv creata senza interprete Python.
exit /b 1

:invalid_venv
echo La virtualenv deve usare CPython 3.12 x64.
exit /b 1

:missing_lock
echo requirements.lock mancante.
exit /b 1

:dependency_failed
echo Installazione delle dipendenze bloccate fallita.
exit /b 1

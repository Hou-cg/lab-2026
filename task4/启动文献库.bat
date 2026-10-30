@echo off
setlocal EnableExtensions
chcp 65001 >nul
set "PYTHONUTF8=1"
cd /d "%~dp0"
title Wenxian Library Launcher

rem Find a usable Python 3.11+ from the active conda env, common install paths,
rem conda envs, or PATH. No hard-coded lab path is required.
set "PY="

call :try_python "%CONDA_PREFIX%\python.exe"
call :try_python "%USERPROFILE%\miniconda3\python.exe"
call :try_python "%USERPROFILE%\anaconda3\python.exe"
call :try_python "%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
call :try_python "%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
call :try_python "C:\Python311\python.exe"
call :try_python "C:\Python312\python.exe"
call :try_python "D:\Python311\python.exe"
call :try_python "D:\Python312\python.exe"
call :try_python "C:\ProgramData\miniconda3\python.exe"
call :try_python "C:\ProgramData\anaconda3\python.exe"
call :try_python "D:\Miniconda3\python.exe"
call :try_python "D:\Miniconda3\envs\lab\python.exe"

rem Search all environments under common conda installations.
for /d %%E in ("%USERPROFILE%\miniconda3\envs\*") do call :try_python "%%E\python.exe"
for /d %%E in ("%USERPROFILE%\anaconda3\envs\*") do call :try_python "%%E\python.exe"
for /d %%E in ("C:\ProgramData\miniconda3\envs\*") do call :try_python "%%E\python.exe"
for /d %%E in ("C:\ProgramData\anaconda3\envs\*") do call :try_python "%%E\python.exe"
for /d %%E in ("D:\Miniconda3\envs\*") do call :try_python "%%E\python.exe"

rem Use conda's own environment list if conda is on PATH.
if not defined PY (
    for /f "delims=" %%B in ('conda info --base 2^>nul') do call :scan_conda_envs "%%B"
)

rem Fall back to any Python on PATH, then the Windows py launcher.
if not defined PY (
    for /f "delims=" %%P in ('where python 2^>nul') do call :try_python "%%P"
)
if not defined PY (
    for /f "delims=" %%P in ('py -3.11 -c "import sys; print(sys.executable)" 2^>nul') do call :try_python "%%P"
)
if not defined PY (
    for /f "delims=" %%P in ('py -3.12 -c "import sys; print(sys.executable)" 2^>nul') do call :try_python "%%P"
)

if not defined PY (
    echo [ERROR] Python 3.11 or newer was not found.
    echo Activate a conda environment or install Python 3.11+, then try again.
    pause
    exit /b 1
)

echo Using Python: %PY%
"%PY%" -c "import flask, jieba, sklearn" >nul 2>&1
if errorlevel 1 (
    echo Installing required packages from Tsinghua mirror...
    "%PY%" -m pip install flask jieba scikit-learn -i https://pypi.tuna.tsinghua.edu.cn/simple --disable-pip-version-check
    if errorlevel 1 (
        echo [ERROR] Package installation failed. Check the network and try again.
        pause
        exit /b 1
    )
    "%PY%" -c "import flask, jieba, sklearn" >nul 2>&1
    if errorlevel 1 (
        echo [ERROR] Required packages are still unavailable in this Python.
        pause
        exit /b 1
    )
)

rem If the service is already running, just open the browser.
"%PY%" -c "import socket,sys; s=socket.socket(); s.settimeout(0.6); rc=s.connect_ex(('127.0.0.1',5000)); s.close(); sys.exit(0 if rc==0 else 1)" >nul 2>&1
if not errorlevel 1 (
    start "" "http://127.0.0.1:5000"
    exit /b 0
)

echo Starting Wenxian Library...
rem Open the browser after a short delay, while the server stays in this window.
start "" /b "%PY%" -c "import time,webbrowser; time.sleep(2); webbrowser.open('http://127.0.0.1:5000')"
"%PY%" "%~dp0app.py"

echo.
echo Server stopped.
pause
exit /b 0

:try_python
if defined PY exit /b 0
if "%~1"=="" exit /b 0
if not exist "%~1" exit /b 0
"%~1" -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)" >nul 2>&1
if errorlevel 1 exit /b 0
set "PY=%~1"
exit /b 0

:scan_conda_envs
if defined PY exit /b 0
if "%~1"=="" exit /b 0
if not exist "%~1\envs" exit /b 0
for /d %%E in ("%~1\envs\*") do call :try_python "%%E\python.exe"
exit /b 0

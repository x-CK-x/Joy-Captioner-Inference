@echo off
REM ─ JoyCaption launcher (GPU 1 only) ──────────────────────────────
set "CUDA_VISIBLE_DEVICES=1"          :: ← set to 0 for the other script
set "ENV_NAME=joycaption"
set "MINICONDA_DIR=%UserProfile%\Miniconda3"
set "ENV_PATH=%MINICONDA_DIR%\envs\%ENV_NAME%"
set "PYPORT=7864"                     :: change if you want two servers at once
set "DEBUG_LOG=%~dp0debug.log"
SET "UPDATE_ENV=0"

echo [%date% %time%]  ────────── JoyCaption launcher ────────── > "%DEBUG_LOG%"

REM ───── parse optional flags ─────────────────────────────────────
FOR %%A IN (%*) DO (
    IF "%%A"=="--update" SET NEED_UPDATE=1
)

REM ───── ensure Miniconda 3 is installed ─────────────────────────
IF EXIST "%MINICONDA_DIR%\Scripts\conda.exe" (
    echo Miniconda detected at %MINICONDA_DIR% >> "%DEBUG_LOG%"
) ELSE (
    echo Miniconda not found – installing… >> "%DEBUG_LOG%"
    echo Downloading Miniconda installer…
    curl -L -o "%TEMP%\miniconda-installer.exe" ^
         https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe
    "%TEMP%\miniconda-installer.exe" /S /D=%MINICONDA_DIR% /InstallationType=JustMe /RegisterPython=0
    DEL "%TEMP%\miniconda-installer.exe"
)

REM ───── add conda to PATH for this session ──────────────────────
SET "PATH=%MINICONDA_DIR%;%MINICONDA_DIR%\Scripts;%PATH%"
CALL conda --version >nul 2>&1 || (
    echo Conda not available even after install – aborting >> "%DEBUG_LOG%"
    echo ERROR: Conda not available.  Press any key to exit.
    echo [!] Miniconda not found in %MINICONDA_DIR%
    echo     Please install Miniconda or edit this script.
    pause >nul
    EXIT /B 1
)

REM Parse optional flags (e.g. run_local.bat --update)
FOR %%A IN (%*) DO (
    IF "%%A"=="--update" SET UPDATE_ENV=1
)



REM parse optional --update flag
FOR %%A IN (%*) DO ( IF "%%A"=="--update" SET UPDATE_ENV=1 )

REM ensure conda on PATH
SET "PATH=%MINICONDA_DIR%;%MINICONDA_DIR%\Scripts;%PATH%"

REM create / update only if needed
IF EXIST "%ENV_PATH%\python.exe" (
    echo [+] Env %ENV_NAME% already present at %ENV_PATH%
    IF "%UPDATE_ENV%"=="1" (
        echo [+] Updating from environment.yml …
        CALL conda env update -n %ENV_NAME% -f "%~dp0environment.yml"
    )
) ELSE (
    echo [+] Creating env %ENV_NAME% …
    CALL conda env create -n %ENV_NAME% -f "%~dp0environment.yml"
)

REM ─── activate existing env (create/update block removed for brevity) ─
call "%MINICONDA_DIR%\Scripts\activate.bat" %ENV_NAME%
if errorlevel 1 (
    echo [!] Failed to activate %ENV_NAME%.  Press any key to exit.
    pause >nul
    exit /b 1
)

REM ─── suppress HF symlink warning on Windows ──────────────────────
set HF_HUB_DISABLE_SYMLINKS_WARNING=1

REM ─── launch app in background, then open browser ─────────────────
start "" /b python "%~dp0app2.py" --port %PYPORT%
timeout /t 5 /nobreak >nul
start "" http://localhost:%PYPORT%

echo.
echo JoyCaption running on GPU %CUDA_VISIBLE_DEVICES%  →  http://localhost:%PYPORT%
echo Press any key to stop this server.
pause

REM ─── kill the python we started ──────────────────────────────────
taskkill /FI "WINDOWTITLE eq app2.py*" /F >nul 2>&1
taskkill /IM python.exe /F >nul 2>&1
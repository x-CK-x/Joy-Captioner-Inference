@echo off
REM ─ JoyCaption launcher – GPU 0 only ──────────────────────────────
setlocal EnableDelayedExpansion

REM ---- USER SETTINGS ---------------------------------------------
set "CUDA_VISIBLE_DEVICES=1"
set "GRADIO_SERVER_PORT=7864"
set "ENV_NAME=joycaption"
set "MINICONDA_DIR=%UserProfile%\Miniconda3"
set "DEBUG_LOG=%~dp0debug.log"
set "UPDATE_ENV=0"                          :: use --update to refresh env
REM ----------------------------------------------------------------

echo [%date% %time%]  JoyCaption launcher > "%DEBUG_LOG%"

REM ---- Parse optional --update flag ------------------------------
for %%A in (%*) do (
    if "%%A"=="--update" set "UPDATE_ENV=1"
)

REM ---- Ensure Miniconda is installed -----------------------------
if not exist "%MINICONDA_DIR%\Scripts\conda.exe" (
    echo Miniconda not found – installing… >> "%DEBUG_LOG%"
    echo Downloading Miniconda installer…
    curl -L -o "%TEMP%\miniconda-installer.exe" ^
         https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe
    "%TEMP%\miniconda-installer.exe" ^
        /InstallationType=JustMe /RegisterPython=0 /S ^
        /D=%MINICONDA_DIR%
    del "%TEMP%\miniconda-installer.exe"
)

REM ---- Add conda to PATH -----------------------------------------
set "PATH=%MINICONDA_DIR%;%MINICONDA_DIR%\Scripts;%PATH%"
conda --version >nul 2>&1 || (
    echo [!] Conda still not available – check %DEBUG_LOG%
    pause
    exit /b 1
)

REM ---- Create / update env if needed -----------------------------
set "ENV_PATH=%MINICONDA_DIR%\envs\%ENV_NAME%"
if exist "%ENV_PATH%\python.exe" (
    echo [+] Env %ENV_NAME% already present at %ENV_PATH%
    if "%UPDATE_ENV%"=="1" (
        echo [+] Updating env from environment.yml …
        conda env update -n "%ENV_NAME%" -f "%~dp0environment.yml"
    )
) else (
    echo [+] Creating env %ENV_NAME% …
    conda env create -n "%ENV_NAME%" -f "%~dp0environment.yml"
    if errorlevel 1 (
        echo [!] Failed to create env – aborting.
        pause
        exit /b 1
    )
)

REM ---- Activate ---------------------------------------------------
call "%MINICONDA_DIR%\Scripts\activate.bat" "%ENV_NAME%"
if errorlevel 1 (
    echo [!] Activation failed.
    pause
    exit /b 1
)

REM ---- Optional: silence HF symlink warning ----------------------
set HF_HUB_DISABLE_SYMLINKS_WARNING=1

REM ---- Launch app -------------------------------------------------
echo.
echo Starting JoyCaption (GPU %CUDA_VISIBLE_DEVICES%) on port %GRADIO_SERVER_PORT%
python "%~dp0app2.py"                               ^
       --port %GRADIO_SERVER_PORT%
if errorlevel 1 (
    echo [!] JoyCaption exited with an error.
    pause
    exit /b 1
)
endlocal

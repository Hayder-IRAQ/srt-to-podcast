@echo off
echo ============================================
echo  SRT-to-Podcast Setup (Chatterbox GPU)
echo ============================================
echo.

REM Check Python 3.11
where python3.11 >nul 2>&1
if %errorlevel%==0 (
    set PYTHON=python3.11
) else (
    where python >nul 2>&1
    if %errorlevel%==0 (
        set PYTHON=python
    ) else (
        echo ERROR: Python not found!
        echo Install Python 3.11 from https://www.python.org/downloads/release/python-3119/
        pause
        exit /b 1
    )
)

echo Using: %PYTHON%
%PYTHON% --version
echo.

REM Create venv
if not exist ".venv" (
    echo Creating virtual environment...
    %PYTHON% -m venv .venv
)

REM Activate
call .venv\Scripts\activate.bat

REM Install PyTorch with CUDA
echo.
echo Installing PyTorch with CUDA support...
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124

REM Install chatterbox + deps
echo.
echo Installing Chatterbox TTS and dependencies...
pip install -r requirements.txt

REM Check ffmpeg
where ffmpeg >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo WARNING: ffmpeg not found!
    echo Install: winget install ffmpeg
    echo Or download from https://ffmpeg.org/download.html
)

echo.
echo ============================================
echo  Setup complete!
echo  Run GUI:  python gui.py
echo  Run CLI:  python cli.py sample.srt -o podcast.mp3
echo ============================================
pause

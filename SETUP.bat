@echo off
title LoanSpark - First Time Setup
color 0A

echo.
echo  ================================================
echo   LoanSpark - First Time Setup
echo   This will install all required dependencies.
echo  ================================================
echo.

REM ── Check Python ──────────────────────────────────────────────
echo [1/4] Checking Python...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo  ERROR: Python not found!
    echo  Please install Python 3.11+ from: https://python.org/downloads
    echo  Make sure to check "Add Python to PATH" during install.
    echo.
    pause
    exit /b 1
)
python --version
echo  Python OK!
echo.

REM ── Install Python packages ────────────────────────────────────
echo [2/4] Installing Python packages...
echo  This may take 2-3 minutes on first run...
echo.
cd /d "%~dp0backend"
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo.
    echo  ERROR: Failed to install packages.
    echo  Try running this file as Administrator.
    echo.
    pause
    exit /b 1
)
echo.
echo  Packages installed!
echo.

REM ── Check Ollama ───────────────────────────────────────────────
echo [3/4] Checking Ollama...
ollama --version >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo  WARNING: Ollama not found.
    echo  Download from: https://ollama.com/download
    echo  Install it, then run SETUP.bat again.
    echo.
    echo  NOTE: App will still work without Ollama
    echo  but AI explanations will use fallback text.
    echo.
) else (
    echo  Ollama found!
    echo.
    echo [4/4] Downloading AI model ^(mistral ~4GB^)...
    echo  This is a one-time download. Please wait...
    echo.
    ollama pull mistral
    echo.
    echo  AI model ready!
)

echo.
echo  ================================================
echo   Setup Complete! 
echo   Now run START.bat to launch LoanSpark.
echo  ================================================
echo.
pause

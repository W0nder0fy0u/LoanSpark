@echo off
title LoanSpark - Starting...
color 0A

echo.
echo  ================================================
echo   LoanSpark AI Loan System
echo  ================================================
echo.

REM ── Check Python ──────────────────────────────────────────────
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo  ERROR: Python not found. Please run SETUP.bat first.
    pause
    exit /b 1
)

REM ── Start Ollama in background (silently, ignore if not installed) ──
echo [1/3] Starting Ollama AI engine...
ollama --version >nul 2>&1
if %errorlevel% equ 0 (
    start /min "" ollama serve
    timeout /t 2 /nobreak >nul
    echo  Ollama started.
) else (
    echo  Ollama not found - running without AI ^(fallback mode^).
)
echo.

REM ── Start FastAPI backend ──────────────────────────────────────
echo [2/3] Starting backend server...
cd /d "%~dp0backend"
start "LoanSpark Backend" /min cmd /c "python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload 2>&1 | more"
echo  Backend starting on http://localhost:8000
echo.

REM ── Wait for backend to be ready ──────────────────────────────
echo [3/3] Waiting for server to be ready...
timeout /t 4 /nobreak >nul

REM Try up to 10 times (10 seconds) to confirm server is up
set /a attempts=0
:WAIT_LOOP
set /a attempts+=1
curl -s http://localhost:8000/health >nul 2>&1
if %errorlevel% equ 0 goto SERVER_READY
if %attempts% geq 10 goto SERVER_TIMEOUT
timeout /t 1 /nobreak >nul
goto WAIT_LOOP

:SERVER_TIMEOUT
echo  Server taking longer than expected... opening anyway.
goto OPEN_BROWSER

:SERVER_READY
echo  Server is ready!

:OPEN_BROWSER
echo.
echo  ================================================
echo   Opening LoanSpark in your browser...
echo.
echo   Applicant App : frontend\index.html
echo   Bank Dashboard: frontend\bank.html
echo   Bank Key      : BANK_SECRET_2024
echo  ================================================
echo.

REM Open both pages
start "" "%~dp0frontend\index.html"
timeout /t 1 /nobreak >nul
start "" "%~dp0frontend\bank.html"

echo.
echo  LoanSpark is running!
echo  Close this window to STOP the server.
echo.
echo  Press any key to stop LoanSpark and exit...
pause >nul

REM ── Cleanup on exit ───────────────────────────────────────────
echo.
echo  Shutting down...
taskkill /f /fi "WINDOWTITLE eq LoanSpark Backend" >nul 2>&1
taskkill /f /im ollama.exe >nul 2>&1
echo  LoanSpark stopped. Goodbye!
timeout /t 2 /nobreak >nul

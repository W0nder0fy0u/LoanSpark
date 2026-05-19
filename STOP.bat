@echo off
title LoanSpark - Stopping
color 0C

echo.
echo  Stopping LoanSpark...
echo.

taskkill /f /fi "WINDOWTITLE eq LoanSpark Backend" >nul 2>&1
taskkill /f /im ollama.exe >nul 2>&1

REM Kill any python process running uvicorn on port 8000
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":8000"') do (
    taskkill /f /pid %%a >nul 2>&1
)

echo  All LoanSpark processes stopped.
echo.
timeout /t 2 /nobreak >nul

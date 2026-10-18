@echo off
title Training Data Factory - Web Interface
cd /d "%~dp0"

echo ======================================================================
echo   Training Data Factory - Modern Web Interface
echo ======================================================================
echo.

set PYTHON_EXE="C:\Users\USER\anaconda3\python.exe"

if not exist %PYTHON_EXE% (
    set PYTHON_EXE=python
)

echo  Local Dashboard URL: http://127.0.0.1:8765
echo  Starting server and launching default browser...
echo.
echo  [Press Ctrl+C to stop the server at any time]
echo ======================================================================
echo.

:: Load environment variables from .env if present
if exist "%~dp0.env" (
    for /f "usebackq tokens=1,* delims==" %%A in ("%~dp0.env") do (
        if not "%%A"=="" if not "%%B"=="" set "%%A=%%B"
    )
)

:: Automatically open default browser after a brief 2-second delay
start /b "" cmd /c "timeout /t 2 /nobreak >nul & start http://127.0.0.1:8765"

:: Launch the FastAPI server
%PYTHON_EXE% -m training_data_bot.web --host 127.0.0.1 --port 8765

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Server stopped with error code %ERRORLEVEL%.
    pause
)

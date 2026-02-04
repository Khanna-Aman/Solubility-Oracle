@echo off
REM Run FastAPI Backend
echo ============================================
echo Starting FastAPI Backend...
echo ============================================
echo.

REM Check if virtual environment exists
if not exist "..\..\.venv-solubility\Scripts\uvicorn.exe" (
    echo ERROR: Virtual environment not found or uvicorn not installed!
    echo Please run install.bat first.
    pause
    exit /b 1
)

REM Run FastAPI directly from venv
echo Starting API server at http://localhost:8050
echo API docs available at http://localhost:8050/docs
echo Press Ctrl+C to stop the server
echo.
cd ..
.venv-solubility\Scripts\uvicorn.exe src.api.main:app --host 0.0.0.0 --port 8050 --reload

pause

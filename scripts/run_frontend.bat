@echo off
REM Run React Frontend
echo ============================================
echo Starting React Frontend...
echo ============================================
echo.

REM Check if frontend directory exists
if not exist "frontend" (
    echo ERROR: Frontend directory not found!
    pause
    exit /b 1
)

REM Check if node_modules exists
if not exist "frontend\node_modules" (
    echo WARNING: Frontend dependencies not installed!
    echo Installing now...
    cd frontend
    call npm install
    if errorlevel 1 (
        echo ERROR: Failed to install frontend dependencies
        pause
        exit /b 1
    )
    cd ..
)

REM Run frontend
cd frontend
echo Starting frontend dev server at http://localhost:5173
echo Press Ctrl+C to stop the server
echo.
call npm run dev

pause

@echo off
REM Verify Solubility Oracle setup
echo ============================================
echo Solubility Oracle - Setup Verification
echo ============================================
echo.

REM Check Python
echo [1/6] Checking Python...
python --version >nul 2>&1
if errorlevel 1 (
    py --version >nul 2>&1
    if errorlevel 1 (
        echo   X Python not found
        goto :error
    ) else (
        py --version
        echo   √ Python found (py launcher)
    )
) else (
    python --version
    echo   √ Python found
)
echo.

REM Check virtual environment
echo [2/6] Checking virtual environment...
if exist ".venv-solubility\Scripts\python.exe" (
    echo   √ Virtual environment exists
) else (
    echo   X Virtual environment not found
    echo   Run: install.bat
    goto :error
)
echo.

REM Check Node.js
echo [3/6] Checking Node.js...
node --version >nul 2>&1
if errorlevel 1 (
    echo   X Node.js not found (required for frontend)
    echo   Download from: https://nodejs.org/
) else (
    node --version
    echo   √ Node.js found
)
echo.

REM Check dataset
echo [4/6] Checking dataset...
if exist "data\processed\solubility_combined.csv" (
    echo   √ Dataset found
) else (
    echo   X Dataset not found
    echo   Run: .venv-solubility\Scripts\python.exe scripts\download_data.py
)
echo.

REM Check model checkpoint
echo [5/6] Checking model checkpoint...
if exist "checkpoints\best_model.pt" (
    echo   √ Model checkpoint found
) else (
    echo   ! Model checkpoint not found (optional)
    echo   Train model: .venv-solubility\Scripts\python.exe train.py --epochs 30
)
echo.

REM Check frontend dependencies
echo [6/6] Checking frontend dependencies...
if exist "frontend\node_modules" (
    echo   √ Frontend dependencies installed
) else (
    echo   ! Frontend dependencies not installed
    echo   Run: cd frontend ^&^& npm install
)
echo.

echo ============================================
echo Setup verification complete!
echo ============================================
echo.
echo Next steps:
echo   1. Train model (if not done): .venv-solubility\Scripts\python.exe train.py --epochs 30
echo   2. Run Streamlit: run_streamlit.bat
echo   3. Or run full stack: run_backend.bat + run_frontend.bat
echo.
pause
exit /b 0

:error
echo.
echo ============================================
echo Setup verification FAILED
echo ============================================
echo Please fix the errors above and try again.
pause
exit /b 1


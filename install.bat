@echo off
REM Solubility Oracle - Installation Script for Windows CMD
echo ============================================
echo Solubility Oracle - Installation
echo ============================================
echo.

REM Check Python
echo [1/3] Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found! Please install Python 3.10+
    pause
    exit /b 1
)
python --version
echo.

REM Create virtual environment if it doesn't exist
echo [2/3] Setting up Python virtual environment...
if not exist ".venv-solubility" (
    echo Creating virtual environment...
    python -m venv .venv-solubility
)
echo.

REM Activate virtual environment and install Python dependencies
echo [3/3] Installing Python dependencies...
call .venv-solubility\Scripts\activate.bat
echo Upgrading pip...
python -m pip install --upgrade pip
echo Installing packages from requirements.txt...
echo This may take several minutes, please wait...
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install Python dependencies
    pause
    exit /b 1
)
echo.

REM Check Node.js
echo [4/4] Checking Node.js installation...
node --version >nul 2>&1
if errorlevel 1 (
    echo WARNING: Node.js not found! Frontend will not work.
    echo Please install Node.js 18+ to use the React frontend.
    echo.
) else (
    node --version
    echo.
    echo Installing frontend dependencies...
    cd frontend
    call npm install
    if errorlevel 1 (
        echo WARNING: Failed to install frontend dependencies
    ) else (
        echo Frontend dependencies installed successfully!
    )
    cd ..
    echo.
)

echo ============================================
echo Installation complete!
echo ============================================
echo.
echo To run the application:
echo   1. Streamlit: run_streamlit.bat
echo   2. Full Stack: run_backend.bat (Terminal 1) and run_frontend.bat (Terminal 2)
echo.
pause

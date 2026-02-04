@echo off
REM Run all tests for Solubility Oracle
echo ============================================
echo Running Solubility Oracle Tests
echo ============================================
echo.

REM Check if virtual environment exists
if not exist ".venv-solubility\Scripts\python.exe" (
    echo ERROR: Virtual environment not found!
    echo Please run install.bat first.
    pause
    exit /b 1
)

echo Running pytest tests...
echo.
.venv-solubility\Scripts\pytest.exe tests\ -v --tb=short

echo.
echo ============================================
echo Tests complete!
echo ============================================
pause


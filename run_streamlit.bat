@echo off
REM Run Streamlit Dashboard
echo ============================================
echo Starting Streamlit Dashboard...
echo ============================================
echo.

REM Check if virtual environment exists
if not exist ".venv-solubility\Scripts\streamlit.exe" (
    echo ERROR: Virtual environment not found or streamlit not installed!
    echo Please run install.bat first.
    pause
    exit /b 1
)

REM Run Streamlit directly from venv
echo Opening Streamlit dashboard at http://localhost:8501
echo Press Ctrl+C to stop the server
echo.
.venv-solubility\Scripts\streamlit.exe run streamlit_app/app.py

pause

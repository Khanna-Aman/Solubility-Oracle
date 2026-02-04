# Quick Start Guide - Windows CMD

## 🚀 Fastest Way to Get Started

### Step 1: Install Dependencies

Open **Command Prompt (CMD)** in the project directory and run:

```cmd
install.bat
```

This will:
- ✅ Create Python virtual environment
- ✅ Install all Python packages
- ✅ Install frontend dependencies (if Node.js is installed)

### Step 2: Run the Application

**Option A: Streamlit Dashboard (Easiest)**

Double-click `run_streamlit.bat` or run:
```cmd
run_streamlit.bat
```

Then open http://localhost:8501 in your browser.

**Option B: Full Stack (FastAPI + React)**

**Terminal 1 - Backend:**
```cmd
run_backend.bat
```

**Terminal 2 - Frontend:**
```cmd
run_frontend.bat
```

Then open http://localhost:5173 in your browser.

---

## 📋 Manual Installation (If batch files don't work)

### 1. Create Virtual Environment
```cmd
python -m venv .venv-solubility
```

### 2. Activate Virtual Environment
```cmd
.venv-solubility\Scripts\activate.bat
```

### 3. Install Python Dependencies
```cmd
pip install -r requirements.txt
```

### 4. Install Frontend Dependencies
```cmd
cd frontend
npm install
cd ..
```

### 5. Run Streamlit
```cmd
.venv-solubility\Scripts\activate.bat
streamlit run streamlit_app/app.py
```

---

## ✅ Verify Installation

Run the test script:
```cmd
test_setup.bat
```

This will check:
- Python installation
- Package imports
- Model imports
- Frontend dependencies
- Batch files

---

## 🐛 Troubleshooting

**Problem: "Python not found"**
- Install Python 3.10+ from python.org
- Make sure Python is added to PATH

**Problem: "pip not found"**
- Use: `python -m pip install -r requirements.txt`

**Problem: "Node.js not found"**
- Install Node.js 18+ from nodejs.org
- Frontend won't work without it, but Streamlit will

**Problem: Batch files don't run**
- Right-click → Run as Administrator
- Or run commands manually in CMD

---

## 📝 Available Batch Files

- `install.bat` - Install all dependencies
- `run_streamlit.bat` - Run Streamlit dashboard
- `run_backend.bat` - Run FastAPI backend
- `run_frontend.bat` - Run React frontend
- `test_setup.bat` - Test installation
- `setup.bat` - Quick setup (venv + pip install)

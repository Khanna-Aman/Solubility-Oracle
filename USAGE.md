# Usage Guide - Solubility Oracle

## Quick Start (Windows CMD)

### 1. Install Dependencies

Open **Command Prompt (CMD)** in the project directory and run:

```cmd
install.bat
```

This will:
- Create Python virtual environment
- Install all Python packages
- Install frontend dependencies (if Node.js is installed)

**Or manually:**
```cmd
python -m venv .venv-solubility
.venv-solubility\Scripts\activate.bat
python -m pip install -r requirements.txt
cd frontend
npm install
cd ..
```

### 2. Run Streamlit Dashboard (Easiest)

Double-click `run_streamlit.bat` or run:
```cmd
run_streamlit.bat
```

Then open http://localhost:8501 in your browser.

**Or manually:**
```cmd
.venv-solubility\Scripts\activate.bat
streamlit run streamlit_app/app.py
```

### 3. Run Full Stack (FastAPI + React)

**Terminal 1 - Backend:**
```cmd
run_backend.bat
```

**Terminal 2 - Frontend:**
```cmd
run_frontend.bat
```

Then open http://localhost:5173 in your browser.

**Or manually:**

Terminal 1:
```cmd
.venv-solubility\Scripts\activate.bat
uvicorn api.main:app --host 0.0.0.0 --port 8050 --reload
```

Terminal 2:
```cmd
cd frontend
npm run dev
```

## Using the Models

### Python API

```python
from models.ensemble import SolubilityEnsemble

# Load model (after training)
model = SolubilityEnsemble.load('checkpoints/best_model.pt')

# Predict solubility
smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"  # Aspirin
prediction = model.predict(smiles)
print(f"Predicted LogS: {prediction:.2f}")
```

### REST API

```cmd
REM Health check
curl http://localhost:8050/api/v1/health

REM Predict solubility
curl -X POST http://localhost:8050/api/v1/predict -H "Content-Type: application/json" -d "{\"smiles\": \"CC(=O)OC1=CC=CC=C1C(=O)O\"}"
```

## Training Models

1. Prepare your dataset in CSV format with `SMILES` and `LogS` columns
2. Run training script:
```cmd
.venv-solubility\Scripts\activate.bat
python train.py --data_path data/raw/solubility_data.csv
```

## Project Structure

- `models/` - Model implementations (AttentiveFP, Hybrid, Ensemble)
- `utils/` - Molecular processing utilities
- `api/` - FastAPI backend
- `frontend/` - React frontend
- `streamlit_app/` - Streamlit dashboard
- `data/` - Dataset storage
- `checkpoints/` - Saved model weights

## Available Batch Files

- `install.bat` - Install all dependencies
- `run_streamlit.bat` - Run Streamlit dashboard
- `run_backend.bat` - Run FastAPI backend
- `run_frontend.bat` - Run React frontend
- `test_setup.bat` - Test installation

## Notes

- Models need to be trained before making predictions
- Place trained model checkpoint in `checkpoints/best_model.pt`
- The demo mode will work but show placeholder predictions

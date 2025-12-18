# 💧 Solubility Oracle

**AI-Powered Aqueous Solubility Prediction System**

Predict molecular solubility using Graph Neural Networks (GNNs) with AttentiveFP architecture.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![React](https://img.shields.io/badge/React-18+-61dafb.svg)](https://reactjs.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🎯 Overview

**Solubility Oracle** is a comprehensive machine learning system for predicting aqueous solubility (LogS) of chemical compounds. Built on the AttentiveFP (Attentive Fingerprinting) architecture, it combines graph neural networks with molecular fingerprints for accurate solubility predictions.

### Key Features

- 🧪 **SMILES-Based Predictions** - Input molecular structures as SMILES strings
- 🤖 **AttentiveFP Architecture** - State-of-the-art graph attention mechanism
- 📊 **Hybrid Model** - Combines GNN features with molecular descriptors
- 🎯 **Ensemble Predictions** - Multiple model averaging for robustness
- 🌐 **Interactive Web UI** - React frontend with real-time predictions
- 📈 **Streamlit Dashboard** - Quick prototyping and visualization
- ⚡ **Intel GPU Optimization** - Hardware-accelerated inference

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│              Solubility Oracle Architecture                  │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  SMILES Input → RDKit → Graph Construction → AttentiveFP    │
│                    ↓                              ↓          │
│              Descriptors ──────────────→ Hybrid Model        │
│                                              ↓               │
│                                         LogS Prediction      │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Technology Stack

| Component | Technology |
|-----------|-----------|
| **Deep Learning** | PyTorch, PyTorch Geometric |
| **Chemistry** | RDKit, Mordred |
| **Backend** | FastAPI, Python 3.10+ |
| **Frontend** | React.js, TypeScript, Vite |
| **Visualization** | Streamlit, Plotly |
| **Hardware** | Intel Extension for PyTorch (IPEX) |

---

## 📁 Project Structure

```
solubility-oracle/
├── api/                    # FastAPI backend
│   ├── main.py            # API endpoints
│   └── routes/            # Route handlers
├── data/                  # Dataset management
│   ├── raw/              # Original datasets
│   ├── processed/        # Preprocessed data
│   └── splits/           # Train/val/test splits
├── models/               # Model implementations
│   ├── attentivefp.py   # AttentiveFP architecture
│   ├── hybrid.py        # Hybrid model
│   └── ensemble.py      # Ensemble wrapper
├── frontend/            # React application
│   ├── src/
│   └── public/
├── streamlit_app/       # Streamlit dashboard
│   ├── app.py
│   └── styles/
├── notebooks/           # Jupyter notebooks
└── tests/              # Test suite
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+ (for frontend)
- Intel GPU (optional, for acceleration)

### Installation

**1. Clone the repository**
```bash
git clone https://github.com/Khanna-Aman/Solubility_Oracle.git
cd Solubility_Oracle
```

**2. Set up Python environment**
```bash
python -m venv .venv-solubility
.venv-solubility\Scripts\activate  # Windows
# source .venv-solubility/bin/activate  # Linux/Mac

pip install -r requirements.txt
```

**3. Install frontend dependencies**
```bash
cd frontend
npm install
```

### Running the Application

**Option 1: Streamlit Dashboard (Quick Start)**
```bash
streamlit run streamlit_app/app.py
```

**Option 2: Full Stack (FastAPI + React)**

Terminal 1 - Backend:
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8050 --reload
```

Terminal 2 - Frontend:
```bash
cd frontend
npm run dev
```

Open http://localhost:5173 in your browser.

---

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| **Test RMSE** | TBD |
| **Test MAE** | TBD |
| **Test R²** | TBD |
| **Parameters** | ~500K |
| **Inference Time** | ~50ms |

---

## 🧪 Usage Example

```python
from models.ensemble import SolubilityEnsemble

# Load model
model = SolubilityEnsemble.load('checkpoints/best_model.pt')

# Predict solubility
smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"  # Aspirin
prediction = model.predict(smiles)

print(f"Predicted LogS: {prediction:.2f}")
```

---

## 📚 Dataset

- **Source**: AqSolDB, ESOL, Delaney
- **Size**: ~9,000 compounds
- **Split**: 80% train, 10% validation, 10% test
- **Target**: LogS (aqueous solubility)

---

## 🔬 Technical Decisions

| Decision | Rationale |
|----------|-----------|
| **AttentiveFP** | Superior performance on molecular property prediction |
| **Scaffold Split** | Prevents data leakage, tests generalization |
| **Ensemble** | Reduces variance, improves robustness |
| **Intel GPU** | Cost-effective hardware acceleration |

---

## 📖 License

MIT License - See [LICENSE](LICENSE) for details.

---

**Version**: 1.0.0 | **Status**: Active Development


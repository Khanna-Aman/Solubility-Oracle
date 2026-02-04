# 🧪 Testing Guide - Solubility Oracle

This document describes how to test the Solubility Oracle application.

---

## Quick Test Commands

### 1. Run All Tests
```cmd
run_tests.bat
```

Or manually:
```cmd
.venv-solubility\Scripts\pytest.exe tests\ -v
```

### 2. Verify Setup
```cmd
verify_setup.bat
```

This checks:
- ✅ Python installation
- ✅ Virtual environment
- ✅ Node.js (for frontend)
- ✅ Dataset availability
- ✅ Model checkpoint
- ✅ Frontend dependencies

---

## Test Coverage

### Unit Tests

**Model Tests** (`tests/test_models.py`):
- ✅ AttentiveFP model creation
- ✅ AttentiveFP forward pass
- ✅ HybridModel creation
- ✅ HybridModel forward pass
- ✅ SolubilityEnsemble creation
- ✅ Model parameter counts

**Utility Tests** (`tests/test_utils.py`):
- ✅ SMILES validation (valid/invalid)
- ✅ Graph conversion from SMILES
- ✅ Molecular descriptor calculation
- ✅ Descriptor consistency

---

## Manual Testing

### Test Streamlit Dashboard

1. **Start Streamlit:**
   ```cmd
   run_streamlit.bat
   ```

2. **Test in browser** (http://localhost:8501):
   - Click example molecule buttons
   - Enter custom SMILES
   - Verify predictions display
   - Check error handling for invalid SMILES

### Test Backend API

1. **Start Backend:**
   ```cmd
   run_backend.bat
   ```

2. **Test API docs** (http://localhost:8050/docs):
   - Try `/api/v1/health` endpoint
   - Test `/api/v1/predict` with example SMILES
   - Verify error responses

3. **Test with curl:**
   ```cmd
   curl -X POST "http://localhost:8050/api/v1/predict" ^
        -H "Content-Type: application/json" ^
        -d "{\"smiles\": \"CCO\"}"
   ```

### Test Full Stack

1. **Terminal 1 - Backend:**
   ```cmd
   run_backend.bat
   ```

2. **Terminal 2 - Frontend:**
   ```cmd
   run_frontend.bat
   ```

3. **Test in browser** (http://localhost:5173):
   - Click example molecules
   - Verify predictions
   - Check UI responsiveness
   - Test error handling

---

## Example Test Molecules

| Molecule | SMILES | Expected LogS Range |
|----------|--------|---------------------|
| Ethanol | `CCO` | -0.5 to 0.5 |
| Aspirin | `CC(=O)OC1=CC=CC=C1C(=O)O` | -2.0 to -1.0 |
| Caffeine | `CN1C=NC2=C1C(=O)N(C(=O)N2C)C` | -1.0 to 0.0 |
| Benzene | `c1ccccc1` | -2.5 to -1.5 |
| Glucose | `C(C1C(C(C(C(O1)O)O)O)O)O` | 0.0 to 1.0 |

---

## Troubleshooting Tests

### Import Errors
If you see `ModuleNotFoundError`:
- Ensure virtual environment is activated
- Check that you're running from project root
- Verify all dependencies installed: `pip install -r requirements.txt`

### Model Not Found
If tests fail due to missing model:
- Train the model first: `.venv-solubility\Scripts\python.exe train.py --epochs 30`
- Or skip prediction tests (model tests don't require trained checkpoint)

### Frontend Tests
Frontend uses Vite dev server:
- Ensure Node.js is installed
- Run `cd frontend && npm install` if node_modules missing
- Check that backend is running on port 8050

---

## Continuous Testing

For development, run tests automatically:

```cmd
.venv-solubility\Scripts\pytest.exe tests\ --watch
```

Or use pytest-watch:
```cmd
pip install pytest-watch
ptw tests/
```

---

## Test Results

After running tests, you should see:
```
tests/test_models.py::test_attentivefp_creation PASSED
tests/test_models.py::test_attentivefp_forward PASSED
tests/test_models.py::test_hybrid_model_creation PASSED
tests/test_models.py::test_hybrid_model_forward PASSED
tests/test_models.py::test_ensemble_creation PASSED
tests/test_models.py::test_model_parameters PASSED
tests/test_utils.py::test_validate_smiles_valid PASSED
tests/test_utils.py::test_validate_smiles_invalid PASSED
tests/test_utils.py::test_smiles_to_graph_valid PASSED
tests/test_utils.py::test_smiles_to_graph_invalid PASSED
tests/test_utils.py::test_get_molecular_descriptors_valid PASSED
tests/test_utils.py::test_get_molecular_descriptors_invalid PASSED
tests/test_utils.py::test_descriptors_consistency PASSED

============ 13 passed in X.XXs ============
```

---

## Next Steps

After all tests pass:
1. ✅ Train the model (if not done)
2. ✅ Test all three interfaces (Streamlit, Backend, Frontend)
3. ✅ Ready for deployment!


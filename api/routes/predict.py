"""
Prediction API routes
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import torch
import sys
from pathlib import Path

# Add parent directory to path for imports
parent_dir = str(Path(__file__).parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from models.ensemble import SolubilityEnsemble
from utils.molecular import validate_smiles

router = APIRouter(prefix="/api/v1", tags=["prediction"])

# Global model instance (loaded on startup)
model: Optional[SolubilityEnsemble] = None


class PredictionRequest(BaseModel):
    """Request model for solubility prediction"""
    smiles: str = Field(..., description="SMILES string of the molecule", example="CC(=O)OC1=CC=CC=C1C(=O)O")
    
    class Config:
        schema_extra = {
            "example": {
                "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O"
            }
        }


class PredictionResponse(BaseModel):
    """Response model for solubility prediction"""
    smiles: str
    predicted_logs: float = Field(..., description="Predicted LogS value")
    success: bool = True
    message: str = "Prediction successful"


@router.post("/predict", response_model=PredictionResponse)
async def predict_solubility(request: PredictionRequest):
    """
    Predict aqueous solubility (LogS) for a molecule given its SMILES string.
    
    Args:
        request: Prediction request with SMILES string
        
    Returns:
        Prediction response with LogS value
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please ensure model checkpoint is available."
        )
    
    # Validate SMILES
    if not validate_smiles(request.smiles):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid SMILES string: {request.smiles}"
        )
    
    try:
        # Get device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Predict
        prediction = model.predict(request.smiles, device=device)
        
        if prediction is None:
            raise HTTPException(
                status_code=500,
                detail="Failed to generate prediction. Please check SMILES string."
            )
        
        return PredictionResponse(
            smiles=request.smiles,
            predicted_logs=round(prediction, 4),
            success=True,
            message="Prediction successful"
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }


def load_model(model_path: str = "checkpoints/best_model.pt"):
    """Load model on startup"""
    global model
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = SolubilityEnsemble.load(model_path, device=device)
        print(f"Model loaded successfully from {model_path}")
    except Exception as e:
        print(f"Warning: Could not load model from {model_path}: {e}")
        print("API will run but predictions will fail until model is loaded.")

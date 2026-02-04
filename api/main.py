"""
FastAPI main application
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path for imports
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from api.routes.predict import router as predict_router, load_model

# Load environment variables
load_dotenv()

# Create FastAPI app
app = FastAPI(
    title="Solubility Oracle API",
    description="AI-Powered Aqueous Solubility Prediction API using AttentiveFP",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],  # React dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(predict_router)


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Solubility Oracle API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/v1/health"
    }


@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    model_path = os.getenv("MODEL_PATH", "checkpoints/best_model.pt")
    load_model(model_path)


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("API_PORT", 8050))
    uvicorn.run(app, host="0.0.0.0", port=port)

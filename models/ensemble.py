"""
Ensemble Model: Combines multiple models for robust predictions
"""

import torch
import torch.nn as nn
from typing import List, Optional, Union
import os
from pathlib import Path

from .attentivefp import AttentiveFP
from .hybrid import HybridModel


class SolubilityEnsemble(nn.Module):
    """
    Ensemble wrapper that combines multiple models for prediction.
    Uses averaging or weighted averaging of predictions.
    """
    
    def __init__(self, models: List[nn.Module], weights: Optional[List[float]] = None):
        """
        Initialize ensemble.
        
        Args:
            models: List of trained models
            weights: Optional weights for each model (default: equal weights)
        """
        super(SolubilityEnsemble, self).__init__()
        
        self.models = nn.ModuleList(models)
        
        if weights is None:
            weights = [1.0 / len(models)] * len(models)
        
        self.weights = torch.tensor(weights, dtype=torch.float32)
        self.weights = self.weights / self.weights.sum()  # Normalize
    
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        Forward pass through all models and average predictions.
        """
        predictions = []
        for model in self.models:
            pred = model(*args, **kwargs)
            predictions.append(pred)
        
        # Stack and weight
        stacked = torch.stack(predictions, dim=0)
        weighted = (stacked * self.weights.view(-1, 1, 1)).sum(dim=0)
        
        return weighted
    
    def predict(self, smiles: str, device: str = 'cpu') -> Optional[float]:
        """
        Predict solubility using ensemble.
        
        Args:
            smiles: SMILES string
            device: Device to run inference on
            
        Returns:
            Predicted LogS value or None if invalid
        """
        predictions = []
        
        for model in self.models:
            if hasattr(model, 'predict'):
                pred = model.predict(smiles, device=device)
                if pred is not None:
                    predictions.append(pred)
        
        if len(predictions) == 0:
            return None
        
        # Weighted average
        weights = self.weights.cpu().numpy()[:len(predictions)]
        weights = weights / weights.sum()
        
        prediction = sum(p * w for p, w in zip(predictions, weights))
        
        return float(prediction)
    
    @classmethod
    def load(cls, checkpoint_path: str, device: str = 'cpu') -> 'SolubilityEnsemble':
        """
        Load ensemble from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            device: Device to load model on
            
        Returns:
            Loaded ensemble model
        """
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Create models based on checkpoint
        models = []
        if 'models' in checkpoint:
            for model_state in checkpoint['models']:
                model_type = model_state['type']
                state_dict = model_state['state_dict']
                config = model_state['config']
                
                if model_type == 'AttentiveFP':
                    model = AttentiveFP(**config)
                elif model_type == 'HybridModel':
                    model = HybridModel(**config)
                else:
                    raise ValueError(f"Unknown model type: {model_type}")
                
                model.load_state_dict(state_dict)
                model.to(device)
                model.eval()
                models.append(model)
        
        weights = checkpoint.get('weights', None)
        ensemble = cls(models, weights)
        
        return ensemble
    
    def save(self, checkpoint_path: str):
        """
        Save ensemble to checkpoint.
        
        Args:
            checkpoint_path: Path to save checkpoint
        """
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        
        models_data = []
        for model in self.models:
            model_data = {
                'type': model.__class__.__name__,
                'state_dict': model.state_dict(),
                'config': self._get_model_config(model)
            }
            models_data.append(model_data)
        
        checkpoint = {
            'models': models_data,
            'weights': self.weights.cpu().numpy().tolist()
        }
        
        torch.save(checkpoint, checkpoint_path)
    
    def _get_model_config(self, model: nn.Module) -> dict:
        """Extract model configuration."""
        if isinstance(model, AttentiveFP):
            return {
                'num_node_features': model.num_node_features,
                'num_edge_features': model.num_edge_features,
                'hidden_dim': model.hidden_dim,
                'num_layers': model.num_layers,
                'num_timesteps': model.num_timesteps,
                'dropout': model.dropout,
                'output_dim': 1
            }
        elif isinstance(model, HybridModel):
            return {
                'num_node_features': 7,
                'num_edge_features': 3,
                'descriptor_dim': model.descriptor_dim,
                'gnn_hidden_dim': model.gnn.hidden_dim,
                'gnn_num_layers': model.gnn.num_layers,
                'fusion_hidden_dim': 256,
                'dropout': model.gnn.dropout,
                'output_dim': 1
            }
        else:
            return {}

"""
Hybrid Model: Combines AttentiveFP GNN features with molecular descriptors
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import numpy as np

from .attentivefp import AttentiveFP
from utils.molecular import smiles_to_graph, get_molecular_descriptors


class HybridModel(nn.Module):
    """
    Hybrid model combining graph neural network (AttentiveFP) 
    with traditional molecular descriptors.
    """
    
    def __init__(
        self,
        num_node_features: int = 7,
        num_edge_features: int = 3,
        descriptor_dim: int = 200,
        gnn_hidden_dim: int = 128,
        gnn_num_layers: int = 3,
        fusion_hidden_dim: int = 256,
        dropout: float = 0.2,
        output_dim: int = 1
    ):
        """
        Initialize Hybrid Model.
        
        Args:
            num_node_features: Number of atom features
            num_edge_features: Number of bond features
            descriptor_dim: Dimension of molecular descriptors
            gnn_hidden_dim: Hidden dimension for GNN
            gnn_num_layers: Number of GNN layers
            fusion_hidden_dim: Hidden dimension for fusion layer
            dropout: Dropout rate
            output_dim: Output dimension
        """
        super(HybridModel, self).__init__()
        
        self.descriptor_dim = descriptor_dim
        
        # GNN branch (AttentiveFP)
        self.gnn = AttentiveFP(
            num_node_features=num_node_features,
            num_edge_features=num_edge_features,
            hidden_dim=gnn_hidden_dim,
            num_layers=gnn_num_layers,
            dropout=dropout,
            output_dim=gnn_hidden_dim  # Output GNN features, not final prediction
        )
        
        # Descriptor branch
        self.descriptor_net = nn.Sequential(
            nn.Linear(descriptor_dim, descriptor_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(descriptor_dim // 2, gnn_hidden_dim)
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(gnn_hidden_dim * 2, fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, fusion_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim // 2, output_dim)
        )
    
    def forward(self, data: torch.Tensor, descriptors: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            data: PyTorch Geometric Data object (graph)
            descriptors: Molecular descriptors tensor
            
        Returns:
            Predicted LogS values
        """
        # GNN branch
        gnn_features = self.gnn(data)
        
        # Descriptor branch
        desc_features = self.descriptor_net(descriptors)
        
        # Concatenate and fuse
        combined = torch.cat([gnn_features, desc_features], dim=1)
        output = self.fusion(combined)
        
        return output
    
    def predict(self, smiles: str, device: str = 'cpu') -> Optional[float]:
        """
        Predict solubility for a SMILES string.
        
        Args:
            smiles: SMILES string
            device: Device to run inference on
            
        Returns:
            Predicted LogS value or None if invalid
        """
        # Get graph
        graph = smiles_to_graph(smiles, device=device)
        if graph is None:
            return None
        
        # Get descriptors
        descriptors = get_molecular_descriptors(smiles)
        if descriptors is None:
            return None
        
        # Pad/truncate descriptors to expected dimension
        if len(descriptors) < self.descriptor_dim:
            descriptors = np.pad(descriptors, (0, self.descriptor_dim - len(descriptors)))
        else:
            descriptors = descriptors[:self.descriptor_dim]
        
        descriptors = torch.tensor(descriptors, dtype=torch.float32).unsqueeze(0).to(device)
        
        self.eval()
        with torch.no_grad():
            graph.batch = torch.zeros(graph.x.size(0), dtype=torch.long, device=device)
            prediction = self.forward(graph, descriptors)
            return prediction.item()

"""
AttentiveFP: Attentive Fingerprinting for Molecular Property Prediction
Based on the architecture from Xiong et al. (2020)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_add_pool, global_mean_pool
from torch_geometric.data import Data
from typing import Optional
import numpy as np


class AttentiveFP(nn.Module):
    """
    AttentiveFP model for molecular property prediction.
    
    Uses graph attention mechanism to learn molecular representations
    for predicting aqueous solubility (LogS).
    """
    
    def __init__(
        self,
        num_node_features: int = 7,
        num_edge_features: int = 3,
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_timesteps: int = 2,
        dropout: float = 0.2,
        output_dim: int = 1
    ):
        """
        Initialize AttentiveFP model.
        
        Args:
            num_node_features: Number of atom features
            num_edge_features: Number of bond features
            hidden_dim: Hidden dimension size
            num_layers: Number of GCN layers
            num_timesteps: Number of attention timesteps
            dropout: Dropout rate
            output_dim: Output dimension (1 for regression)
        """
        super(AttentiveFP, self).__init__()
        
        self.num_node_features = num_node_features
        self.num_edge_features = num_edge_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_timesteps = num_timesteps
        self.dropout = dropout
        
        # Node feature embedding
        self.node_embedding = nn.Linear(num_node_features, hidden_dim)
        
        # Edge feature embedding
        self.edge_embedding = nn.Linear(num_edge_features, hidden_dim)
        
        # GCN layers
        self.gcn_layers = nn.ModuleList([
            GCNConv(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        
        # Attention mechanism
        self.attention_weights = nn.ModuleList([
            nn.Linear(hidden_dim, 1) for _ in range(num_timesteps)
        ])
        
        # Output layers
        self.output_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
    def forward(self, data: Data) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            data: PyTorch Geometric Data object
            
        Returns:
            Predicted LogS values
        """
        x, edge_index, edge_attr, batch = (
            data.x, data.edge_index, data.edge_attr, data.batch
        )
        
        # Embed node and edge features
        x = self.node_embedding(x)
        edge_attr = self.edge_embedding(edge_attr)
        
        # Apply GCN layers
        for i, gcn_layer in enumerate(self.gcn_layers):
            x = gcn_layer(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Attentive fingerprinting
        # Initialize with node features
        h = x
        
        # Graph-level embedding for attention
        graph_emb = None
        
        for t in range(self.num_timesteps):
            # Compute attention weights
            if graph_emb is not None:
                # Use previous graph embedding for attention
                attn_input = h + graph_emb[batch]
            else:
                attn_input = h
            
            attn_logits = self.attention_weights[t](attn_input)
            attn_weights = F.softmax(attn_logits, dim=0)
            
            # Weighted aggregation
            weighted_h = h * attn_weights
            graph_emb = global_add_pool(weighted_h, batch)
            
            # Update node features for next timestep
            if t < self.num_timesteps - 1:
                h = h + graph_emb[batch]
        
        # Final prediction from graph embedding
        output = self.output_layers(graph_emb)
        
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
        from utils.molecular import smiles_to_graph
        
        graph = smiles_to_graph(smiles, device=device)
        if graph is None:
            return None
        
        self.eval()
        with torch.no_grad():
            graph.batch = torch.zeros(graph.x.size(0), dtype=torch.long, device=device)
            prediction = self.forward(graph)
            return prediction.item()

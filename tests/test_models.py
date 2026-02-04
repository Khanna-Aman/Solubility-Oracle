"""
Tests for model implementations
"""

import pytest
import torch
from torch_geometric.data import Data

from models.attentivefp import AttentiveFP
from models.hybrid import HybridModel
from models.ensemble import SolubilityEnsemble


def test_attentivefp_creation():
    """Test AttentiveFP model creation"""
    model = AttentiveFP(
        num_node_features=7,
        num_edge_features=3,
        hidden_dim=64,
        num_layers=2,
        dropout=0.2
    )
    assert model is not None
    assert isinstance(model, torch.nn.Module)


def test_attentivefp_forward():
    """Test AttentiveFP forward pass"""
    model = AttentiveFP(
        num_node_features=7,
        num_edge_features=3,
        hidden_dim=64,
        num_layers=2
    )
    
    # Create dummy graph
    x = torch.randn(10, 7)  # 10 atoms, 7 features
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    edge_attr = torch.randn(3, 3)  # 3 edges, 3 features
    batch = torch.zeros(10, dtype=torch.long)
    
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)
    
    output = model(data)
    assert output.shape == (1, 1)  # 1 graph, 1 output


def test_hybrid_model_creation():
    """Test HybridModel creation"""
    model = HybridModel(
        num_node_features=7,
        num_edge_features=3,
        descriptor_dim=200,
        gnn_hidden_dim=64,
        gnn_num_layers=2
    )
    assert model is not None
    assert isinstance(model, torch.nn.Module)


def test_hybrid_model_forward():
    """Test HybridModel forward pass"""
    model = HybridModel(
        num_node_features=7,
        num_edge_features=3,
        descriptor_dim=200,
        gnn_hidden_dim=64,
        gnn_num_layers=2
    )
    
    # Create dummy graph
    x = torch.randn(10, 7)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    edge_attr = torch.randn(3, 3)
    batch = torch.zeros(10, dtype=torch.long)
    
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)
    
    # Create dummy descriptors
    descriptors = torch.randn(1, 200)
    
    output = model(data, descriptors)
    assert output.shape == (1, 1)


def test_ensemble_creation():
    """Test SolubilityEnsemble creation"""
    model1 = AttentiveFP(num_node_features=7, num_edge_features=3, hidden_dim=64)
    model2 = AttentiveFP(num_node_features=7, num_edge_features=3, hidden_dim=64)
    
    ensemble = SolubilityEnsemble([model1, model2])
    assert ensemble is not None
    assert len(ensemble.models) == 2


def test_model_parameters():
    """Test that models have trainable parameters"""
    model = HybridModel(
        num_node_features=7,
        num_edge_features=3,
        descriptor_dim=200,
        gnn_hidden_dim=64,
        gnn_num_layers=2
    )
    
    num_params = sum(p.numel() for p in model.parameters())
    assert num_params > 0
    
    # Check that parameters require gradients
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert trainable_params == num_params


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


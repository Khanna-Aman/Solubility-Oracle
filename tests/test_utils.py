"""
Tests for utility functions
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path for imports
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.utils.molecular import smiles_to_graph, get_molecular_descriptors, validate_smiles


def test_validate_smiles_valid():
    """Test SMILES validation with valid molecules"""
    assert validate_smiles("CCO") == True  # Ethanol
    assert validate_smiles("CC(=O)OC1=CC=CC=C1C(=O)O") == True  # Aspirin
    assert validate_smiles("c1ccccc1") == True  # Benzene


def test_validate_smiles_invalid():
    """Test SMILES validation with invalid molecules"""
    assert validate_smiles("") == False
    assert validate_smiles("INVALID") == False
    assert validate_smiles("C(C(C") == False  # Unbalanced parentheses


def test_smiles_to_graph_valid():
    """Test graph conversion with valid SMILES"""
    graph = smiles_to_graph("CCO")  # Ethanol
    
    assert graph is not None
    assert graph.x.shape[0] == 3  # 3 atoms (C, C, O)
    assert graph.edge_index.shape[1] > 0  # Has edges
    assert graph.edge_attr.shape[0] > 0  # Has edge features


def test_smiles_to_graph_invalid():
    """Test graph conversion with invalid SMILES"""
    graph = smiles_to_graph("INVALID")
    assert graph is None


def test_smiles_to_graph_features():
    """Test that graph has correct feature dimensions"""
    graph = smiles_to_graph("CC(=O)OC1=CC=CC=C1C(=O)O")  # Aspirin
    
    assert graph is not None
    assert graph.x.shape[1] == 7  # 7 atom features
    assert graph.edge_attr.shape[1] == 3  # 3 edge features


def test_get_molecular_descriptors_valid():
    """Test descriptor calculation with valid SMILES"""
    descriptors = get_molecular_descriptors("CCO")  # Ethanol
    
    assert descriptors is not None
    assert isinstance(descriptors, np.ndarray)
    assert len(descriptors) > 0
    assert not np.any(np.isnan(descriptors))  # No NaN values


def test_get_molecular_descriptors_invalid():
    """Test descriptor calculation with invalid SMILES"""
    descriptors = get_molecular_descriptors("INVALID")
    assert descriptors is None


def test_get_molecular_descriptors_aspirin():
    """Test descriptor calculation for aspirin"""
    descriptors = get_molecular_descriptors("CC(=O)OC1=CC=CC=C1C(=O)O")
    
    assert descriptors is not None
    assert len(descriptors) > 10  # Should have multiple descriptors
    
    # Check that basic descriptors are reasonable
    # (First few are RDKit descriptors: MW, LogP, MR, TPSA, etc.)
    assert descriptors[0] > 0  # Molecular weight should be positive


def test_descriptors_consistency():
    """Test that descriptors are consistent across calls"""
    smiles = "c1ccccc1"  # Benzene
    
    desc1 = get_molecular_descriptors(smiles)
    desc2 = get_molecular_descriptors(smiles)
    
    assert desc1 is not None
    assert desc2 is not None
    assert np.allclose(desc1, desc2)  # Should be identical


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


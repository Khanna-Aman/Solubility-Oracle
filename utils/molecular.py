"""
Molecular processing utilities for graph construction and descriptor calculation
"""

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import Descriptors
from mordred import Calculator, descriptors
from torch_geometric.data import Data
from typing import Optional, Tuple


def smiles_to_graph(smiles: str, device: str = 'cpu') -> Optional[Data]:
    """
    Convert SMILES string to PyTorch Geometric graph representation.
    
    Args:
        smiles: SMILES string of the molecule
        device: Device to place tensors on
        
    Returns:
        PyTorch Geometric Data object or None if invalid SMILES
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # Get atom features (atomic number, degree, formal charge, etc.)
        atom_features = []
        for atom in mol.GetAtoms():
            features = [
                atom.GetAtomicNum(),
                atom.GetDegree(),
                atom.GetFormalCharge(),
                int(atom.GetHybridization()),
                int(atom.GetIsAromatic()),
                atom.GetNumRadicalElectrons(),
                atom.GetNumImplicitHs(),
            ]
            atom_features.append(features)
        
        atom_features = np.array(atom_features, dtype=np.float32)
        
        # Get edge indices and edge features
        edge_indices = []
        edge_features = []
        
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            
            # Add both directions (undirected graph)
            edge_indices.extend([[i, j], [j, i]])
            
            bond_features = [
                int(bond.GetBondType()),
                int(bond.GetIsAromatic()),
                int(bond.IsInRing()),
            ]
            edge_features.extend([bond_features, bond_features])
        
        if len(edge_indices) == 0:
            # Single atom molecule
            edge_indices = [[0, 0]]
            edge_features = [[0, 0, 0]]
        
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_features, dtype=torch.float32)
        x = torch.tensor(atom_features, dtype=torch.float32)
        
        # Create graph data object
        graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        graph = graph.to(device)
        
        return graph
    
    except Exception as e:
        print(f"Error processing SMILES {smiles}: {e}")
        return None


def get_molecular_descriptors(smiles: str) -> Optional[np.ndarray]:
    """
    Calculate molecular descriptors using RDKit and Mordred.
    
    Args:
        smiles: SMILES string of the molecule
        
    Returns:
        Array of molecular descriptors or None if invalid SMILES
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # Basic RDKit descriptors
        rdkit_descriptors = [
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            Descriptors.MolMR(mol),
            Descriptors.TPSA(mol),
            Descriptors.NumHDonors(mol),
            Descriptors.NumHAcceptors(mol),
            Descriptors.NumRotatableBonds(mol),
            Descriptors.NumAromaticRings(mol),
            Descriptors.NumSaturatedRings(mol),
        ]

        # Try to add FractionCsp3 if available (different RDKit versions)
        try:
            from rdkit.Chem import Lipinski
            rdkit_descriptors.append(Lipinski.FractionCsp3(mol))
        except (ImportError, AttributeError):
            rdkit_descriptors.append(0.0)  # Fallback
        
        # Mordred descriptors (subset for speed)
        calc = Calculator(descriptors, ignore_3D=True)
        mordred_descriptors = calc(mol)
        
        # Convert to array, handling errors
        mordred_array = []
        for desc in mordred_descriptors.values():
            try:
                val = float(desc) if desc is not None else 0.0
                if np.isnan(val) or np.isinf(val):
                    val = 0.0
                mordred_array.append(val)
            except (ValueError, TypeError):
                mordred_array.append(0.0)
        
        # Combine descriptors
        all_descriptors = np.array(rdkit_descriptors + mordred_array, dtype=np.float32)
        
        return all_descriptors
    
    except Exception as e:
        print(f"Error calculating descriptors for {smiles}: {e}")
        return None


def validate_smiles(smiles: str) -> bool:
    """
    Validate SMILES string.
    
    Args:
        smiles: SMILES string to validate
        
    Returns:
        True if valid, False otherwise
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except:
        return False

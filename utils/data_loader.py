"""
Data loading and preprocessing utilities
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
from sklearn.model_selection import train_test_split
from rdkit import Chem
from rdkit.Chem import Scaffolds


def load_dataset(filepath: str) -> pd.DataFrame:
    """
    Load dataset from CSV file.
    
    Expected columns: SMILES, LogS (or similar)
    
    Args:
        filepath: Path to CSV file
        
    Returns:
        DataFrame with SMILES and solubility data
    """
    df = pd.read_csv(filepath)
    return df


def scaffold_split(df: pd.DataFrame, test_size: float = 0.1, val_size: float = 0.1) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split dataset using molecular scaffolds to prevent data leakage.
    Falls back to random split if too few scaffolds.

    Args:
        df: DataFrame with SMILES column
        test_size: Fraction for test set
        val_size: Fraction for validation set

    Returns:
        train_df, val_df, test_df
    """
    def get_scaffold(smiles: str) -> str:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return "invalid"
            scaffold = Scaffolds.MurckoScaffold.GetScaffoldForMol(mol)
            return Chem.MolToSmiles(scaffold) if scaffold else "invalid"
        except:
            return "invalid"

    df = df.copy()
    df['scaffold'] = df['SMILES'].apply(get_scaffold)
    unique_scaffolds = df['scaffold'].unique()

    # Check if we have enough scaffolds for splitting
    min_scaffolds_needed = 10  # Minimum number of scaffolds for scaffold split

    if len(unique_scaffolds) < min_scaffolds_needed:
        print(f"Warning: Only {len(unique_scaffolds)} unique scaffolds found. Using random split instead.")
        # Fall back to random split
        train_df, temp_df = train_test_split(df, test_size=(test_size + val_size), random_state=42)
        val_df, test_df = train_test_split(temp_df, test_size=test_size/(test_size + val_size), random_state=42)
    else:
        # Split scaffolds
        train_scaffolds, temp_scaffolds = train_test_split(
            unique_scaffolds, test_size=(test_size + val_size), random_state=42
        )
        val_scaffolds, test_scaffolds = train_test_split(
            temp_scaffolds, test_size=test_size/(test_size + val_size), random_state=42
        )

        train_df = df[df['scaffold'].isin(train_scaffolds)]
        val_df = df[df['scaffold'].isin(val_scaffolds)]
        test_df = df[df['scaffold'].isin(test_scaffolds)]

    # Drop scaffold column before returning
    train_df = train_df.drop(columns=['scaffold'])
    val_df = val_df.drop(columns=['scaffold'])
    test_df = test_df.drop(columns=['scaffold'])

    return train_df, val_df, test_df


def preprocess_data(df: pd.DataFrame, target_col: str = 'LogS') -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Preprocess dataset for training.
    
    Args:
        df: DataFrame with SMILES and target column
        target_col: Name of target column
        
    Returns:
        Processed DataFrame and target array
    """
    # Remove invalid SMILES
    from utils.molecular import validate_smiles
    df = df[df['SMILES'].apply(validate_smiles)].copy()
    
    # Remove missing targets
    df = df.dropna(subset=[target_col])
    
    targets = df[target_col].values
    
    return df, targets

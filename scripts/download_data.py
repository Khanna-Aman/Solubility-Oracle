"""
Download and prepare solubility datasets
"""

import os
import pandas as pd
import requests
from pathlib import Path

# Create directories
DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def download_esol():
    """Download ESOL (Delaney) solubility dataset"""
    print("Downloading ESOL dataset...")
    
    # ESOL dataset from MoleculeNet
    url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv"
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # Save raw data
        output_path = RAW_DIR / "esol.csv"
        with open(output_path, 'wb') as f:
            f.write(response.content)
        
        print(f"✓ Downloaded ESOL dataset to {output_path}")
        
        # Load and inspect
        df = pd.read_csv(output_path)
        print(f"  - {len(df)} compounds")
        print(f"  - Columns: {list(df.columns)}")
        
        return df
    
    except Exception as e:
        print(f"✗ Error downloading ESOL: {e}")
        return None


def download_aqsoldb():
    """Download AqSolDB dataset"""
    print("\nDownloading AqSolDB dataset...")
    
    # AqSolDB from GitHub
    url = "https://raw.githubusercontent.com/PatWalters/solubility/master/data/curated-solubility-dataset.csv"
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # Save raw data
        output_path = RAW_DIR / "aqsoldb.csv"
        with open(output_path, 'wb') as f:
            f.write(response.content)
        
        print(f"✓ Downloaded AqSolDB to {output_path}")
        
        # Load and inspect
        df = pd.read_csv(output_path)
        print(f"  - {len(df)} compounds")
        print(f"  - Columns: {list(df.columns)}")
        
        return df
    
    except Exception as e:
        print(f"✗ Error downloading AqSolDB: {e}")
        print("  Note: This dataset may not be available at this URL")
        return None


def prepare_combined_dataset():
    """Combine and prepare datasets"""
    print("\nPreparing combined dataset...")
    
    datasets = []
    
    # Load ESOL
    esol_path = RAW_DIR / "esol.csv"
    if esol_path.exists():
        df_esol = pd.read_csv(esol_path)
        # Standardize column names
        if 'smiles' in df_esol.columns:
            df_esol = df_esol.rename(columns={'smiles': 'SMILES'})
        if 'measured log solubility in mols per litre' in df_esol.columns:
            df_esol = df_esol.rename(columns={'measured log solubility in mols per litre': 'LogS'})
        
        df_esol = df_esol[['SMILES', 'LogS']].copy()
        df_esol['source'] = 'ESOL'
        datasets.append(df_esol)
        print(f"  - Loaded {len(df_esol)} compounds from ESOL")
    
    # Load AqSolDB if available
    aqsoldb_path = RAW_DIR / "aqsoldb.csv"
    if aqsoldb_path.exists():
        df_aqsol = pd.read_csv(aqsoldb_path)
        # Standardize column names (adjust based on actual columns)
        if 'SMILES' in df_aqsol.columns and 'Solubility' in df_aqsol.columns:
            df_aqsol = df_aqsol[['SMILES', 'Solubility']].copy()
            df_aqsol = df_aqsol.rename(columns={'Solubility': 'LogS'})
            df_aqsol['source'] = 'AqSolDB'
            datasets.append(df_aqsol)
            print(f"  - Loaded {len(df_aqsol)} compounds from AqSolDB")
    
    if not datasets:
        print("✗ No datasets available!")
        return None
    
    # Combine datasets
    df_combined = pd.concat(datasets, ignore_index=True)
    
    # Remove duplicates (keep first occurrence)
    df_combined = df_combined.drop_duplicates(subset=['SMILES'], keep='first')
    
    # Remove missing values
    df_combined = df_combined.dropna(subset=['SMILES', 'LogS'])
    
    # Save combined dataset
    output_path = PROCESSED_DIR / "solubility_combined.csv"
    df_combined.to_csv(output_path, index=False)
    
    print(f"\n✓ Combined dataset saved to {output_path}")
    print(f"  - Total: {len(df_combined)} unique compounds")
    print(f"  - LogS range: [{df_combined['LogS'].min():.2f}, {df_combined['LogS'].max():.2f}]")
    print(f"  - LogS mean: {df_combined['LogS'].mean():.2f} ± {df_combined['LogS'].std():.2f}")
    
    return df_combined


if __name__ == "__main__":
    print("=" * 60)
    print("Solubility Oracle - Data Download Script")
    print("=" * 60)
    print()
    
    # Download datasets
    download_esol()
    download_aqsoldb()
    
    # Prepare combined dataset
    prepare_combined_dataset()
    
    print("\n" + "=" * 60)
    print("Download complete!")
    print("=" * 60)


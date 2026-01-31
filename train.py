"""
Training script for Solubility Oracle models
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch
import numpy as np
from tqdm import tqdm
import os
from pathlib import Path

from models.attentivefp import AttentiveFP
from models.hybrid import HybridModel
from models.ensemble import SolubilityEnsemble
from utils.molecular import smiles_to_graph, get_molecular_descriptors
from utils.data_loader import load_dataset, preprocess_data, scaffold_split


class SolubilityDataset(Dataset):
    """Dataset for solubility prediction"""
    
    def __init__(self, df, target_col='LogS', device='cpu'):
        self.df = df.reset_index(drop=True)
        self.target_col = target_col
        self.device = device
        
        # Filter valid molecules
        from utils.molecular import validate_smiles
        valid_mask = self.df['SMILES'].apply(validate_smiles)
        self.df = self.df[valid_mask].reset_index(drop=True)
        
        print(f"Loaded {len(self.df)} valid molecules")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        smiles = self.df.iloc[idx]['SMILES']
        target = float(self.df.iloc[idx][self.target_col])
        
        graph = smiles_to_graph(smiles, device=self.device)
        descriptors = get_molecular_descriptors(smiles)
        
        if descriptors is None:
            descriptors = np.zeros(200, dtype=np.float32)
        else:
            if len(descriptors) < 200:
                descriptors = np.pad(descriptors, (0, 200 - len(descriptors)))
            else:
                descriptors = descriptors[:200]
        
        return {
            'graph': graph,
            'descriptors': torch.tensor(descriptors, dtype=torch.float32),
            'target': torch.tensor(target, dtype=torch.float32),
            'smiles': smiles
        }


def collate_fn(batch):
    """Collate function for DataLoader"""
    graphs = [item['graph'] for item in batch if item['graph'] is not None]
    descriptors = torch.stack([item['descriptors'] for item in batch])
    targets = torch.stack([item['target'] for item in batch])
    smiles = [item['smiles'] for item in batch]
    
    if len(graphs) == 0:
        return None
    
    batch_graph = Batch.from_data_list(graphs)
    
    return {
        'graph': batch_graph,
        'descriptors': descriptors,
        'target': targets,
        'smiles': smiles
    }


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        if batch is None:
            continue
        
        optimizer.zero_grad()
        
        graph = batch['graph'].to(device)
        descriptors = batch['descriptors'].to(device)
        targets = batch['target'].to(device).unsqueeze(1)
        
        if isinstance(model, HybridModel):
            predictions = model(graph, descriptors)
        else:
            predictions = model(graph)
        
        loss = criterion(predictions, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0
    predictions_list = []
    targets_list = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            if batch is None:
                continue
            
            graph = batch['graph'].to(device)
            descriptors = batch['descriptors'].to(device)
            targets = batch['target'].to(device).unsqueeze(1)
            
            if isinstance(model, HybridModel):
                predictions = model(graph, descriptors)
            else:
                predictions = model(graph)
            
            loss = criterion(predictions, targets)
            total_loss += loss.item()
            
            predictions_list.append(predictions.cpu().numpy())
            targets_list.append(targets.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    
    # Calculate metrics
    all_preds = np.concatenate(predictions_list)
    all_targets = np.concatenate(targets_list)
    
    mae = np.mean(np.abs(all_preds - all_targets))
    rmse = np.sqrt(np.mean((all_preds - all_targets) ** 2))
    
    return avg_loss, mae, rmse


def main():
    """Main training function"""
    # Configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Data loading (placeholder - user should provide actual data)
    print("Note: This is a template training script.")
    print("Please provide your dataset in CSV format with 'SMILES' and 'LogS' columns.")
    print("\nExample usage:")
    print("  python train.py --data_path data/raw/solubility_data.csv")
    
    # Create checkpoint directory
    os.makedirs('checkpoints', exist_ok=True)
    
    # Model configuration
    model_config = {
        'num_node_features': 7,
        'num_edge_features': 3,
        'descriptor_dim': 200,
        'gnn_hidden_dim': 128,
        'gnn_num_layers': 3,
        'fusion_hidden_dim': 256,
        'dropout': 0.2,
        'output_dim': 1
    }
    
    # Create model
    model = HybridModel(**model_config).to(device)
    
    print(f"\nModel created with {sum(p.numel() for p in model.parameters())} parameters")
    print("Training script ready. Add your dataset to begin training.")


if __name__ == "__main__":
    main()

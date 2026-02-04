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
    import argparse

    parser = argparse.ArgumentParser(description='Train Solubility Oracle model')
    parser.add_argument('--data_path', type=str, default='data/processed/solubility_combined.csv',
                        help='Path to dataset CSV file')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension')
    parser.add_argument('--num_layers', type=int, default=3, help='Number of GNN layers')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Checkpoint directory')

    args = parser.parse_args()

    # Configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("=" * 60)
    print("Solubility Oracle - Training Script")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Dataset: {args.data_path}")
    print()

    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Load dataset
    if not os.path.exists(args.data_path):
        print(f"Error: Dataset not found at {args.data_path}")
        print("Please run: python scripts/download_data.py")
        return

    print("Loading dataset...")
    df = load_dataset(args.data_path)
    print(f"Loaded {len(df)} compounds")

    # Preprocess data
    df, targets = preprocess_data(df, target_col='LogS')
    print(f"Valid compounds: {len(df)}")

    # Split dataset using scaffold split
    print("\nSplitting dataset (scaffold split)...")
    train_df, val_df, test_df = scaffold_split(df, test_size=0.1, val_size=0.1)
    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    # Create datasets
    train_dataset = SolubilityDataset(train_df, device=device)
    val_dataset = SolubilityDataset(val_df, device=device)
    test_dataset = SolubilityDataset(test_df, device=device)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # Model configuration
    model_config = {
        'num_node_features': 7,
        'num_edge_features': 3,
        'descriptor_dim': 200,
        'gnn_hidden_dim': args.hidden_dim,
        'gnn_num_layers': args.num_layers,
        'fusion_hidden_dim': 256,
        'dropout': args.dropout,
        'output_dim': 1
    }

    # Create model
    model = HybridModel(**model_config).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: HybridModel")
    print(f"Parameters: {num_params:,}")

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    # Training loop
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)

    best_val_loss = float('inf')
    patience_counter = 0
    max_patience = 20

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")

        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss, val_mae, val_rmse = validate(model, val_loader, criterion, device)

        # Update scheduler
        scheduler.step(val_loss)

        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val MAE: {val_mae:.4f} | Val RMSE: {val_rmse:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            checkpoint_path = os.path.join(args.checkpoint_dir, 'best_model.pt')

            # Save as ensemble format for compatibility
            ensemble = SolubilityEnsemble([model])
            ensemble.save(checkpoint_path)

            print(f"✓ Saved best model (val_loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                print(f"\nEarly stopping after {epoch+1} epochs")
                break

    # Test evaluation
    print("\n" + "=" * 60)
    print("Evaluating on test set...")
    print("=" * 60)

    test_loss, test_mae, test_rmse = validate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test MAE: {test_mae:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")

    # Calculate R²
    predictions_list = []
    targets_list = []

    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            if batch is None:
                continue

            graph = batch['graph'].to(device)
            descriptors = batch['descriptors'].to(device)
            targets = batch['target'].to(device).unsqueeze(1)

            predictions = model(graph, descriptors)

            predictions_list.append(predictions.cpu().numpy())
            targets_list.append(targets.cpu().numpy())

    all_preds = np.concatenate(predictions_list)
    all_targets = np.concatenate(targets_list)

    # R² score
    ss_res = np.sum((all_targets - all_preds) ** 2)
    ss_tot = np.sum((all_targets - np.mean(all_targets)) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    print(f"Test R²: {r2:.4f}")

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    print(f"Best model saved to: {os.path.join(args.checkpoint_dir, 'best_model.pt')}")


if __name__ == "__main__":
    main()

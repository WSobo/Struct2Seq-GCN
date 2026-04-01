import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import argparse
import torch
import torch.nn as nn
from torch.optim import Adam
from torch_geometric.loader import DataLoader
from utils.dataset import Struct2SeqDataset
from utils.model_utils import Struct2SeqGCN

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total_samples = 0
    
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        logits = model(batch)
        
        # We only want to compute loss on the regions we care about
        # chain_M comes natively from LigandMPNN (1 for residues to predict, 0 otherwise)
        mask = batch.chain_M.bool() if hasattr(batch, 'chain_M') else torch.ones_like(batch.y).bool()
        
        if mask.sum() == 0:
            continue
            
        # Filter logits and targets by mask
        masked_logits = logits[mask]
        masked_targets = batch.y[mask]
        
        # Compute loss
        loss = criterion(masked_logits, masked_targets)
        loss.backward()
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        
        preds = masked_logits.argmax(dim=-1)
        correct += (preds == masked_targets).sum().item()
        total_samples += mask.sum().item()
        
    return total_loss / len(loader), correct / total_samples

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            
            mask = batch.chain_M.bool() if hasattr(batch, 'chain_M') else torch.ones_like(batch.y).bool()
            if mask.sum() == 0:
                continue
                
            masked_logits = logits[mask]
            masked_targets = batch.y[mask]
            
            loss = criterion(masked_logits, masked_targets)
            total_loss += loss.item()
            
            preds = masked_logits.argmax(dim=-1)
            correct += (preds == masked_targets).sum().item()
            total_samples += mask.sum().item()
            
    return total_loss / len(loader), correct / total_samples

def main():
    parser = argparse.ArgumentParser(description="Train Struct2Seq-GCN")
    parser.add_argument("--json_train", type=str, default="LigandMPNN/training/train.json", help="Path to train JSON")
    parser.add_argument("--json_valid", type=str, default="LigandMPNN/training/valid.json", help="Path to valid JSON")
    parser.add_argument("--pdb_dir", type=str, default="LigandMPNN/inputs/", help="Directory containing raw PDBs")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out_dir", type=str, default="outputs/")
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Dataset & Dataloaders
    print("Initializing datasets... (This will process raw PDBs into PyG graphs)")
    train_dataset = Struct2SeqDataset(root="training/train_data", json_file=args.json_train, pdb_dir=args.pdb_dir)
    valid_dataset = Struct2SeqDataset(root="training/valid_data", json_file=args.json_valid, pdb_dir=args.pdb_dir)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
    
    # 2. Model setup
    model = Struct2SeqGCN(node_features=3, hidden_dim=128, num_classes=21).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr)
    
    # Loss: cross entropy over the 21 classes (ignoring padding is already handled by masking)
    criterion = nn.CrossEntropyLoss()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    # 3. Training Loop
    print("Starting training...")
    best_val_loss = float('inf')
    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, valid_loader, criterion, device)
        
        print(f"Epoch {epoch+1:03d} | "
              f"Train Loss: {train_loss:.4f} - Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} - Acc: {val_acc:.4f}")
              
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(args.out_dir, "best_model.pt"))
            print("  -> Saved new best model!")

if __name__ == "__main__":
    main()

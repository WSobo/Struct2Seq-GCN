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
        if hasattr(batch['protein'], 'chain_M'):
            mask = batch['protein'].chain_M.bool()
        else:
            mask = torch.ones_like(batch['protein'].y).bool()
        
        if mask.sum() == 0:
            continue
            
        # Filter logits and targets by mask
        masked_logits = logits[mask]
        masked_targets = batch['protein'].y[mask]
        
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
            
            if hasattr(batch['protein'], 'chain_M'):
                mask = batch['protein'].chain_M.bool()
            else:
                mask = torch.ones_like(batch['protein'].y).bool()
                
            if mask.sum() == 0:
                continue
                
            masked_logits = logits[mask]
            masked_targets = batch['protein'].y[mask]
            
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
    parser.add_argument("--max_samples", type=int, default=None, help="Randomly sample PDBs to prevent 100GB full dataset downloads during testing")
    parser.add_argument("--out_dir", type=str, default="outputs/")
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Dataset & Dataloaders
    print("Initializing datasets... (This will process raw PDBs into PyG graphs)")
    train_dataset = Struct2SeqDataset(root="training/train_data", json_file=args.json_train, pdb_dir=args.pdb_dir, max_samples=args.max_samples)
    valid_dataset = Struct2SeqDataset(root="training/valid_data", json_file=args.json_valid, pdb_dir=args.pdb_dir, max_samples=args.max_samples // 10 if args.max_samples else None)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
    
    # 2. Model setup
    model = Struct2SeqGCN(node_features=6, hidden_dim=128, num_classes=21).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr)
    
    # Loss: cross entropy over the 21 classes (ignoring padding is already handled by masking)
    criterion = nn.CrossEntropyLoss()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    # 3. Training Loop
    print("Starting training...")
    best_val_loss = float('inf')
    early_stop_patience = 10
    early_stop_counter = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, valid_loader, criterion, device)
        
        # Log metrics to history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        
        print(f"Epoch {epoch+1:03d}/{args.epochs:03d} | "
            f"Train Loss: {train_loss:.4f} - Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} - Acc: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), os.path.join(args.out_dir, "best_model.pt"))
            print(f"  -> Saved new best model! (Val Loss: {val_loss:.4f})")
        else:
            early_stop_counter += 1
            print(f"  -> Early stopping counter: {early_stop_counter}/{early_stop_patience}")
            if early_stop_counter >= early_stop_patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs.")
                break
                
    # Save the training history for plotting later
    import json
    with open(os.path.join(args.out_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=4)
    print("\nTraining complete! History saved to outputs/training_history.json")

if __name__ == "__main__":
    main()

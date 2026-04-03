import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import argparse
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.optim import Adam
from torch_geometric.loader import DataLoader
from utils.dataset import Struct2SeqDataset
from utils.model_utils import Struct2SeqGNN
import datetime

def setup_ddp():
    """Initialize Distributed Data Parallel environment."""
    if "WORLD_SIZE" in os.environ:
        dist.init_process_group("nccl", timeout=datetime.timedelta(hours=4))
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return local_rank, dist.get_world_size()
    return 0, 1

def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()

def train_epoch(model, loader, optimizer, criterion, device, epoch, global_step, log_interval, checkpoint_interval, out_dir):
    model.train()
    total_loss = 0
    correct = 0
    total_samples = 0
    
    for step, batch in enumerate(loader):
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
        
        global_step += 1
        
        # Track metrics
        loss_val = loss.item()
        total_loss += loss_val
        
        preds = masked_logits.argmax(dim=-1)
        batch_correct = (preds == masked_targets).sum().item()
        batch_samples = mask.sum().item()
        
        correct += batch_correct
        total_samples += batch_samples
        
        # Step-level Live Logging
        if log_interval > 0 and global_step % log_interval == 0:
            batch_acc = batch_correct / batch_samples if batch_samples > 0 else 0.0
            print(f"  [Epoch {epoch+1} | Step {global_step}] Loss: {loss_val:.4f} | Acc: {batch_acc:.4f}")
            
        # Step-level Mid-Epoch Checkpointing
        if checkpoint_interval > 0 and global_step % checkpoint_interval == 0:
            ckpt_path = os.path.join(out_dir, f"checkpoint_step_{global_step}.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"  -> Saved step checkpoint: {ckpt_path}")
        
    return total_loss / len(loader), correct / total_samples, global_step

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
    parser = argparse.ArgumentParser(description="Train Struct2Seq-GNN")
    parser.add_argument("--json_train", type=str, default="LigandMPNN/training/train.json", help="Path to train JSON")
    parser.add_argument("--json_valid", type=str, default="LigandMPNN/training/valid.json", help="Path to valid JSON")
    parser.add_argument("--pdb_dir", type=str, default="LigandMPNN/inputs/", help="Directory containing raw PDBs")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max_samples", type=int, default=None, help="Randomly sample PDBs to prevent 100GB full dataset downloads during testing")
    parser.add_argument("--out_dir", type=str, default="outputs/")
    
    # Model scaling limits
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden dimensions for nodes and edges (Scale up for 150k dataset)")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of graph convolution layers (Scale up for 150k dataset)")
    
    # Advanced HPC / MLOps parameters
    parser.add_argument("--num_workers", type=int, default=4, help="CPU workers for data prefetching to prevent GPU starvation")
    parser.add_argument("--pin_memory", action="store_true", help="Pin memory for faster host-to-GPU data transfers")
    parser.add_argument("--log_interval", type=int, default=100, help="Print live training loss every N batches")
    parser.add_argument("--checkpoint_interval", type=int, default=5000, help="Save mid-epoch checkpoints every N batches")
    
    args = parser.parse_args()
    
    local_rank, world_size = setup_ddp()
    is_main_process = local_rank == 0
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    
    if is_main_process:
        print(f"Using device: {device} | World Size: {world_size}")
    
    # 1. Dataset & Dataloaders
    if not is_main_process and dist.is_initialized():
        dist.barrier()  # Wait here to ensure rank 0 safely verifies the cached data

    if is_main_process:
        print("Loading cached PyG datasets... (Assuming preprocessing was completed by CPU job)")
        
    train_dataset = Struct2SeqDataset(root="training/train_data", json_file=args.json_train, pdb_dir=args.pdb_dir, max_samples=args.max_samples)
    valid_dataset = Struct2SeqDataset(root="training/valid_data", json_file=args.json_valid, pdb_dir=args.pdb_dir, max_samples=args.max_samples // 10 if args.max_samples else None)
    
    if is_main_process and dist.is_initialized():
        dist.barrier()  # Release other ranks since local processing/caching is completely finished
    
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=local_rank, shuffle=True) if dist.is_initialized() else None
    valid_sampler = DistributedSampler(valid_dataset, num_replicas=world_size, rank=local_rank, shuffle=False) if dist.is_initialized() else None

    # Adjust batch size strictly to per-GPU batch size
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=(train_sampler is None), 
        sampler=train_sampler,
        num_workers=args.num_workers, 
        pin_memory=args.pin_memory,
        persistent_workers=True if args.num_workers > 0 else False
    )
    valid_loader = DataLoader(
        valid_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        sampler=valid_sampler,
        num_workers=args.num_workers, 
        pin_memory=args.pin_memory,
        persistent_workers=True if args.num_workers > 0 else False
    )
    
    # 2. Model setup
    model = Struct2SeqGNN(node_features=6, ligand_features=6, hidden_dim=args.hidden_dim, num_classes=21, num_layers=args.num_layers, dropout=0.1).to(device)
    
    if dist.is_initialized():
        # Using DistributedDataParallel
        model = DDP(model, device_ids=[local_rank])
        
    optimizer = Adam(model.parameters(), lr=args.lr)
    
    # Loss: cross entropy over the 21 classes (ignoring padding is already handled by masking)
    criterion = nn.CrossEntropyLoss()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    # 3. Training Loop
    if is_main_process:
        print("Starting training...")
    best_val_loss = float('inf')
    early_stop_patience = 10
    early_stop_counter = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    global_step = 0

    for epoch in range(args.epochs):
        if hasattr(train_loader, 'sampler') and hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
            
        train_loss, train_acc, global_step = train_epoch(
            model, train_loader, optimizer, criterion, device, 
            epoch, global_step, args.log_interval if is_main_process else 0, args.checkpoint_interval if is_main_process else 0, args.out_dir
        )
        val_loss, val_acc = evaluate(model, valid_loader, criterion, device)
        
        if dist.is_initialized():
            # Gather metrics across GPUs
            val_loss_tensor = torch.tensor(val_loss, device=device)
            val_acc_tensor = torch.tensor(val_acc, device=device)
            dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(val_acc_tensor, op=dist.ReduceOp.SUM)
            val_loss = (val_loss_tensor / world_size).item()
            val_acc = (val_acc_tensor / world_size).item()
            
        if is_main_process:
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
                
                # Unwrap model properly when saving
                model_to_save = model.module if isinstance(model, DDP) else model
                torch.save(model_to_save.state_dict(), os.path.join(args.out_dir, "best_model.pt"))
                print(f"  -> Saved new best model! (Val Loss: {val_loss:.4f})")
            else:
                early_stop_counter += 1
                print(f"  -> Early stopping counter: {early_stop_counter}/{early_stop_patience}")
                if early_stop_counter >= early_stop_patience:
                    print(f"\nEarly stopping triggered after {epoch+1} epochs.")
                    break
        
        # Sync the early stop across all nodes to gracefully exit
        stop_tensor = torch.tensor(early_stop_counter >= early_stop_patience, device=device, dtype=torch.int)
        if dist.is_initialized():
            dist.all_reduce(stop_tensor, op=dist.ReduceOp.MAX)
        if stop_tensor.item() > 0:
            break
                
    if is_main_process:
        # Save the training history for plotting later
        import json
        with open(os.path.join(args.out_dir, "training_history.json"), "w") as f:
            json.dump(history, f, indent=4)
        print("\nTraining complete! History saved to outputs/training_history.json")
        
    cleanup_ddp()

if __name__ == "__main__":
    main()

import sys
import os
import argparse
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.dataset import Struct2SeqDataset

def main():
    parser = argparse.ArgumentParser(description="Preprocess PDBs for Struct2Seq-GNN")
    parser.add_argument("--json_train", type=str, default="LigandMPNN/training/train.json", help="Path to train JSON")
    parser.add_argument("--json_valid", type=str, default="LigandMPNN/training/valid.json", help="Path to valid JSON")
    parser.add_argument("--pdb_dir", type=str, default="inputs/", help="Directory containing raw PDBs")
    parser.add_argument("--max_samples", type=int, default=None, help="Randomly sample PDBs")
    
    args = parser.parse_args()
    
    print("Starting full CPU preprocessing for training set...")
    train_dataset = Struct2SeqDataset(
        root="training/train_data", 
        json_file=args.json_train, 
        pdb_dir=args.pdb_dir, 
        max_samples=args.max_samples
    )
    
    print("Starting full CPU preprocessing for validation set...")
    valid_dataset = Struct2SeqDataset(
        root="training/valid_data", 
        json_file=args.json_valid, 
        pdb_dir=args.pdb_dir, 
        max_samples=args.max_samples // 10 if args.max_samples else None
    )
    
    print("Done! All .pt files cached and ready for GPU training.")

if __name__ == "__main__":
    main()

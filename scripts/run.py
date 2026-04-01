import torch
import torch.nn.functional as F
from utils.data_utils import pdb_to_pyg_data
from utils.model_utils import Struct2SeqGCN
import argparse

def main():
    parser = argparse.ArgumentParser(description="Run Struct2Seq-GCN Prediction")
    parser.add_argument("--pdb", type=str, required=True, help="Path to input PDB file")
    parser.add_argument("--radius", type=float, default=8.0, help="Distance cutoff in Angstroms")
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Data Parsing & Graph Construction
    print(f"Parsing {args.pdb}...")
    data = pdb_to_pyg_data(args.pdb, radius=args.radius).to(device)
    
    # 2. Sequence Prediction Model
    model = Struct2SeqGCN(node_features=3, hidden_dim=128, num_classes=21).to(device)
    model.eval()
    
    with torch.no_grad():
        logits = model(data)
        probabilities = F.softmax(logits, dim=-1)
        predicted_classes = torch.argmax(probabilities, dim=-1)
        
    print(f"Predicted Sequence Length: {len(predicted_classes)}")
    print("Done!")

if __name__ == "__main__":
    main()

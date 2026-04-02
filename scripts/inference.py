import os
import sys
import argparse
import torch
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.graph_builder import pdb_to_pyg_data
from utils.model_utils import Struct2SeqGCN

try:
    from LigandMPNN.data_utils import restype_int_to_str
except ImportError:
    # Fallback if LigandMPNN is not available in path
    restype_int_to_str = {
        0: 'A', 1: 'C', 2: 'D', 3: 'E', 4: 'F', 5: 'G', 6: 'H', 7: 'I', 8: 'K',
        9: 'L', 10: 'M', 11: 'N', 12: 'P', 13: 'Q', 14: 'R', 15: 'S', 16: 'T',
        17: 'V', 18: 'W', 19: 'Y', 20: 'X'
    }


def main():
    parser = argparse.ArgumentParser(description="Inference Script for Protein-Ligand Co-Design (Struct2Seq-GCN)")
    parser.add_argument("--pdb", type=str, required=True, help="Path to input PDB file")
    parser.add_argument("--weights", type=str, default="outputs/best_model.pt", help="Path to trained model weights (.pt)")
    parser.add_argument("--radius", type=float, default=8.0, help="Distance cutoff in Angstroms for graph edges")
    parser.add_argument("--out_fasta", type=str, default=None, help="Path to save the predicted sequence as FASTA")
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Parse Data
    print(f"Parsing coordinates from {args.pdb}...")
    data = pdb_to_pyg_data(args.pdb, radius=args.radius).to(device)
    
    protein_nodes = data['protein'].x.shape[0]
    ligand_nodes = data['ligand'].x.shape[0] if 'ligand' in data.node_types else 0
    print(f"Graph Built: {protein_nodes} Protein Nodes, {ligand_nodes} Ligand Nodes.")

    # 2. Setup Model Architecture
    # Ensure num_classes=21 matches the vocabulary (20 std amino acids + 1 gap/unknown)
    model = Struct2SeqGCN(
        node_features=6, 
        hidden_dim=128, 
        num_classes=21
    ).to(device)
    
    # 3. Load Trained Weights
    if not os.path.exists(args.weights):
        print(f"Warning: Weights file {args.weights} not found. Running with randomly initialized weights for test purposes.")
    else:
        print(f"Loading trained weights from {args.weights}...")
        model.load_state_dict(torch.load(args.weights, map_location=device, weights_only=True))
    
    model.eval()
    
    # 4. Predict
    print("Running inference...")
    with torch.no_grad():
        # Struct2SeqGCN expects the full HeteroData object
        logits = model(data)
        
        # HeteroConv returns a dict, extract the protein node predictions
        if isinstance(logits, dict):
            logits = logits['protein']
            
        probabilities = F.softmax(logits, dim=-1)
        predicted_classes = torch.argmax(probabilities, dim=-1)
        
    # 5. Decode classes to Amino Acids & Calculate Recovery
    predicted_seq = "".join([restype_int_to_str.get(c.item(), "X") for c in predicted_classes])
    
    # Extract Native Sequence
    native_classes = data['protein'].y
    native_seq = "".join([restype_int_to_str.get(c.item(), "X") for c in native_classes])
    
    # Calculate Native Sequence Recovery (NSR)
    matches = (predicted_classes == native_classes).sum().item()
    total_residues = len(native_classes)
    nsr_percentage = (matches / total_residues) * 100.0 if total_residues > 0 else 0.0
    
    print("\n" + "="*50)
    print("INFERENCE RESULTS")
    print("="*50)
    print(f"Native Sequence    : {native_seq}")
    print(f"Predicted Sequence : {predicted_seq}")
    print("-" * 50)
    print(f"Native Sequence Recovery (NSR): {nsr_percentage:.2f}% ({matches}/{total_residues} residues)")
    print("="*50 + "\n")

    # 6. Save to FASTA (optional)
    if args.out_fasta:
        out_dir = os.path.dirname(args.out_fasta)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            
        header = os.path.basename(args.pdb).replace('.pdb', '')
        with open(args.out_fasta, "w") as f:
            f.write(f">Struct2SeqGCN_Predicted_{header}\n")
            f.write(f"{predicted_seq}\n")
        print(f"Saved FASTA to {args.out_fasta}")

if __name__ == "__main__":
    main()
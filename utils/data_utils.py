import os
import sys
import torch
from torch_geometric.data import Data
from torch_geometric.nn import radius_graph

# Make sure we can import LigandMPNN tools
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "examples", "LigandMPNN"))

try:
    from data_utils import parse_PDB, featurize
except ImportError:
    # Alternative path if it's placed differently
    sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "LigandMPNN"))
    from data_utils import parse_PDB, featurize

def get_ligandmpnn_features(pdb_path, device="cpu"):
    """
    Leverages LigandMPNN's exact parsing logic to ensure 1:1 equivalency.
    """
    protein_dict, backbone, other_atoms, icodes, _ = parse_PDB(pdb_path, device=device)
    # Featurize the same way LigandMPNN does
    # This prepares the raw dictionary into model-ready tensors
    # S, X, mask, chain_M, etc.
    feature_dict = featurize(protein_dict, cutoff_for_score=8.0)
    return feature_dict

def dict_to_pyg_data(feature_dict, radius=8.0):
    """
    Converts LigandMPNN's feature_dict into a PyTorch Geometric Data object
    so we can train our GCN on the exact same data splits.
    """
    X = feature_dict["X"].squeeze(0)  # [L, 14, 3] usually
    if X.dim() == 3 and X.size(1) >= 4:
        # N, CA, C, O => index 1 is CA
        ca_coords = X[:, 1, :] 
    else:
        ca_coords = X # Fallback
        
    sequence_labels = feature_dict["S"].squeeze(0) # [L]
    mask = feature_dict["mask"].squeeze(0).bool() # [L]
    
    # Filter out invalid residues (where mask is 0)
    ca_coords = ca_coords[mask]
    sequence_labels = sequence_labels[mask]
    
    # v1.0 spec: absolute coordinates as features (node features)
    x = ca_coords.clone().float()
    
    # Edge construct
    edge_index = radius_graph(x, r=radius, loop=False)
    
    data = Data(x=x, edge_index=edge_index, y=sequence_labels)
    
    # Pass along mask/chain variables for identical scoring downstream
    if "chain_M" in feature_dict:
        data.chain_M = feature_dict["chain_M"].squeeze(0)[mask]
    
    return data

def pdb_to_pyg_data(pdb_path, radius=8.0, device="cpu"):
    """
    Replaces the initial ProDy parser with LigandMPNN's native parser snippet.
    """
    feature_dict = get_ligandmpnn_features(pdb_path, device=device)
    data = dict_to_pyg_data(feature_dict, radius=radius)
    return data


import os
import importlib.util
import torch
from torch_geometric.data import Data
from torch_geometric.nn import radius_graph

def _load_ligandmpnn_parsers():
    """Load LigandMPNN parser functions from a concrete file path."""
    repo_root = os.path.dirname(os.path.dirname(__file__))
    candidates = [
        os.path.join(repo_root, "LigandMPNN", "data_utils.py"),
        os.path.join(repo_root, "examples", "LigandMPNN", "data_utils.py"),
    ]

    parser_file = None
    for candidate in candidates:
        if os.path.exists(candidate):
            parser_file = candidate
            break

    if parser_file is None:
        raise ImportError(
            "Could not find LigandMPNN/data_utils.py. "
            "If you cloned from GitHub, initialize submodules with: "
            "git submodule update --init --recursive"
        )

    spec = importlib.util.spec_from_file_location("ligandmpnn_data_utils", parser_file)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.parse_PDB, module.featurize


parse_PDB, featurize = _load_ligandmpnn_parsers()

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


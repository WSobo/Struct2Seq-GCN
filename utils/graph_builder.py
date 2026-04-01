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
    
    # LigandMPNN run.py adds chain_mask to designate which residues are redesigned (1) or fixed (0).
    # For baseline GCN training, we assume all parsed residues are valid design targets.
    if "chain_letters" in protein_dict:
        protein_dict["chain_mask"] = torch.ones(
            len(protein_dict["chain_letters"]), 
            dtype=torch.int32, 
            device=device
        )

    # Featurize the same way LigandMPNN does
    # This prepares the raw dictionary into model-ready tensors
    # S, X, mask, chain_M, etc.
    feature_dict = featurize(protein_dict, cutoff_for_score=8.0)
    return feature_dict

def compute_dihedrals(X):
    """
    Computes backbone dihedral angles (phi, psi, omega) for a sequence of 3D coordinates.
    X: Tensor of shape [L, atom_type, 3] where atom_type -> 0: N, 1: CA, 2: C
    Returns: Tensor of [L, 6] containing sin and cos of each angle.
    """
    N = X[:, 0, :]
    CA = X[:, 1, :]
    C = X[:, 2, :]
    
    # Pad to handle edges (i-1 and i+1)
    C_prev = torch.cat([C[0:1], C[:-1]], dim=0)
    N_next = torch.cat([N[1:], N[-1:]], dim=0)
    CA_next = torch.cat([CA[1:], CA[-1:]], dim=0)
    
    def dihedral(p0, p1, p2, p3):
        b0 = p0 - p1
        b1 = p2 - p1
        b2 = p3 - p2
        
        # Normalize b1
        b1_norm = b1 / (torch.linalg.norm(b1, dim=-1, keepdim=True) + 1e-7)
        
        n1 = torch.linalg.cross(b0, b1_norm, dim=-1)
        n2 = torch.linalg.cross(b1_norm, b2, dim=-1)
        m = torch.linalg.cross(n1, b1_norm, dim=-1)
        
        x = torch.sum(n1 * n2, dim=-1)
        y = torch.sum(m * n2, dim=-1)
        
        return torch.atan2(y, x)
        
    phi = dihedral(C_prev, N, CA, C)
    psi = dihedral(N, CA, C, N_next)
    omega = dihedral(CA, C, N_next, CA_next)
    
    # Use sin/cos to wrap correctly in Neural Networks
    dihedrals = torch.stack([phi, psi, omega], dim=-1) # [L, 3]
    return torch.cat([torch.sin(dihedrals), torch.cos(dihedrals)], dim=-1) # [L, 6]

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
    
    # Compute SE(3) invariant node features BEFORE masking to preserve i-1/i+1 structure
    # This gives us [L, 6] outputs (sin, cos of phi, psi, omega)
    dihedral_features = compute_dihedrals(X)
    
    # Filter out invalid residues (where mask is 0)
    ca_coords = ca_coords[mask]
    sequence_labels = sequence_labels[mask]
    dihedral_features = dihedral_features[mask]
    
    # v2.0 spec: SE(3) Invariant representations (Backbone dihedral angles)
    x = dihedral_features.clone().float()
    
    # Edge construct based on real coordinates
    ca_float = ca_coords.clone().float()
    edge_index = radius_graph(ca_float, r=radius, loop=False)
    
    # Calculate pairwise distances (Edge Features) to work towards SE(3) invariance
    row, col = edge_index
    distances = torch.norm(ca_float[row] - ca_float[col], dim=1, p=2).unsqueeze(-1)
    
    data = Data(x=x, edge_index=edge_index, edge_attr=distances, y=sequence_labels.long())
    
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


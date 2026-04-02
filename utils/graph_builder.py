import os
import importlib.util
import torch
from torch_geometric.data import Data, HeteroData
from torch_geometric.nn import radius_graph, radius

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
    feature_dict = featurize(protein_dict, cutoff_for_score=8.0)
    
    # We explicitly inject the raw ligand atoms extracted by parse_PDB
    # so we can build them as separate nodes in a Heterogeneous Graph.
    feature_dict["ligand_Y"] = protein_dict.get("Y", None)
    feature_dict["ligand_Y_t"] = protein_dict.get("Y_t", None)
    feature_dict["ligand_Y_m"] = protein_dict.get("Y_m", None)
    
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

def encode_ligand_elements(element_ids):
    """
    Converts PyTorch integer elemental IDs into a [M, 6] one-hot tensor.
    LigandMPNN maps elements functionally to their atomic numbers:
    Carbon=6, Nitrogen=7, Oxygen=8, Phosphorus=15, Sulfur=16.
    
    We map these to 6 biological bins for the network:
    0: Carbon
    1: Nitrogen
    2: Oxygen
    3: Sulfur
    4: Phosphorus
    5: Other / Halogens
    """
    M = element_ids.shape[0]
    one_hot = torch.zeros((M, 6), dtype=torch.float32, device=element_ids.device)
    
    mask_C = (element_ids == 6)
    mask_N = (element_ids == 7)
    mask_O = (element_ids == 8)
    mask_S = (element_ids == 16)
    mask_P = (element_ids == 15)
    
    one_hot[mask_C, 0] = 1.0
    one_hot[mask_N, 1] = 1.0
    one_hot[mask_O, 2] = 1.0
    one_hot[mask_S, 3] = 1.0
    one_hot[mask_P, 4] = 1.0
    
    mask_other = ~(mask_C | mask_N | mask_O | mask_S | mask_P)
    one_hot[mask_other, 5] = 1.0
    
    return one_hot

def dict_to_pyg_data(feature_dict, radius_cutoff=8.0):
    """
    Converts LigandMPNN's feature_dict into a PyTorch Geometric HeteroData object
    capable of processing distinct 'protein' and 'ligand' nodes symmetrically.
    """
    data = HeteroData()

    # ==================================================
    # 1. Build Protein Nodes
    # ==================================================
    X = feature_dict["X"].squeeze(0)  # [L, 14, 3] usually
    if X.dim() == 3 and X.size(1) >= 4:
        ca_coords = X[:, 1, :]
    else:
        ca_coords = X
        
    sequence_labels = feature_dict["S"].squeeze(0)
    mask = feature_dict["mask"].squeeze(0).bool()
    
    dihedral_features = compute_dihedrals(X)
    
    ca_coords = ca_coords[mask]
    sequence_labels = sequence_labels[mask]
    dihedral_features = dihedral_features[mask]
    
    data['protein'].x = dihedral_features.clone().float()
    data['protein'].pos = ca_coords.clone().float()
    data['protein'].y = sequence_labels.long()
    
    if "chain_M" in feature_dict:
        data['protein'].chain_M = feature_dict["chain_M"].squeeze(0)[mask]
    
    # Protein -> Protein Edges
    p_pos = data['protein'].pos
    pp_edge_index = radius_graph(p_pos, r=radius_cutoff, loop=False)
    p_row, p_col = pp_edge_index
    pp_dist = torch.norm(p_pos[p_row] - p_pos[p_col], dim=1, p=2).unsqueeze(-1)
    
    data['protein', 'interacts_with', 'protein'].edge_index = pp_edge_index
    data['protein', 'interacts_with', 'protein'].edge_attr = pp_dist

    # ==================================================
    # 2. Build Ligand Nodes
    # ==================================================
    Y = feature_dict.get("ligand_Y")        # [M, 3]
    Y_t = feature_dict.get("ligand_Y_t")    # [M] Elemental IDs
    Y_m = feature_dict.get("ligand_Y_m")    # [M] Valid mask
    
    # Default to an empty, floating 0-node tensor setup if no ligands exist
    num_ligand_atoms = 0
    if Y is not None and Y_m is not None:
        Y_mask = Y_m.bool()
        if Y_mask.sum() > 0:
            Y = Y[Y_mask]
            Y_t = Y_t[Y_mask]
            num_ligand_atoms = Y.shape[0]
            
            # Simple embedding: treating element ID as a biological one-hot categorical block.
            # Using our encode_ligand_elements to create an [M, 6] tensor structurally identical 
            # to the protein's dihedral format for symmetry.
            lig_x = encode_ligand_elements(Y_t)
            
            data['ligand'].x = lig_x
            data['ligand'].pos = Y.float()
            
    if num_ligand_atoms > 0:
        # Cross-edges: protein -> ligand
        l_pos = data['ligand'].pos
        
        # radius(x, y, r) Output edge_index: [2, E] where row 0 is y, row 1 is x
        # radius(l_pos, p_pos, r) -> row 0 = p_pos indices, row 1 = l_pos indices
        pl_edge_index = radius(l_pos, p_pos, r=radius_cutoff)
        
        if pl_edge_index.size(1) > 0:
            p_idx, l_idx = pl_edge_index[0], pl_edge_index[1]
            
            # ['ligand', 'binds', 'protein']: row 0 is ligand (src), row 1 is protein (dst)
            lp_edge_index = torch.stack([l_idx, p_idx], dim=0)
            lp_dist = torch.norm(l_pos[l_idx] - p_pos[p_idx], dim=1, p=2).unsqueeze(-1)
            
            data['ligand', 'binds', 'protein'].edge_index = lp_edge_index
            data['ligand', 'binds', 'protein'].edge_attr = lp_dist
            
            # And reverse: ['protein', 'binds', 'ligand']
            pl_edge_index_rev = torch.stack([p_idx, l_idx], dim=0)
            data['protein', 'binds', 'ligand'].edge_index = pl_edge_index_rev
            data['protein', 'binds', 'ligand'].edge_attr = lp_dist.clone()
        else:
            # Fallback empty tensors if no atoms are within radius
            data['ligand', 'binds', 'protein'].edge_index = torch.empty((2, 0), dtype=torch.long)
            data['ligand', 'binds', 'protein'].edge_attr = torch.empty((0, 1), dtype=torch.float32)
            data['protein', 'binds', 'ligand'].edge_index = torch.empty((2, 0), dtype=torch.long)
            data['protein', 'binds', 'ligand'].edge_attr = torch.empty((0, 1), dtype=torch.float32)
            
    else:
        # Fallback empty state for PDBs strictly lacking any small molecules (Apos)
        data['ligand'].x = torch.empty((0, 6), dtype=torch.float32)
        data['ligand'].pos = torch.empty((0, 3), dtype=torch.float32)
        data['ligand', 'binds', 'protein'].edge_index = torch.empty((2, 0), dtype=torch.long)
        data['ligand', 'binds', 'protein'].edge_attr = torch.empty((0, 1), dtype=torch.float32)
        data['protein', 'binds', 'ligand'].edge_index = torch.empty((2, 0), dtype=torch.long)
        data['protein', 'binds', 'ligand'].edge_attr = torch.empty((0, 1), dtype=torch.float32)

    return data


def pdb_to_pyg_data(pdb_path, radius=8.0, device="cpu"):
    """
    Replaces the initial ProDy parser with LigandMPNN's native parser snippet.
    """
    feature_dict = get_ligandmpnn_features(pdb_path, device=device)
    data = dict_to_pyg_data(feature_dict, radius_cutoff=radius)
    return data


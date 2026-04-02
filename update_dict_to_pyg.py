import re

with open("utils/graph_builder.py", "r") as f:
    content = f.read()

hetero_import = "from torch_geometric.data import HeteroData\n"
if "HeteroData" not in content:
    content = content.replace("from torch_geometric.data import Data\n", "from torch_geometric.data import Data, HeteroData\n")

from_radius = "from torch_geometric.nn import radius_graph"
to_radius = "from torch_geometric.nn import radius_graph, radius"
if "import radius_graph, radius" not in content:
    content = content.replace(from_radius, to_radius)

new_func = '''def dict_to_pyg_data(feature_dict, radius_cutoff=8.0):
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
            
            # Simple embedding: treating element ID as numerical feature, 
            # ideally one-hot encoded, but let's shape it as [M, 6] for symmetry 
            # with the dihedrals to start (with dummy values or elemental integers).
            # LigandMPNN element lists: H, He, Li... so Y_t ranges 0-30 mostly.
            lig_x = torch.zeros(num_ligand_atoms, 6, dtype=torch.float32)
            lig_x[:, 0] = Y_t.float() # Just slot elemental ID in first col
            
            data['ligand'].x = lig_x
            data['ligand'].pos = Y.float()
            
    if num_ligand_atoms > 0:
        # Cross-edges: protein -> ligand
        l_pos = data['ligand'].pos
        
        # radius(x, y, r) finds all edges from x to y within r
        # Output edge_index: [2, num_edges] where row 0 is y, row 1 is x
        # So radius(l_pos, p_pos, r) -> edges from l_pos to p_pos
        # Note: PyG radius returns (col, row) mapping. Meaning y is row 0, x is row 1
        pl_edge_index = radius(l_pos, p_pos, r=radius_cutoff) # mapping from Ligand to Protein
        # The returned pl_edge_index row 0 is indices in l_pos, row 1 is indices in p_pos.
        # Format: ['ligand', 'binds', 'protein'] -> Row 0 = ligand id, Row 1 = protein id
        lp_edge_index = pl_edge_index # [2, E] where row[0] is ligand, row[1] is protein
        
        if lp_edge_index.size(1) > 0:
            lp_row, lp_col = lp_edge_index[0], lp_edge_index[1]
            lp_dist = torch.norm(l_pos[lp_row] - p_pos[lp_col], dim=1, p=2).unsqueeze(-1)
            
            data['ligand', 'binds', 'protein'].edge_index = lp_edge_index
            data['ligand', 'binds', 'protein'].edge_attr = lp_dist
            
            # And reverse: Protein to Ligand
            pl_edge_index_rev = torch.stack([lp_col, lp_row], dim=0)
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
'''

import re
content = re.sub(r'def dict_to_pyg_data.*?return data', new_func, content, flags=re.DOTALL)

with open("utils/graph_builder.py", "w") as f:
    f.write(content)

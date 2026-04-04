import torch
import torch.nn as nn
from torch_geometric.nn import TransformerConv, Linear, HeteroConv
import torch.nn.functional as F

class GaussianSmearing(nn.Module):
    """
    Expands a scalar distance tensor into a 16-dimensional Radial Basis Function (RBF) vector.
    This creates a rich representation for neural networks out of a single scalar distance.
    """
    def __init__(self, start=0.0, stop=8.0, num_gaussians=16):
        super(GaussianSmearing, self).__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        # Calculate the Gaussian coefficient directly handling width scale
        self.coeff = -0.5 / ((stop - start) / (num_gaussians - 1))**2
        self.register_buffer('offset', offset)

    def forward(self, dist):
        # dist shape: [E, 1]
        # output shape: [E, 16]
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))

class ResidualAttentionBlock(nn.Module):
    def __init__(self, hidden_dim, edge_dim=19, dropout=0.1):
        super(ResidualAttentionBlock, self).__init__()
        # TransformerConv natively absorbs edge_attr into its attention mechanism.
        # We split the 128 hidden_dim across 4 attention heads (32 dim per head) to maintain exact tensor shapes.
        self.conv = TransformerConv(hidden_dim, hidden_dim // 4, heads=4, concat=True, edge_dim=edge_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, edge_index, edge_attr):
        # Handle bipartite graphs (e.g. ['ligand', 'binds', 'protein'])
        if isinstance(x, tuple):
            x_src, x_dst = x
            identity = x_dst
        else:
            identity = x
            
        out = self.conv(x, edge_index, edge_attr)
        out = self.norm(out)
        out = F.relu(out)
        out = self.dropout(out)
        return out + identity

class Struct2SeqGNN(nn.Module):
    def __init__(self, node_features=6, ligand_features=6, hidden_dim=128, num_classes=21, num_layers=4, dropout=0.1):
        super(Struct2SeqGNN, self).__init__()
        
        # Initial node embeddings for distinct node types
        self.protein_emb = Linear(node_features, hidden_dim)
        self.ligand_emb = Linear(ligand_features, hidden_dim)
        
        # Distinct RBF distance expansions for each edge type
        # Decoupling these allows the network to scale bonds differently (e.g. covalent backbone vs weak ionic ligand bonds)
        self.edge_embs = nn.ModuleDict({
            'protein__interacts_with__protein': GaussianSmearing(start=0.0, stop=8.0, num_gaussians=16),
            'ligand__binds__protein': GaussianSmearing(start=0.0, stop=8.0, num_gaussians=16),
            'protein__binds__ligand': GaussianSmearing(start=0.0, stop=8.0, num_gaussians=16)
        })
        
        # Deep module list involving HeteroConv
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                ('protein', 'interacts_with', 'protein'): ResidualAttentionBlock(hidden_dim, edge_dim=19, dropout=dropout),
                ('ligand', 'binds', 'protein'): ResidualAttentionBlock(hidden_dim, edge_dim=19, dropout=dropout),
                ('protein', 'binds', 'ligand'): ResidualAttentionBlock(hidden_dim, edge_dim=19, dropout=dropout)
            }, aggr='sum')
            self.layers.append(conv)
        
        # Final layer normalization before classification (only applied to protein output)
        self.norm_out = nn.LayerNorm(hidden_dim)
        
        # Sequence prediction: linear classification layer for standard amino acids
        self.fc = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, data):
        x_dict = data.x_dict
        edge_index_dict = data.edge_index_dict
        edge_attr_dict = data.edge_attr_dict
        
        # 1. Expand features
        x_dict_expanded = {
            'protein': self.protein_emb(x_dict['protein'])
        }
        if 'ligand' in x_dict:
            x_dict_expanded['ligand'] = self.ligand_emb(x_dict['ligand'])
        x_dict = x_dict_expanded
        
        edge_attr_dict_expanded = {}
        for edge_type, edge_attr in edge_attr_dict.items():
            src_type, rel_type, dst_type = edge_type
            
            # Using the pre-calculated distance
            dist = edge_attr[:, 0]
            
            # Generate 3D direction vectors symmetrically on-the-fly!
            # By generating these dynamically inside the forward pass, we skip having to delete
            # and re-generate the 150k hard drive .pt datasets!
            src_pos = data[src_type].pos
            dst_pos = data[dst_type].pos
            src_idx, dst_idx = edge_index_dict[edge_type]
            
            vec = dst_pos[dst_idx] - src_pos[src_idx]
            # Convert raw coordinates into purely directional unit vectors
            vec_norm = vec / (torch.norm(vec, dim=-1, keepdim=True) + 1e-7)

            key = f"{src_type}__{rel_type}__{dst_type}"
            if key in self.edge_embs:
                # Expand standard scalar distance to [E, 16] 
                dist_smeared = self.edge_embs[key](dist)
                
                # Combine length and direction! Output -> [E, 19]
                edge_attr_dict_expanded[edge_type] = torch.cat([dist_smeared, vec_norm], dim=-1)
            else:
                edge_attr_dict_expanded[edge_type] = edge_attr
        
        # 2. Iterative Message Passing through HeteroConv
        for layer in self.layers:
            x_dict = layer(x_dict, edge_index_dict, edge_attr_dict=edge_attr_dict_expanded)
            
        # 3. Readout on Protein nodes
        protein_x = x_dict['protein']
        protein_x = self.norm_out(protein_x)
        logits = self.fc(protein_x)
        
        return logits

import torch
import torch.nn as nn
from torch_geometric.nn import CGConv, Linear, HeteroConv
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

class ResidualCGConvBlock(nn.Module):
    def __init__(self, hidden_dim, edge_dim=16, dropout=0.1):
        super(ResidualCGConvBlock, self).__init__()
        self.conv = CGConv(hidden_dim, dim=edge_dim, batch_norm=True)
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
                ('protein', 'interacts_with', 'protein'): ResidualCGConvBlock(hidden_dim, edge_dim=16, dropout=dropout),
                ('ligand', 'binds', 'protein'): ResidualCGConvBlock(hidden_dim, edge_dim=16, dropout=dropout),
                ('protein', 'binds', 'ligand'): ResidualCGConvBlock(hidden_dim, edge_dim=16, dropout=dropout)
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
        x_dict = {
            'protein': self.protein_emb(x_dict['protein']),
            'ligand': self.ligand_emb(x_dict['ligand'])
        }
        
        edge_attr_dict_expanded = {}
        for edge_type, edge_attr in edge_attr_dict.items():
            # PyG creates string representations of edge type tuples (src, edge, dst)
            # which we convert to valid py keys for the ModuleDict lookup via '__'
            key = f"{edge_type[0]}__{edge_type[1]}__{edge_type[2]}"
            if key in self.edge_embs:
                edge_attr_dict_expanded[edge_type] = self.edge_embs[key](edge_attr)
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

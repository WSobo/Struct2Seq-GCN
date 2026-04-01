import torch
import torch.nn as nn
from torch_geometric.nn import CGConv, Linear
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

class Struct2SeqGCN(nn.Module):
    def __init__(self, node_features=6, hidden_dim=128, num_classes=21):
        super(Struct2SeqGCN, self).__init__()
        
        # Initial node embedding to match hidden dimensions
        self.node_emb = Linear(node_features, hidden_dim)
        
        # Distance Expansion: Maps [E, 1] distance scalars -> [E, 16] RBF embeddings
        self.edge_emb = GaussianSmearing(start=0.0, stop=8.0, num_gaussians=16)
        
        # Edge-Conditioned Message Passing (CGConv explicitly takes edge attributes)
        # dim=16 because our edge_attr is now an expanded 16D RBF vector
        self.conv1 = CGConv(hidden_dim, dim=16, batch_norm=True)
        self.conv2 = CGConv(hidden_dim, dim=16, batch_norm=True)
        
        # Sequence prediction: linear classification layer for standard amino acids
        self.fc = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # Expand raw scalar distances [E, 1] to enriched RBF representations [E, 16]
        edge_attr = self.edge_emb(edge_attr)
        
        # Convert SE(3) Invariant dihedral features into high-dimensional node features
        x = self.node_emb(x)
        
        # Message Passing Layer 1 (now conditioned on enriched 16D RBF edge targets)
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = F.dropout(x, p=0.1, training=self.training)
        
        # Message Passing Layer 2
        x = self.conv2(x, edge_index, edge_attr)
        x = F.relu(x)
        
        # Output Logits
        logits = self.fc(x)
        return logits

import torch
import torch.nn as nn
from torch_geometric.nn import CGConv, Linear
import torch.nn.functional as F

class Struct2SeqGCN(nn.Module):
    def __init__(self, node_features=6, hidden_dim=128, num_classes=21):
        super(Struct2SeqGCN, self).__init__()
        
        # Initial node embedding to match hidden dimensions
        self.node_emb = Linear(node_features, hidden_dim)
        
        # Edge-Conditioned Message Passing (CGConv explicitly takes edge attributes)
        # dim=1 because our edge_attr is the [E, 1] distance calculation
        self.conv1 = CGConv(hidden_dim, dim=1, batch_norm=True)
        self.conv2 = CGConv(hidden_dim, dim=1, batch_norm=True)
        
        # Sequence prediction: linear classification layer for standard amino acids
        self.fc = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # Convert raw Cartesian coordinates into high-dimensional node features
        x = self.node_emb(x)
        
        # Message Passing Layer 1 (now conditioned on Euclidean pairwise distances)
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = F.dropout(x, p=0.1, training=self.training)
        
        # Message Passing Layer 2
        x = self.conv2(x, edge_index, edge_attr)
        x = F.relu(x)
        
        # Output Logits
        logits = self.fc(x)
        return logits

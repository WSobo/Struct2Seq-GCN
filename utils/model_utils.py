import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

class Struct2SeqGCN(nn.Module):
    def __init__(self, node_features=3, hidden_dim=128, num_classes=21):
        super(Struct2SeqGCN, self).__init__()
        # v1.0 architecture uses absolute (X, Y, Z) coordinates
        self.conv1 = GCNConv(node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        
        # Sequence prediction: linear classification layer for standard amino acids
        self.fc = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # Message Passing Layer 1
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.1, training=self.training)
        
        # Message Passing Layer 2
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        
        # Output Logits
        logits = self.fc(x)
        return logits

import torch
from utils.model_utils import Struct2SeqGNN
from torch_geometric.data import HeteroData

data = HeteroData()
data['protein'].x = torch.randn(10, 6)
data['protein'].pos = torch.randn(10, 3)
data['ligand'].x = torch.randn(5, 6)
data['ligand'].pos = torch.randn(5, 3)

data['protein', 'interacts_with', 'protein'].edge_index = torch.empty((2, 0), dtype=torch.long)
data['protein', 'interacts_with', 'protein'].edge_attr = torch.empty((0, 1))

model = Struct2SeqGNN()
out = model(data)
print("Forward pass successful. Output shape:", out.shape)

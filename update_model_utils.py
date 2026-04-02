import re

with open("utils/model_utils.py", "r") as f:
    content = f.read()

hetero_conv_import = "from torch_geometric.nn import CGConv, Linear, HeteroConv\n"
content = content.replace("from torch_geometric.nn import CGConv, Linear\n", hetero_conv_import)

new_res_block = '''class ResidualCGConvBlock(nn.Module):
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
        return out + identity'''

content = re.sub(r'class ResidualCGConvBlock.*?return x \+ identity', new_res_block, content, flags=re.DOTALL)

new_model = '''class Struct2SeqGCN(nn.Module):
    def __init__(self, node_features=6, ligand_features=6, hidden_dim=128, num_classes=21, num_layers=4, dropout=0.1):
        super(Struct2SeqGCN, self).__init__()
        
        # Initial node embeddings for distinct node types
        self.protein_emb = Linear(node_features, hidden_dim)
        self.ligand_emb = Linear(ligand_features, hidden_dim)
        
        # Shared distance expansion for all edge types
        self.edge_emb = GaussianSmearing(start=0.0, stop=8.0, num_gaussians=16)
        
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
            edge_attr_dict_expanded[edge_type] = self.edge_emb(edge_attr)
        
        # 2. Iterative Message Passing through HeteroConv
        for layer in self.layers:
            x_dict = layer(x_dict, edge_index_dict, edge_attr_dict_expanded)
            
        # 3. Readout on Protein nodes
        protein_x = x_dict['protein']
        protein_x = self.norm_out(protein_x)
        logits = self.fc(protein_x)
        
        return logits'''

content = re.sub(r'class Struct2SeqGCN.*?return logits', new_model, content, flags=re.DOTALL)

with open("utils/model_utils.py", "w") as f:
    f.write(content)

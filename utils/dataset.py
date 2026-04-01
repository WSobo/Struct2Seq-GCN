import os
import json
import torch
from torch_geometric.data import Dataset
from utils.graph_builder import pdb_to_pyg_data

class Struct2SeqDataset(Dataset):
    def __init__(self, root, json_file, pdb_dir, radius=8.0, transform=None, pre_transform=None):
        """
        Args:
            root (string): Root directory where the dataset should be saved.
            json_file (string): Path to LigandMPNN training JSON (e.g., train.json) containing list of PDB IDs.
            pdb_dir (string): Directory containing the actual .pdb or .pt files.
            radius (float): Distance cutoff for the graph edges.
        """
        self.pdb_dir = pdb_dir
        self.radius = radius
        with open(json_file, 'r') as f:
            self.pdb_ids = json.load(f)
            
        super().__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        # We assume the raw files are inside pdb_dir and correspond to the IDs
        return [f"{pdb_id}.pdb" for pdb_id in self.pdb_ids]

    @property
    def processed_file_names(self):
        return [f"data_{pdb_id}_v3.pt" for pdb_id in self.pdb_ids]

    def download(self):
        # In a real scenario, you might download missing PDBs here.
        pass

    def process(self):
        for pdb_id in self.pdb_ids:
            # Common extensions could be .pdb or .cif or .pt
            pdb_path = os.path.join(self.pdb_dir, f"{pdb_id}.pdb")
            
            # Skip if file doesn't exist to prevent crash during setup
            if not os.path.exists(pdb_path):
                # Try .pt (LigandMPNN format)
                pdb_path = os.path.join(self.pdb_dir, f"{pdb_id}.pt")
                if not os.path.exists(pdb_path):
                    continue
                    
            try:
                # Convert to PyG Data using LigandMPNN's exact methods
                data = pdb_to_pyg_data(pdb_path, radius=self.radius)
                
                # Save processed data
                torch.save(data, os.path.join(self.processed_dir, f"data_{pdb_id}_v3.pt"))
            except Exception as e:
                print(f"Error processing {pdb_id}: {e}")

    def len(self):
        return len(self.pdb_ids)

    def get(self, idx):
        pdb_id = self.pdb_ids[idx]
        data = torch.load(os.path.join(self.processed_dir, f"data_{pdb_id}_v3.pt"), weights_only=False)
        return data

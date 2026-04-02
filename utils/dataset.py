import os
import json
import torch
from torch_geometric.data import Dataset
from utils.graph_builder import pdb_to_pyg_data

class Struct2SeqDataset(Dataset):
    def __init__(self, root, json_file, pdb_dir, radius=8.0, max_samples=None, transform=None, pre_transform=None):
        """
        Args:
            root (string): Root directory where the dataset should be saved.
            json_file (string): Path to LigandMPNN training JSON (e.g., train.json) containing list of PDB IDs.
            pdb_dir (string): Directory containing the actual .pdb or .pt files.
            radius (float): Distance cutoff for the graph edges.
            max_samples (int, optional): Maximum number of random files to process (useful for testing).
        """
        self.pdb_dir = pdb_dir
        self.radius = radius
        with open(json_file, 'r') as f:
            self.pdb_ids = json.load(f)
            
        if max_samples is not None and len(self.pdb_ids) > max_samples:
            import random
            random.seed(42)  # Set seed for reproducible subsets
            self.pdb_ids = random.sample(self.pdb_ids, max_samples)
            print(f"Randomly selected {max_samples} structures from {json_file}")
            
        super().__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        # We assume the raw files are inside pdb_dir and correspond to the IDs
        return [f"{pdb_id}.pdb" for pdb_id in self.pdb_ids]

    @property
    def processed_file_names(self):
        return [f"data_{pdb_id}.pt" for pdb_id in self.pdb_ids]

    def download(self):
        # In a real scenario, you might download missing PDBs here.
        pass

    def process(self):
        import urllib.request
        from concurrent.futures import ProcessPoolExecutor, as_completed
        os.makedirs(self.pdb_dir, exist_ok=True)
        
        def process_single(pdb_id):
            processed_path = os.path.join(self.processed_dir, f"data_{pdb_id}.pt")
            # Skip if already processed
            if os.path.exists(processed_path):
                return
                
            # Parse RCSB-style HPC-safe directories
            pdb_id_str = str(pdb_id).lower()
            sub_dir = pdb_id_str[1:3] if len(pdb_id_str) >= 4 else "misc"
                
            # Common extensions could be .pdb or .cif or .pt
            pdb_path = os.path.join(self.pdb_dir, sub_dir, f"{pdb_id}.pdb")
            pt_path = os.path.join(self.pdb_dir, sub_dir, f"{pdb_id}.pt")
            
            active_path = None
            if os.path.exists(pdb_path):
                active_path = pdb_path
            elif os.path.exists(pt_path):
                active_path = pt_path
            else:
                # Disable HTTP fallback to avoid RCSB API rate limits on massive datasets
                print(f"File missing for {pdb_id}. Ensure it is fetched via rsync beforehand. Skipping.")
                return
                    
            try:
                # Convert to PyG Data using LigandMPNN's exact methods
                data = pdb_to_pyg_data(active_path, radius=self.radius)
                
                # Save processed data
                torch.save(data, processed_path)
            except Exception as e:
                print(f"Error processing {pdb_id}: {e}")

        # Use multiprocessing to speed up building the dataset maps massively
        num_workers = min(os.cpu_count() or 1, 16)
        print(f"Generating parsed graph files concurrently with {num_workers} processes...")
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(process_single, pdb) for pdb in self.pdb_ids]
            for future in as_completed(futures):
                pass

    def len(self):
        return len(self.pdb_ids)

    def get(self, idx):
        # Prevent expensive glob fallback during rapid dataloader batching
        pdb_id = self.pdb_ids[idx]
        pt_file = os.path.join(self.processed_dir, f"data_{pdb_id}.pt")
        
        if not os.path.exists(pt_file):
            raise FileNotFoundError(f"Missing graph for {pdb_id}. Run process() properly offline first.")
                
        data = torch.load(pt_file, weights_only=False)
        return data

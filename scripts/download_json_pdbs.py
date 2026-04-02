import os
import sys
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

def download_pdb(pdb_id, out_dir):
    pdb_id = str(pdb_id).lower()
    out_path = os.path.join(out_dir, f"{pdb_id}.pdb")
    
    if os.path.exists(out_path):
        return True
        
    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
    for attempt in range(3):
        try:
            urllib.request.urlretrieve(url, out_path)
            return True
        except Exception as e:
            time.sleep(2 * (attempt + 1))  # Exponential backoff
            
    print(f"Failed to fetch {pdb_id}")
    return False

def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    inputs_dir = os.path.join(project_root, "inputs")
    os.makedirs(inputs_dir, exist_ok=True)
    
    json_files = [
        os.path.join(project_root, "LigandMPNN", "training", "train.json"),
        os.path.join(project_root, "LigandMPNN", "training", "valid.json"),
        os.path.join(project_root, "LigandMPNN", "training", "test_small_molecule.json")
    ]
    
    pdb_ids = set()
    for json_file in json_files:
        if os.path.exists(json_file):
            print(f"Parsing IDs from: {json_file}")
            with open(json_file, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    pdb_ids.update(data)
                elif isinstance(data, dict):
                    pdb_ids.update(data.keys())
                    
    print(f"Discovered {len(pdb_ids)} unique PDB structures to download.")
    
    # 8 workers typically avoids RCSB 429 rate limits safely
    success = 0
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(download_pdb, pid, inputs_dir) for pid in pdb_ids]
        for i, future in enumerate(as_completed(futures)):
            if future.result():
                success += 1
            if (i + 1) % 500 == 0:
                print(f"Progress: {i + 1}/{len(pdb_ids)}...")
                
    print(f"✅ Successfully prepared {success} out of {len(pdb_ids)} PDB files in {inputs_dir}/")

if __name__ == "__main__":
    main()
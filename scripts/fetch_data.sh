#!/bin/bash
#SBATCH --job-name=fetch_pdb
#SBATCH --partition=cpu             # Use CPU partition! Do not waste expensive GPU hours on a network download
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4           # Minimal CPU needed just for unzipping/rsyncing
#SBATCH --mem=8G                    # Minimal RAM
#SBATCH --time=24:00:00             # Allow up to 24 hours for the massive initial sync
#SBATCH --output=slurm_fetch_%j.out

echo "Running on hosts: $SLURM_NODELIST"
echo "Current working directory is $PWD"

# Change to the submission directory automatically
cd "${SLURM_SUBMIT_DIR}"

# Step 1. Navigate up to root and ensure inputs dir exists
cd ..
mkdir -p inputs/
cd scripts/

# Step 2. Download the official, safe RCSB high-throughput mirroring script
echo "Fetching official RCSB rsync payload script..."
curl -s -O https://cdn.rcsb.org/rcsb-pdb/general_information/news_publications/rsyncPDB.sh
chmod +x rsyncPDB.sh

# Step 3. Execute the mirror targeting the inputs folder
echo "Beginning massive bulk PDB download via Rsync (This may take hours)..."
./rsyncPDB.sh ../inputs/

echo "✅ 150k Bulk sync complete!"
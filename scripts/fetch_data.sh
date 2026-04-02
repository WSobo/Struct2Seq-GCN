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

# Provide Micromamba context so python is available
export MAMBA_ROOT_PREFIX=$HOME/micromamba
eval "$(micromamba shell hook --shell bash)"
micromamba activate s2s-gnn

echo "Scraping specific JSON IDs from LigandMPNN and downloading flat files safely..."
python scripts/download_json_pdbs.py

echo "✅ Targeted JSON download complete!"
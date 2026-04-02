#!/bin/bash
#SBATCH --job-name=fetch_pdb
#SBATCH --output=/private/groups/yehlab/wsobolew/02_projects/computational/Struct2Seq-GNN/logs/out/fetch_pdb_%j.out
#SBATCH --error=/private/groups/yehlab/wsobolew/02_projects/computational/Struct2Seq-GNN/logs/err/fetch_pdb_%j.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --partition=long
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=wsobolew@ucsc.edu

# Error handling
set -e

# Initialize micromamba for this shell session
eval "$(micromamba shell hook --shell bash)"
micromamba activate S2S-GNN

# Store the original working directory
ORIGINAL_DIR=$(pwd)

# Change to the project directory for execution
cd /private/groups/yehlab/wsobolew/02_projects/computational/Struct2Seq-GNN

echo "Running on hosts: $SLURM_NODELIST"
echo "Timestamp: $(date)"

echo "Scraping specific JSON IDs from LigandMPNN and downloading flat files safely..."
srun python scripts/download_json_pdbs.py

echo "✅ Targeted JSON download complete at: $(date)"
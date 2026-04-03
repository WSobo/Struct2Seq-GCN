#!/bin/bash
#SBATCH --job-name=s2s-preproc
#SBATCH --output=/private/groups/yehlab/wsobolew/02_projects/computational/Struct2Seq-GNN/logs/out/s2s-preproc_%j.out
#SBATCH --error=/private/groups/yehlab/wsobolew/02_projects/computational/Struct2Seq-GNN/logs/err/s2s-preproc_%j.err
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --partition=long
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=wsobolew@ucsc.edu

# Error handling
set -e

# Initialize micromamba for this shell session
eval "$(micromamba shell hook --shell bash)"
micromamba activate S2S-GNN

# Change to the project directory for execution
cd /private/groups/yehlab/wsobolew/02_projects/computational/Struct2Seq-GNN

echo "Running on node: $SLURM_NODELIST"
echo "Timestamp: $(date)"

echo "Kicking off CPU-only preprocessing to cache .pt files..."
srun python scripts/preprocess.py --pdb_dir inputs/

echo "Pre-processing completed at: $(date)"

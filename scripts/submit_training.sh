#!/bin/bash
#SBATCH --job-name=s2s-gnn
#SBATCH --output=/private/groups/yehlab/wsobolew/02_projects/computational/Struct2Seq-GNN/logs/out/s2s-gnn_%j.out
#SBATCH --error=/private/groups/yehlab/wsobolew/02_projects/computational/Struct2Seq-GNN/logs/err/s2s-gnn_%j.err
#SBATCH --time=96:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
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

# Enable Tensor Core optimization by setting precision
python -c "import torch; torch.set_float32_matmul_precision('medium'); print('Tensor precision optimized')"

echo "Booting Multi-GPU TorchRun..."
srun torchrun --standalone --nproc_per_node=4 scripts/train.py \
    --pdb_dir inputs/ \
    --batch_size 16 \
    --num_workers 4 \
    --pin_memory \
    --hidden_dim 256 \
    --num_layers 6 \
    --epochs 50

# NOTE: A batch size of 16 distributed over 4 DDP GPUs yields a real batch size of 64 per step.
echo "Training completed at: $(date)"
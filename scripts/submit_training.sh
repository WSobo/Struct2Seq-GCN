#!/bin/bash
#SBATCH --job-name=s2s-gnn
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:A5500:4          # Request 4x A5500 GPUs
#SBATCH --cpus-per-task=16          # 16 CPU cores for dataset multiprocessing
#SBATCH --mem=64G                   # 64GB RAM for 150k loaded dataset pointers
#SBATCH --time=96:00:00             # 96 hours bounds overkill for 50 epochs at batch size 16*4
#SBATCH --output=slurm_train_%j.out # Standard text output dump

echo "Running on hosts: $SLURM_NODELIST"
echo "Running on $SLURM_NNODES nodes."
echo "Running on $SLURM_NPROCS processors."
echo "Current working directory is $PWD"

# Change to the submission directory automatically
cd "${SLURM_SUBMIT_DIR}"

# 1. Provide Micromamba context (adjust paths to where micromamba lives on your HPC)
export MAMBA_ROOT_PREFIX=$HOME/micromamba
eval "$(micromamba shell hook --shell bash)"

# 2. Environment Bootstrapping
echo "Building Micromamba Environment..."
micromamba create -n s2s-gnn python=3.10 -y
micromamba activate s2s-gnn

# 3. Dependency Sync
echo "Installing pip requirements from Google Colab equivalencies..."
pip install -r requirements.txt
# Additional install for PyG scatter & cluster dependencies compiled cleanly against standard torch
# (Adjust the CUDA version in the URL if your HPC GPUs require a different version)
pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.2.0+cu118.html

# 4. Distributed Execution Trigger (DDP scale-up)
echo "Booting Multi-GPU TorchRun..."
torchrun --standalone --nproc_per_node=4 scripts/train.py \
    --pdb_dir inputs/ \
    --batch_size 16 \
    --num_workers 4 \
    --pin_memory \
    --hidden_dim 256 \
    --num_layers 6 \
    --epochs 50

# NOTE: A batch size of 16 distributed over 4 DDP GPUs yields a real batch size of 64 per step.
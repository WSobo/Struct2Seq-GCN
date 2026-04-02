# Struct2Seq-GCN: Heterogeneous Protein-Ligand Sequence Design

## Overview
Struct2Seq-GCN is an advanced Heterogeneous Graph Neural Network built in PyTorch Geometric (PyG). It dynamically predicts optimal amino acid sequences by co-embedding the 3D geometric environments of protein backbones alongside spatial ligand interactions.

## The Pipeline

- **Data Parsing:** Hooks directly into **LigandMPNN's native parser**, leveraging their exact dictionaries and masking to ensure 1:1 equivalency. 
- **Heterogeneous Graph Construction:** Converts 3D space into a directed `HeteroData` graph. Nodes represent both structural residues and ligand atoms. Edges establish discrete domains: `protein-protein`, `protein-ligand`, and `ligand-protein`.
- **Decoupled Message Passing:** Distinct radial basis functions (RBF) via decoupled Gaussian Smearing to independently learn structural constraints (e.g. covalent backbone topology vs. non-covalent ligand binding constraints).
- **Sequence Prediction:** A classification layer outputs logits representing the standard amino acid vocabulary, evaluated by predicting and recovering native sequence residues (NSR%).

## Recent Upgrades (HPC & Scale-Ready)

✅ **Heterogeneous Edge Architectures**: Transitioned from homogenous networks to full multi-modal interactions.
✅ **HPC MLOps Scaling**: Optimized `scripts/train.py` with multi-node prefetching (`num_workers`, `pin_memory`) and high-frequency `global_step` checkpointing to prevent job evictions.
✅ **End-to-End Inference**: Added `scripts/inference.py` and `notebooks/inference_demo.ipynb` to evaluate structures against ground truths and emit valid FASTA sequences.
✅ **Bulk 150k PDB Processing**: Added `notebooks/train_full_scale.ipynb` with `rsync` workflows to seamlessly mirror the RCSB database for SLURM cluster training without hitting HTTP 429 API rate limits.

## Setup & Installation

To install the necessary components, ensure you have an Anaconda or Python virtual environment setup:

```bash
# Clone the repository
git clone https://github.com/WSobo/Struct2Seq-GCN.git
cd Struct2Seq-GCN

# Install PyTorch
pip install torch torchvision torchaudio

# Install PyTorch Geometric (PyG) and helpers
pip install torch_geometric prody torch-cluster
```

## Running the Pipeline

### Training (HPC / SLURM Full Scale)
To train the model natively using scaled architecture sizes and memory configurations:
```bash
python scripts/train.py \
    --pdb_dir data/pdb_archive \
    --batch_size 16 \
    --num_workers 8 \
    --pin_memory \
    --epochs 30 \
    --hidden_dim 256 \
    --num_layers 6 \
    --checkpoint_interval 5000
```
*See `notebooks/train_full_scale.ipynb` for instructions on performing the bulk RCSB Rsync to acquire the dataset.*

### Inference & Sequence Design
To make predictions from custom backbones directly and calculate Native Sequence Recovery (NSR):
```bash
python scripts/inference.py \
    --pdb LigandMPNN/inputs/1BC8.pdb \
    --weights outputs/best_model.pt \
    --out_fasta outputs/1BC8_predicted.fasta
```
*See `notebooks/inference_demo.ipynb` for a fully interactive Colab-ready environment.*

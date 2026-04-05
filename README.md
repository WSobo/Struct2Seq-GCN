# Struct2Seq-GNN: Heterogeneous Protein-Ligand Sequence Design

## Overview
Struct2Seq-GNN is a Heterogeneous Graph Neural Network (HGNN) built in PyTorch Geometric (PyG). It predicts optimal amino acid sequences by co-embedding the 3D geometric environments of protein backbones alongside spatial ligand interactions, using **LigandMPNN** as its structural parser to guarantee 1:1 feature equivalency.

A pretrained v1.0 checkpoint is available in `pretrained_models/v1.0/best_model.pt`.

## Architecture

### Graph Construction (`utils/graph_builder.py`)
Raw PDB files are parsed through LigandMPNN's native `parse_PDB` + `featurize` pipeline. The output is a `HeteroData` graph with two node types and three directed edge types:

| Component | Details |
|---|---|
| **Protein nodes** | 6-dim backbone dihedral features (sin/cos of φ, ψ, ω) at Cα positions |
| **Ligand nodes** | 6-dim one-hot element encoding (C, N, O, S, P, Other) |
| `protein → protein` | Radius graph at 8 Å cutoff |
| `ligand → protein` | Cross-radius edges within 8 Å |
| `protein → ligand` | Reverse cross-radius edges within 8 Å |

### Message Passing (`utils/model_utils.py`)
- **`GaussianSmearing`**: Expands each scalar edge distance into a 16-dim RBF vector. Each edge type has its own independent smearing module, allowing the network to learn separate distance scales for covalent backbone topology versus non-covalent ligand binding.
- **`ResidualCGConvBlock`**: Wraps PyG's `CGConv` with `LayerNorm`, `ReLU`, `Dropout`, and a residual skip connection.
- **`HeteroConv`**: Applies a `ResidualCGConvBlock` per edge type simultaneously across `num_layers` message-passing rounds.
- **Classification head**: `LayerNorm` → `Linear(hidden_dim, 21)` over protein nodes only; evaluated by Native Sequence Recovery (NSR%).

## Repository Structure

```
Struct2Seq-GNN/
├── LigandMPNN/               # Submodule: structural parser
├── pretrained_models/v1.0/   # Released checkpoint + training history
├── utils/
│   ├── graph_builder.py      # PDB → HeteroData conversion
│   ├── dataset.py            # PyG Dataset with multiprocess preprocessing
│   └── model_utils.py        # GNN architecture definition
├── scripts/
│   ├── preprocess.py         # CPU-only preprocessing (cache .pt graphs)
│   ├── train.py              # DDP-ready training loop
│   ├── inference.py          # Inference + FASTA export
│   └── run.py                # Minimal single-PDB forward pass
├── notebooks/
│   ├── inference_demo.ipynb  # Interactive inference walkthrough
│   ├── test_pipeline.ipynb   # End-to-end pipeline smoke test
│   └── train_full_scale.ipynb # Bulk RCSB rsync + SLURM training guide
└── requirements.txt
```

## Setup & Installation

Requires CUDA 11.8. Tested with PyTorch 2.2.1.

```bash
# 1. Clone the repository and initialise the LigandMPNN submodule
git clone https://github.com/WSobo/Struct2Seq-GNN.git
cd Struct2Seq-GNN
git submodule update --init --recursive

# 2. Install PyTorch (CUDA 11.8)
pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 \
    --index-url https://download.pytorch.org/whl/cu118

# 3. Install PyTorch Geometric and core dependencies
pip install torch_geometric prody "numpy<2" pandas scipy tqdm jupyter ipykernel

# 4. Install PyG sparse/cluster extensions
pip install pyg_lib torch_scatter torch_sparse torch_cluster \
    -f https://data.pyg.org/whl/torch-2.2.0+cu118.html
```

## Running the Pipeline

### 1. Preprocessing (CPU job — cache graphs before GPU training)
Build and cache all `HeteroData` `.pt` files to disk before launching GPU training.
On HPC clusters, run this as a dedicated CPU preprocessing job so the GPU nodes skip the overhead entirely.
```bash
python scripts/preprocess.py \
    --json_train LigandMPNN/training/train.json \
    --json_valid LigandMPNN/training/valid.json \
    --pdb_dir data/pdb_archive
```

### 2. Training
Single-GPU or multi-GPU DDP training with mid-epoch checkpointing:
```bash
# Single GPU
python scripts/train.py \
    --json_train LigandMPNN/training/train.json \
    --json_valid LigandMPNN/training/valid.json \
    --pdb_dir data/pdb_archive \
    --epochs 50 \
    --batch_size 32 \
    --hidden_dim 128 \
    --num_layers 4 \
    --num_workers 4 \
    --pin_memory \
    --checkpoint_interval 5000

# Multi-GPU (e.g. 4 GPUs via torchrun)
torchrun --nproc_per_node=4 scripts/train.py \
    --json_train LigandMPNN/training/train.json \
    --json_valid LigandMPNN/training/valid.json \
    --pdb_dir data/pdb_archive \
    --hidden_dim 256 \
    --num_layers 6
```

Key training arguments:

| Argument | Default | Description |
|---|---|---|
| `--hidden_dim` | `128` | Node embedding size (scale up for larger datasets) |
| `--num_layers` | `4` | Number of HeteroConv message-passing rounds |
| `--num_workers` | `4` | CPU prefetch workers (prevents GPU starvation) |
| `--pin_memory` | `False` | Pinned memory for faster host-to-GPU transfers |
| `--log_interval` | `100` | Print loss every N steps |
| `--checkpoint_interval` | `5000` | Overwrite rolling checkpoint every N steps |
| `--max_samples` | `None` | Randomly subsample PDBs (useful for debugging) |

*See `notebooks/train_full_scale.ipynb` for bulk RCSB `rsync` instructions to mirror the full ~150k PDB dataset for SLURM cluster training.*

### 3. Inference & Sequence Design
Predict sequences from any PDB file and calculate Native Sequence Recovery (NSR):
```bash
python scripts/inference.py \
    --pdb LigandMPNN/inputs/1BC8.pdb \
    --weights pretrained_models/v1.0/best_model.pt \
    --out_fasta outputs/1BC8_predicted.fasta \
    --temperature 0.1
```
Key inference arguments:

| Argument | Default | Description |
|---|---|---|
| `--weights` | `outputs/best_model.pt` | Path to trained `.pt` checkpoint |
| `--radius` | `8.0` | Distance cutoff (Å) for graph edges |
| `--temperature` | `0.1` | Sampling temperature (`0.0` = greedy argmax) |
| `--fixed_residues` | `None` | Comma-separated zero-indexed positions to keep native (e.g. `10,11,15`) |

*See `notebooks/inference_demo.ipynb` for an interactive, Colab-ready walkthrough.*

## Benchmark Results (v2.0)

Evaluated on the LigandMPNN validation set (7448 structures).

- **Validation Loss:** 2.2295
- **Global Accuracy:** 30.36%
- **5.0Å Pocket Accuracy:** 35.13%

*Note: The model demonstrates stronger recovery around the ligand-binding pocket (5.0Å) compared to the overall global structure.*

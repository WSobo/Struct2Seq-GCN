# Struct2Seq-GCN: Backbone-Conditioned Sequence Design

## Overview
Struct2Seq-GCN is a lightweight Graph Convolutional Network (GCN) built in PyTorch Geometric (PyG). It is designed to predict optimal amino acid sequences by learning from the local 3D geometric environments of a protein backbone.

## The Pipeline

- **Data Parsing:** Hooks directly into **LigandMPNN's native parser**, leveraging their exact dictionaries and masking to ensure 1:1 equivalency, before extracting C-alpha coordinates for the graph.
- **Graph Construction:** Converts 3D space into a directed graph where nodes are residues and edges are defined by an 8.0 Ångström distance cutoff.
- **Message Passing:** A 2-layer GCN aggregates spatial features from neighboring nodes to learn the topography of localized structural pockets.
- **Sequence Prediction:** A linear classification layer outputs logits for the 20 standard amino acids.

## Biological Context & Limitations

*Note on Invariance:* This v1.0 architecture utilizes absolute (X, Y, Z) Cartesian coordinates as node features. Therefore, the model is not naturally translationally or rotationally invariant. Future iterations will map relative distances and dihedral angles to create a fully SE(3)-invariant graph, similar to state-of-the-art MPNNs.

## Roadmap: Preparing for Training

✅ **PyG Dataset & DataLoader Implementation**: Wrapper built (`utils/dataset.py`) to ingest `train.json` splits and batch dynamic spatial graphs asynchronously using PyG.
✅ **Masked Loss Function**: Defined loss that masks out padded residues using exact `chain_M` arrays inherited from LigandMPNN.
✅ **Training Routine**: Full `scripts/train.py` equipped with Adam optimization and checkpoint monitoring complete.

## Setup & Installation

To install the necessary components, ensure you have Anaconda or Python virtual environment setup:

```bash
# Clone the repository
git clone https://github.com/yourusername/Struct2Seq-GCN.git
cd Struct2Seq-GCN

# Install PyTorch
pip install torch torchvision torchaudio

# Install PyTorch Geometric (PyG) and helpers
pip install torch_geometric

# Note: LigandMPNN logic is used under the hood for 1:1 comparable data extraction.
```

## Running the Pipeline

### Training
To train the model natively using the matched LigandMPNN dataset splits:
```bash
python scripts/train.py \
    --json_train examples/LigandMPNN/training/train.json \
    --json_valid examples/LigandMPNN/training/valid.json \
    --pdb_dir examples/LigandMPNN/inputs/ \
    --batch_size 32 \
    --epochs 50
```

### Inference
To make predictions from custom backbones directly:
```bash
python scripts/run.py --pdb examples/LigandMPNN/outputs/autoregressive_score_wo_seq/1BC8_1.pt
```
# gaussian-autoencoder

Gaussian Splatting Compression via Learned Merging

## Overview

Voxel 단위로 Gaussian들을 압축하는 Transformer 기반 AutoEncoder

- **Input**: PLY 파일 (3D Gaussian Splatting 형식)
- **Output**: 압축된 PLY 파일 (더 적은 Gaussian 수)

## Installation

```bash
# Clone repository
cd gaussian-autoencoder

# Install package
pip install -e .

# For training (gsplat required)
pip install -e ".[train]"
```

## Usage

### Training

```bash
python scripts/train.py \
    --ply path/to/point_cloud.ply \
    --epochs 100 \
    --batch_size 32 \
    --save_dir ./checkpoints
```

### Compression

```bash
python scripts/compress.py \
    --ply path/to/input.ply \
    --checkpoint ./checkpoints/best.pt \
    --output path/to/compressed.ply
```

## Project Structure

```
gaussian-autoencoder/
├── gs_merge/                   # Main package
│   ├── model/                  # Neural network models
│   │   ├── encoder.py          # Positional encodings
│   │   ├── heads.py            # Gaussian output heads
│   │   └── ae.py               # AutoEncoder main model
│   ├── data/                   # Data handling
│   │   ├── gaussian.py         # GaussianData dataclass
│   │   ├── ply_io.py           # PLY load/save
│   │   ├── voxelizer.py        # Octree voxelization
│   │   └── dataset.py          # PyTorch Dataset
│   ├── loss/                   # Loss functions (UNIFIED)
│   │   ├── gmae_loss.py        # All losses + utilities
│   │   └── __init__.py         # - GMAELoss, ChamferLoss
│   ├── training/               # Training infrastructure
│   │   ├── trainer.py          # Main trainer
│   │   ├── callbacks.py        # TensorBoard, Checkpoint, Logging
│   │   └── schedulers.py       # Learning rate, Gumbel schedulers
│   ├── config/                 # Configuration
│   │   ├── args.py             # CLI arguments
│   │   └── loader.py           # YAML config loader
│   └── utils/                  # Utilities
│       ├── voxel_utils.py      # Voxel ↔ Gaussian conversion
│       └── debug_utils.py      # Debug statistics
├── scripts/                    # CLI scripts
│   ├── train.py                # Training script
│   ├── compress.py             # Compression script
│   └── utils/                  # Utility scripts
│       └── voxel_info.py       # Voxel inspection tool
├── configs/                    # YAML configurations
│   ├── default.yaml            # Default training config
│   ├── custom.yaml             # Custom experiments
│   └── chamfer.yaml            # Chamfer loss baseline
├── docs/                       # Documentation
│   ├── TENSORBOARD.md          # TensorBoard guide
│   ├── MULTI_PLY.md            # Multi-file training
│   ├── CHANGELOG_INPUT_STATS.md # Recent changes
│   └── how_to_install.txt      # Installation guide
├── tests/                      # Unit tests
├── checkpoints/                # Model checkpoints
└── outputs/                    # Output files
```

### Key Changes in Refactoring

✅ **Documentation organized** → All docs moved to `docs/`  
✅ **Loss module unified** → `gmae_loss.py` contains everything  
✅ **Scripts organized** → Utilities in `scripts/utils/`  
✅ **Cleaner structure** → Removed redundant files

## Architecture

### GaussianMergingAE

```
Input: [B, N, 59]  (N Gaussians per voxel)
  ↓
Transformer Encoder (4 layers)
  ↓
Learned Query Tokens [B, M, D]
  ↓
Transformer Decoder (4 layers)
  ↓
Output Heads
  ↓
Output: M Gaussians per voxel
```

### Gaussian Feature Format (59-dim)

| Index | Feature | Dim |
|-------|---------|-----|
| 0-2   | XYZ (normalized) | 3 |
| 3-6   | Rotation (quaternion) | 4 |
| 7-9   | Scale (log space) | 3 |
| 10    | Opacity (logit space) | 1 |
| 11-13 | SH DC | 3 |
| 14-58 | SH Rest | 45 |

## Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.0.0
- CUDA (recommended)
- gsplat ≥ 1.0.0 (for training)

## License

MIT License

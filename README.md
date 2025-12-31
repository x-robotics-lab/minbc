# MinBC - Minimal Behavior Cloning

A simple implementation for robot behavior cloning with support for Vanilla BC, Diffusion Policy, and Choice Policy.

## Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU

### Install Dependencies

```bash
pip install -r requirements.txt
```

Required packages:
- PyTorch >= 2.0.0
- torchvision >= 0.15.0
- diffusers >= 0.21.0
- tyro >= 0.5.0
- tensorboard >= 2.13.0

## Quick Start

```bash
# View all available parameters
python train.py train --help

# Basic training (without images)
python train.py train \
  --gpu 0 \
  --data.data-key joint_positions joint_velocities \
  --optim.batch-size 128 \
  --optim.num-epoch 300

# With images (DINOv3)
python train.py train \
  --gpu 0 \
  --data.data-key img joint_positions \
  --data.im-encoder DINOv3 \
  --data.dinov3-model-dir /path/to/dinov3 \
  --data.dinov3-weights-path /path/to/dinov3.ckpt

# Multi-GPU training
bash train.sh
```

## Key Configuration Options

| Parameter | Description |
|-----------|-------------|
| `--gpu` | GPU IDs (e.g., "0" or "0,1,2,3") |
| `--policy-type` | `bc` (Vanilla BC) or `dp` (Diffusion Policy) |
| `--dp.num-proposal` | `1` = Standard BC, `>1` = Choice Policy (CP) |
| `--data.data-key` | Data modalities: `img`, `joint_positions`, `joint_velocities`, `xhand_pos`, etc. |
| `--data.im-encoder` | Vision encoder: `DINOv3`, `DINO`, `CLIP`, `scratch` |
| `--optim.batch-size` | Batch size (default: 128) |
| `--optim.num-epoch` | Number of epochs (default: 30) |

Default values can be modified in `configs/base.py`.

## Data Format

```
data/
└── your_dataset/
    ├── train/
    │   ├── episode_000/
    │   │   ├── step_000.pkl
    │   │   └── ...
    │   └── ...
    └── test/
        └── ...
```

Each `.pkl` file should contain:
- `action`: numpy array `(action_dim,)` - **required**
- `joint_positions`, `joint_velocities`, `xhand_pos`, etc.: numpy arrays - optional
- `base_rgb`: numpy array `(H, W, 3)` - optional, for vision

## Training Output

Results saved to `outputs/<output_name>/`:
- `config.json` - Training configuration
- `model_best.ckpt` - Best model checkpoint
- `stats.pkl`, `norm.pkl` - Normalization parameters

Monitor with: `tensorboard --logdir outputs/`

## Acknowledgements

MinBC is modified from [HATO](https://github.com/toruowo/hato) DP part, which is a simplification of the original Diffusion Policy.

## Citations
```
@article{hsieh2025learning,
  title={Learning Dexterous Manipulation Skills from Imperfect Simulations},
  author={Hsieh, Elvis and Hsieh, Wen-Han and Wang, Yen-Jen and Lin, Toru and Malik, Jitendra and Sreenath, Koushil and Qi, Haozhi},
  journal={arXiv:2512.02011},
  year={2025}
}
```

## Questions?
Contact [Yen-Jen Wang](https://wangyenjen.github.io/) or [Haozhi Qi](https://haozhi.io/).

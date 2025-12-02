# MinBC - Minimal Behavior Cloning

A simple and efficient implementation for robot behavior cloning with support for both Vanilla BC and Diffusion Policy.

## Features

- 🚀 **Two Policy Types**: Vanilla BC and Diffusion Policy
- 🖼️ **Flexible Vision Encoders**: Support for DINOv3, DINO, CLIP, or train from scratch
- 🎯 **Multi-Modal Input**: RGB images, joint positions, velocities, tactile sensors, etc.
- ⚡ **Multi-GPU Training**: Efficient distributed training support
- 📊 **TensorBoard Logging**: Real-time training monitoring

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

### Option 1: Training Without Images (Fastest)

If your task doesn't require vision, use only proprioceptive data:

```bash
# Single GPU
python train.py train \
  --gpu 0 \
  --data.data-key joint_positions joint_velocities eef_speed xhand_pos \
  --optim.batch-size 128 \
  --optim.num-epoch 300
```

**Benefits**: No need to configure vision encoders, faster training, lower GPU memory.

### Option 2: Training With Images
Please refer to [dinov3](https://github.com/facebookresearch/dinov3) for the model and available checkpoints.
```bash
# Single GPU with DINOv3
python train.py train \
  --gpu 0 \
  --data.im-encoder DINOv3 \
  --data.dinov3-model-dir /path/to/dinov3 \
  --data.dinov3-weights-path /path/to/dinov3/dinov3.ckpt \
  --optim.batch-size 64 \
  --optim.num-epoch 300
```

```bash
# Or use DINO (auto-downloads from PyTorch Hub)
python train.py train \
  --gpu 0 \
  --data.im-encoder DINO \
  --optim.batch-size 64
```

### Option 3: Multi-GPU Training

```bash
# Edit train.sh to set your GPU IDs
vim train.sh

# Run multi-GPU training
bash train.sh
```

Or directly:

```bash
OMP_NUM_THREADS=1 torchrun --standalone --nnodes=1 --nproc_per_node=2 \
  train.py train \
  --gpu 0,1 \
  --multi-gpu \
  --optim.batch-size 256 \
  --optim.num-epoch 300
```

## Configuration

### View All Available Parameters

```bash
python train.py train --help
```

### Configuration Method

MinBC uses **command-line arguments** to configure training. There is only one configuration file: `configs/base.py`, which defines default values.

**Priority**: Command-line arguments > Default values in `configs/base.py`

### Common Configuration Options

#### Basic Training Settings
```bash
--gpu STR                    # GPU IDs (e.g., "0" or "0,1,2,3")
--multi-gpu                  # Enable multi-GPU training
--seed INT                   # Random seed (default: 0)
--optim.batch-size INT       # Batch size (default: 128)
--optim.num-epoch INT        # Number of epochs (default: 30)
--optim.learning-rate FLOAT  # Learning rate (default: 0.0002)
--output_name STR            # Experiment name
```

#### Data Configuration
```bash
--data.data-key [KEYS...]    # Data modalities to use
                             # Options: img, joint_positions, joint_velocities,
                             #          eef_speed, ee_pos_quat, xhand_pos, xhand_tactile

--data.im-encoder STR        # Vision encoder (only if using 'img')
                             # Options: DINOv3, DINO, CLIP, scratch

--data.dinov3-model-dir STR       # DINOv3 model directory (if using DINOv3)
--data.dinov3-weights-path STR    # DINOv3 weights path (if using DINOv3)
```

#### Policy Type
```bash
--policy-type STR            # Policy type: "bc" (Vanilla BC) or "dp" (Diffusion Policy)
```

#### Diffusion Policy Settings (if using policy-type=dp)
```bash
--dp.diffusion-iters INT     # Number of diffusion iterations (default: 100)
--dp.obs-horizon INT         # Observation horizon (default: 1)
--dp.act-horizon INT         # Action horizon (default: 8)
--dp.pre-horizon INT         # Prediction horizon (default: 16)
```

### How to Modify Configuration

#### Method 1: Command-Line Arguments (Recommended)

Override any parameter directly in the command:

```bash
python train.py train \
  --gpu 2 \
  --optim.batch-size 64 \
  --optim.learning-rate 0.0005 \
  --data.dinov3-model-dir /your/custom/path
```

#### Method 2: Edit Default Values

Modify `configs/base.py` to change default values:

```python
# configs/base.py

@dataclass(frozen=True)
class MinBCConfig:
    seed: int = 0
    gpu: str = '0'              # Change default GPU
    data_dir: str = 'data/'     # Change default data path
    ...

@dataclass(frozen=True)
class DataConfig:
    dinov3_model_dir: str = '/your/path/to/dinov3'  # Change default DINOv3 path
    ...
```

#### Method 3: Use Training Scripts

Create or modify training scripts like `train.sh`:

```bash
#!/bin/bash
timestamp=$(date +%Y%m%d_%H%M%S)

python train.py train \
  --gpu 0 \
  --optim.batch-size 128 \
  --optim.num-epoch 300 \
  --data.dinov3-model-dir /your/path \
  --output_name "exp-${timestamp}"
```

## Data Format

### Directory Structure

```
data/
└── your_dataset/
    ├── train/
    │   ├── episode_000/
    │   │   ├── step_000.pkl
    │   │   ├── step_001.pkl
    │   │   └── ...
    │   ├── episode_001/
    │   └── ...
    └── test/
        ├── episode_000/
        └── ...
```

### Data Requirements

Each `.pkl` file should contain a dictionary with the following keys:

#### Required Keys
- `action`: numpy array of shape `(action_dim,)` - Robot action at this timestep

#### Optional Keys (depending on `data.data-key` configuration)

**Proprioceptive Data:**
- `joint_positions`: numpy array of shape `(12,)` - Joint positions
- `joint_velocities`: numpy array of shape `(12,)` - Joint velocities
- `eef_speed`: numpy array of shape `(12,)` - End-effector speed
- `ee_pos_quat`: numpy array of shape `(12,)` - End-effector pose (position + quaternion)
- `xhand_pos`: numpy array of shape `(12,)` - Hand position
- `xhand_tactile`: numpy array of shape `(1800,)` - Tactile sensor data

**Visual Data (if using images):**
- `base_rgb`: numpy array of shape `(H, W, 3)` - RGB image (default: 240x320x3)
  - Values should be in range [0, 255], dtype: uint8 or uint16

### Data Format Example

```python
# Example pickle file content
import pickle
import numpy as np

data = {
    'action': np.array([...]),           # Shape: (24,)
    'joint_positions': np.array([...]),  # Shape: (12,)
    'joint_velocities': np.array([...]), # Shape: (12,)
    'base_rgb': np.array([...]),         # Shape: (240, 320, 3), uint8
}

with open('step_000.pkl', 'wb') as f:
    pickle.dump(data, f)
```

### Data Configuration

Specify which data modalities to use:

```bash
# With images
python train.py train \
  --data.data-key img joint_positions xhand_pos

# Without images (only proprioceptive)
python train.py train \
  --data.data-key joint_positions joint_velocities eef_speed
```

### Data Paths

Set data paths in command line:

```bash
python train.py train \
  --data-dir /path/to/your/data \
  --train-data your_dataset/train \
  --test-data your_dataset/test
```

Or modify defaults in `configs/base.py`:

```python
@dataclass(frozen=True)
class MinBCConfig:
    data_dir: str = '/path/to/your/data'
    train_data: str = 'your_dataset/train/'
    test_data: str = 'your_dataset/test/'
```

## Training Examples

### Example 1: Minimal Setup (No Images)

```bash
python train.py train \
  --gpu 0 \
  --data.data-key joint_positions \
  --optim.batch-size 128 \
  --optim.num-epoch 100
```

### Example 2: Multi-Modal (No Images)

```bash
python train.py train \
  --gpu 0 \
  --data.data-key joint_positions joint_velocities eef_speed xhand_pos \
  --optim.batch-size 128 \
  --optim.num-epoch 300
```

### Example 3: With Vision (DINOv3)

```bash
python train.py train \
  --gpu 0 \
  --data.data-key img joint_positions xhand_pos \
  --data.im-encoder DINOv3 \
  --data.dinov3-model-dir /path/to/dinov3 \
  --data.dinov3-weights-path /path/to/dinov3/dinov3.ckpt \
  --optim.batch-size 64 \
  --optim.num-epoch 300
```

### Example 4: Diffusion Policy

```bash
python train.py train \
  --gpu 0 \
  --policy-type dp \
  --data.data-key joint_positions joint_velocities \
  --dp.diffusion-iters 100 \
  --optim.batch-size 64 \
  --optim.num-epoch 300
```

### Example 5: Multi-GPU Training

```bash
OMP_NUM_THREADS=1 torchrun --standalone --nnodes=1 --nproc_per_node=4 \
  train.py train \
  --gpu 0,1,2,3 \
  --multi-gpu \
  --data.data-key img joint_positions xhand_pos \
  --data.im-encoder DINO \
  --optim.batch-size 256 \
  --optim.num-epoch 300
```

## Training Output

Training results are saved to `outputs/<output_name>/`:

```
outputs/bc-20251125_143022/
├── config.json              # Training configuration
├── model_last.ckpt          # Latest model checkpoint
├── model_best.ckpt          # Best model (lowest test loss)
├── stats.pkl                # Data statistics for normalization
├── norm.pkl                 # Normalization parameters
├── diff_*.patch             # Git diff at training time
└── events.out.tfevents.*    # TensorBoard logs
```

### Monitor Training

```bash
tensorboard --logdir outputs/
# Open browser to http://localhost:6006
```

## Troubleshooting

### Issue: DINOv3 Not Found

**Solution**: Either set the correct path or use a different encoder:

```bash
# Set correct path
python train.py train --data.dinov3-model-dir /correct/path

# Or use DINO (auto-downloads)
python train.py train --data.im-encoder DINO

# Or train without images
python train.py train --data.data-key joint_positions joint_velocities
```

### Issue: Out of GPU Memory

**Solutions**:
1. Reduce batch size: `--optim.batch-size 32`
2. Reduce prediction horizon: `--dp.pre-horizon 8`
3. Use fewer workers (modify `num_workers` in `dp/agent.py`)
4. Train without images if not needed

### Issue: Multi-GPU Training Hangs

**Solutions**:
1. Set `OMP_NUM_THREADS=1` before torchrun
2. Use `torchrun` instead of direct python execution
3. Check NCCL configuration

## Tips and Best Practices

1. **Start Simple**: Try training without images first to validate your pipeline
2. **Data Modalities**: Only include necessary data modalities for faster training
3. **Batch Size**: Adjust based on your GPU memory (64-128 for single GPU, 128-256 for multi-GPU)
4. **Vision Encoder**: Use DINO for ease (auto-downloads), DINOv3 for best performance (requires setup)
5. **Policy Type**: Use Vanilla BC for faster training, Diffusion Policy for better performance
6. **Monitoring**: Always check TensorBoard logs to ensure training is progressing

## Acknowledgements

MinBC is modified from [HATO](https://github.com/toruowo/hato) DP part, which is a simplification of the original Diffusion Policy.

## Citations
```
@article{hsieh2025learning,
  title={Learning Dexterous Manipulation Skills from Imperfect Simulations},
  author={Hsieh$^{*}$, Elvis and Hsieh$^{*}$, Wen-Han and Wang$^{*}$, Yen-Jen 
           and Lin, Toru and Malik, Jitendra and Sreenath$^{\dagger}$, Koushil 
           and Qi$^{\dagger}$, Haozhi},
  journal={arXiv preprint arXiv:2512.02011},
  year={2025},
  note={* Equal contribution (alphabetical order). † Equal advising.},
  url={https://arxiv.org/abs/2512.02011}
}

## Questions?
If you have any questions, please feel free to contact [Yen-Jen Wang](https://wangyenjen.github.io/) and [Haozhi Qi](https://haozhi.io/).

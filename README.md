<h1 align="center"> MinBC: Minimal Behavior Cloning </h1>

MinBC is a lightweight library for training behavior cloning (BC) policies for robot manipulation. It has been used in several of our research projects and supports multiple imitation learning algorithms with a unified training interface.

Currently supported policies:

- Standard (Vanilla) Behavior Cloning with action chunking 
- Diffusion Policy 
- Choice Policy (see our paper for details)

Disclaimer: This codebase is under active development. For the reference implementation used in the [dexscrew](https://dexscrew.github.io/) paper, please refer to [this version](https://github.com/x-robotics-lab/minbc/tree/b64c53f59ccb47230df16b3da31def8f16694557).

## Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU

### Install Dependencies

```bash
conda create -n minbc python=3.11
conda activate minbc
pip install -r requirements.txt
```

## Quick Start

This section demonstrates how to train a policy using a small humanoid manipulation dataset from our Choice Policy paper.

### Dataset Preparation

Download the `gr1_dishwasher_debug` dataset
```
gdown 1iQJRAC0CMp4P2UhDFx9o1tBmG4bGfGkF -O data/gr1_dishwasher_debug.zip
cd data
unzip gr1_dishwasher_debug.zip -d ./
rm gr1_dishwasher_debug.zip
cd ../
```

It contains 1 training trajectory and 1 test trajectory. The expected directory structure is:
```
data/
  gr1_dishwasher_debug/
    train/
    test/
```

### Training

Using this minimal data, you can run the following command for training and evaluation:
```
python train.py train --gpu 0 --seed 0 \
--optim.num_epoch 100 --optim.learning-rate 0.0005 --optim.batch-size 64 \
--data.pred-head-act \
--policy-type bc --dp.action_decoder cond_hourglass --data.base_action_dim 26 \
--data.data-key img gr1_upper hand_qpos --data.im-key head_image left_wrist_image right_wrist_image \
--data.im-encoder scratch \
--train_data gr1_dishwasher_debug/train --test_data gr1_dishwasher_debug/test \
--output_name gr1_dishwasher_debug/example_test/
```

In the above command, `num_epoch, learning_rate, batch_size` are standard optimization parameters. `train_data, test_data, output_name` are training path setup. `policy-type` specifies which imitation learning policy you want to use, use `policy-type bc` for naive behavior cloning, `policy-type dp` for diffusion policy, `--policy-type bc --dp.num-proposl 5` for choice policy.

Another useful setup parameter is `data-key` and `im-key`, as the name indicates, it specify what input modality is used. If your dataset has another input type, you should also manually add the vector dimension for the `INPUT_DIM` variable in `config/base.py`.

For multi-gpu training, you can use the following command 
`
OMP_NUM_THREADS=1 torchrun --standalone --nnodes=1 --nproc_per_node=4 learning/train.py train --gpu 0,1,2,3 --multi-gpu
`
For using pretrained image encoder one can use `--data.im-encoder DINOv3 --dp.encoder.im-encoder-frozen`

```
OMP_NUM_THREADS=1 torchrun --standalone --nnodes=1 --nproc_per_node=4 learning/train.py train --gpu 0,1,2,3 --multi-gpu --seed 3 \
--optim.num_epoch 100 --optim.learning-rate 0.0005 \
--data.pred-head-act --optim.batch-size 128 \
--policy-type bc --dp.action_decoder cond_hourglass \
--train_data 250815_dishwasher_nowaist/train \
--test_data 250815_dishwasher_nowaist/test \
--data.im-encoder DINOv3 --dp.encoder.im-encoder-frozen --data.data-key img gr1_upper hand_qpos \
--output_name 250815_dishwasher_nowaist/250827_all_bc_without_clip_3
```

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

## Citations
```
@article{hsieh2025learning,
  title={Learning Dexterous Manipulation Skills from Imperfect Simulations},
  author={Hsieh, Elvis and Hsieh, Wen-Han and Wang, Yen-Jen and Lin, Toru and Malik, Jitendra and Sreenath, Koushil and Qi, Haozhi},
  journal={arXiv:2512.02011},
  year={2025}
}

@article{qi2025coordinated,
  title={Coordinated Humanoid Manipulation with Choice Policies},
  author={Qi, Haozhi and Wang, Yen-Jen and Lin, Toru and Yi, Brent and Ma, Yi and Sreenath, Koushil and Malik, Jitendra},
  journal={arXiv:2512.25072},
  year={2025}
}
```

## Questions?
Contact [Yen-Jen Wang](https://wangyenjen.github.io/) or [Haozhi Qi](https://haozhi.io/).


## Acknowledgements

MinBC is modified from [HATO](https://github.com/toruowo/hato) DP part, which is a simplification of the original [Diffusion Policy](https://github.com/real-stanford/diffusion_policy).

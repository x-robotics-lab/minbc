import os
import tyro
import json
import torch
import random
import numpy as np
import dataclasses

from dp.agent import Agent
from configs.base import MinBCConfig
from utils.misc import set_seed, git_hash, git_diff_config


if __name__ == "__main__":
    configs = {
        "debug": ("formal training config", MinBCConfig()),
        "train": ("formal training config", MinBCConfig()),
    }
    config = tyro.extras.overridable_config_cli(configs)

    if config.multi_gpu:
        rank = int(os.getenv("LOCAL_RANK", "0"))
        # torchrun --standalone --nnodes=1 --nproc_per_node=2 train.py
        gpu_list = config.gpu.split(',')
        gpu_id = gpu_list[rank]
        device = f'cuda:{gpu_id}'
        # sets seed. if seed is -1 will pick a random one
        set_seed(config.seed + rank)
    else:
        rank = 0
        set_seed(config.seed)
        device = f'cuda:{config.gpu}'

    torch.cuda.set_device(device)
    # process output directory
    model_path = os.path.join(
        config.output_dir, config.output_name,
    )
    print(f"Saving to model path {model_path}")
    if not os.path.exists(model_path):
        os.makedirs(model_path, exist_ok=True)
    # saving config
    config_dict = dataclasses.asdict(config)
    with open(os.path.join(model_path, "config.json"), "w") as json_file:
        json.dump(config_dict, json_file, indent=4)
    # saving gitdiff
    if rank == 0:
        print(git_diff_config('./'))
        os.system(f'git diff HEAD > {model_path}/diff_{git_hash()}.patch')

    # Check if images are needed
    load_img = 'img' in config.data.data_key
    
    agent = Agent(
        config, clip_far=False,
        num_diffusion_iters=config.dp.diffusion_iters,
        load_img=load_img, num_workers=8,
        # binarize_touch=config.dp.encoder.hand_touch_input_binary,
        dit=False, device=device,
    )
    if agent.stats is not None:
        agent.save_stats(model_path)

    agent.train(
        batch_size=config.optim.batch_size,
        num_epoch=config.optim.num_epoch,
        save_path=model_path, save_freq=5, eval_freq=5,
        wandb_logger=None
    )

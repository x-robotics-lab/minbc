import time

import tyro
import numpy as np
from dataclasses import dataclass
from agents.diffusion.diffusion_agent_sync import DiffusionAgent
from agents.diffusion.diffusion_agent_client import DiffusionAgentClient

from camera_node import ZMQClientCamera

@dataclass
class EvalConfig:
    ckpt_path: str
    use_async: bool = False
    num_diffusion_iters: int = 5


def main(config):
    frequency = 20  # Target frequency in Hz
    interval = 1 / frequency  # Time per cycle in seconds

    if config.use_async:
        dp_agent = DiffusionAgentClient(ckpt_path=config.ckpt_path, temporal_ensemble_mode="avg")
    else:
        dp_agent = DiffusionAgent(ckpt_path=config.ckpt_path)

    camera_client = Camera()
    # env = (rate=100, gripper_type="ability_haGR1nd")
    obs_dict = env.reset()

    from utils.utils import MovingAverageQueue
    action_queue = MovingAverageQueue(20, 26, 0.7)

    # sometimes ability hand is not working, this is only for monitor
    left_hand_norm, right_hand_norm = 0, 0
    left_touch_norm, right_touch_norm = 0, 0

    for _ in range(2000):
        start_time = time.time()
        head_image, head_depth, rw_image, rw_depth, lw_image, lw_depth = camera_client.get_image()
        obs_dict = {
            "states": obs_dict["joint_pos"],
            "joint_vel": obs_dict["joint_vel"],
            "base_imu": obs_dict["base_imu"],
            "base_angvel": obs_dict["base_angvel"],
            "hand_states": np.array(env.hand_joints),  # a list
            "hand_touch": np.array(env.hand_touch),
            "timestamp": time.time(),
            "head_image": head_image,
            "head_depth": head_depth,
            "right_wrist_image": rw_image,
            "right_wrist_depth": rw_depth,
            "left_wrist_image": lw_image,
            "left_wrist_depth": lw_depth,
        }
        rgb = [obs_dict[k] for k in ['head_image', 'right_wrist_image', 'left_wrist_image']]
        obs_dict["base_rgb"] = np.stack(rgb, axis=0)

        action = dp_agent.act(obs_dict)
        assert not dp_agent.config.data.pred_waist_act
        if dp_agent.config.data.pred_head_act:
            head_actions = action[-3:]
            action = action[:-3]
        else:
            head_actions = np.array([0.6, 0.0, 0.0])

        if dp_agent.config.data.pred_waist_act:
            raise NotImplementedError
        else:
            waist_actions = np.array([0.0, -0.1, 0.0])

        if _ == 0:
            for k in range(20):
                action_queue.add(action)
        copied_action = action.copy()
        action = action_queue.add(action)
        action[7:13] = copied_action[7:13]
        action[20:] = copied_action[20:]

        while True:
            obs_dict = env.step(action, head_actions, waist_actions)
            elapsed_time = time.time() - start_time
            if elapsed_time > interval:
                break


if __name__ == "__main__":
    main(tyro.cli(EvalConfig))

import time
from dataclasses import dataclass
from glob import glob
from envs.gr1 import GR1
import tyro
import numpy as np
from agents.diffusion.diffusion_agent_sync import DiffusionAgent
from agents.diffusion.diffusion_agent_client import DiffusionAgentClient
from dataset.data_processing import iterate


@dataclass
class EvalConfig:
    ckpt_path: str
    data_dir: str
    use_async: bool = False
    hardware_eval: bool = False


def main(config):
    frequency = 20  # Target frequency in Hz
    interval = 1 / frequency  # Time per cycle in seconds

    if config.hardware_eval:
        env = GR1(
            rate=100,
            gripper_type="ability_hand",
        )

    if config.use_async:
        dp_agent = DiffusionAgentClient(ckpt_path=config.ckpt_path, temporal_ensemble_mode="avg")
    else:
        dp_agent = DiffusionAgent(ckpt_path=config.ckpt_path)

    test_trajs = sorted(glob(config.data_dir + '*/'))

    all_hand_error = 0
    all_arm_error = 0
    cnt = 0
    for data_dir in test_trajs:
        data = iterate(data_dir, dp_agent.config)
        from utils.utils import MovingAverageQueue
        action_queue = MovingAverageQueue(20, 26, 0.7)
        for i, obs in enumerate(data):
            # supposed to be 10Hz
            # if config.use_async:
            #     dp_agent.compile_inference(
            #         obs, num_diffusion_iters=30
            #     )
            start_time = time.time()
            action = dp_agent.act(obs)
            gt_action = obs["actions"]

            hand_error = (((action[7:13] - gt_action[7:13]) ** 2).mean() + ((action[20:26] - gt_action[20:26]) ** 2).mean()) / 2
            arm_error = (((action[0:7] - gt_action[0:7]) ** 2).mean() + ((action[13:20] - gt_action[13:20]) ** 2).mean()) / 2
            all_hand_error += hand_error
            all_arm_error += arm_error
            cnt += 1
            print(f'{cnt}/{len(data)} | Average Hand Error {all_hand_error / cnt:.3f} | Average Arm Error {all_arm_error / cnt:.3f}')

            if dp_agent.config.data.pred_head_act:
                head_actions = action[-3:]
                action = action[:-3]
            else:
                head_actions = np.array([0.6, 0.0, 0.0])

            if dp_agent.config.data.pred_waist_act:
                raise NotImplementedError
            else:
                waist_actions = np.array([0.0, -0.1, 0.0])

            if i == 0:
                for k in range(20):
                    action_queue.add(action)

            copied_action = action.copy()
            action = action_queue.add(action)
            action[7:13] = copied_action[7:13]
            action[20:] = copied_action[20:]

            if config.hardware_eval:
                while True:
                    env.step(action, head_actions, waist_actions)
                    elapsed_time = time.time() - start_time
                    if elapsed_time > interval:
                        break

        exit()


if __name__ == "__main__":
    # python tools/deploy/openloop.py \
    # --data-dir data/250725_dishwasher_nowaist_nohandeye/ \
    # --ckpt-path outputs/250725_dishwasher_nowaist_nohandeye/250801_dp/model_last.ckpt
    main(tyro.cli(EvalConfig))

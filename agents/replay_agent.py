"""
This file implements replay agent. It takes a human collected trajectory as input.
It will either replay the actions, or use a network run on the observations.
It is open-loop, only for debugging purpose.
"""
import pickle
import numpy as np
from typing import Literal, Dict
from agents.agent import Agent
from pathlib import Path


class ReplayAgent(Agent):
    def __init__(
        self, robot_type: Literal["GR1"], replay_type: Literal["Action"],
        traj_path: Path,
    ):
        self.robot_type = robot_type
        self.replay_type = replay_type
        self.trajs = sorted(list(traj_path.glob("*.pkl")))
        self.counter = 0

    def act(self, obs: Dict) -> np.ndarray:
        if self.replay_type == "Action":
            return self._act_action_replay()
        else:
            raise NotImplementedError

    def _act_action_replay(self):
        # Open the file in binary read mode and load the data
        pkl_path = self.trajs[self.counter]
        with pkl_path.open("rb") as file:
            data = pickle.load(file)
        # left arm, left hand; right arm, right hand
        actions = data["actions"]
        arm_hand_actions = actions
        print('replay:', arm_hand_actions)
        head_actions = np.zeros(3)
        self.counter += 1
        return arm_hand_actions, head_actions

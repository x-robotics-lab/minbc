import os
import json
import time
import torch
import pickle
import collections
import numpy as np
from typing import Any, Dict
from dataclasses import fields
from utils.obs import get_observation

from dp.agent import Agent as DPAgent
from configs.base import MinBCConfig

from agents.zmq_server_client import ZMQInferenceServer


def merge_dataclass_and_dict(dataclass_obj: Any, update_dict: dict) -> Any:
    """
    Recursively merge a dataclass and a dictionary, returning a new dataclass instance.
    """
    if not hasattr(dataclass_obj, "__dataclass_fields__"):
        raise ValueError("Provided object is not a dataclass instance.")

    dataclass_fields = fields(dataclass_obj)
    dataclass_values = {field.name: getattr(dataclass_obj, field.name) for field in dataclass_fields}
    merged_values = {}

    for key, value in dataclass_values.items():
        if key in update_dict:
            if hasattr(value, "__dataclass_fields__"):  # If the field is a nested dataclass
                # Recursively merge the nested dataclass
                merged_values[key] = merge_dataclass_and_dict(value, update_dict[key])
            else:
                # Use the new value from the dictionary
                merged_values[key] = update_dict[key]
        else:
            # Keep the existing value if not in update_dict
            merged_values[key] = value

    # Create a new dataclass instance with merged values
    return dataclass_obj.__class__(**merged_values)


class DiffusionAgentServer(ZMQInferenceServer):
    def __init__(
        self,
        ckpt_path,
        binarize_finger_action=False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        dp_args = os.path.join(os.path.dirname(ckpt_path), "config.json")
        with open(dp_args, "r") as f:
            dp_args = json.load(f)
        dp_args['multi_gpu'] = False
        dp_args['dp']['diffusion_iters'] = 20
        config = merge_dataclass_and_dict(MinBCConfig(), dp_args)
        torch.cuda.set_device(0)
        self.config = config
        self.dp = DPAgent(
            config,
            clip_far=False,
            num_diffusion_iters=config.dp.diffusion_iters,
            load_img=True,
            num_workers=8,
            device="cuda:0",
            # binarize_touch=config.dp.encoder.hand_touch_input_binary,
            dit=False,
        )
        self.data_key = config.data.data_key
        self.image_height = config.data.im_height
        self.image_width = config.data.im_width
        self.dp_args = dp_args
        self.dp.load(ckpt_path)
        self.obsque = collections.deque(maxlen=config.dp.obs_horizon)
        self.action_queue = collections.deque(maxlen=config.dp.act_horizon)
        self.max_length = 100
        self.count = 0
        self.num_diffusion_iters = config.dp.diffusion_iters
        self.trigger_state = {"l": True, "r": True}

    def compile_inference(self, precision="high"):
        message = self._socket.recv()
        start_time = time.time()
        state_dict = pickle.loads(message)
        self.num_diffusion_iters = state_dict["num_diffusion_iters"]
        example_obs = state_dict["example_obs"]
        print(
            f"received compilation request: # diff iters = {state_dict['num_diffusion_iters']}"
        )

        torch.set_float32_matmul_precision(precision)
        self.dp.policy.forward = torch.compile(torch.no_grad(self.dp.policy.forward))

        for i in range(25):  # burn in
            self.act(example_obs)
        print("success, compile time: " + str(time.time() - start_time))
        self._socket.send_string("success")

    def infer(self, obs: Dict[str, Any]) -> np.ndarray:
        return self.dp.predict([obs], num_diffusion_iters=self.num_diffusion_iters)

    def act(self, obs: Dict[str, Any]) -> np.ndarray:
        # dict_keys(['cache_name', 'actions', 'states', 'hand_states', 'base_rgb', 'mask_path', 'file_path'])
        obs = get_observation(
            self.data_key, [obs], im_c=3, im_h=self.image_height, im_w=self.image_width,
            clip_far=False, threshold=10000, load_img=True
        )
        if "img" in obs:
            obs["img"] = self.dp.eval_transform(obs["img"].squeeze(0))
        return self.infer(obs)

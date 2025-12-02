import os
import json
import pickle
import collections
import numpy as np
from typing import Any, Dict
from dataclasses import fields
from ..zmq_server_client import ZMQInferenceClient, DEFAULT_INFERENCE_PORT
from configs.base import MinBCConfig


LEFT_HAND_IDX = list(range(7, 13))
RIGHT_HAND_IDX = list(range(20, 26))


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


class DiffusionAgentClient(ZMQInferenceClient):
    def __init__(
        self,
        ckpt_path,
        binarize_finger_action=False,
        port=DEFAULT_INFERENCE_PORT,
        host="127.0.0.1",
        temporal_ensemble_mode="new",
        temporal_ensemble_act_tau=0.5,
    ):
        super().__init__(
            port=port,
            host=host,
            ensemble_mode=temporal_ensemble_mode,
            act_tau=temporal_ensemble_act_tau,
        )

        dp_args = os.path.join(os.path.dirname(ckpt_path), "config.json")
        with open(dp_args, "r") as f:
            dp_args = json.load(f)
        dp_args['multi_gpu'] = False
        dp_args['dp']['diffusion_iters'] = 20
        config = merge_dataclass_and_dict(MinBCConfig(), dp_args)
        self.config = config
        self.dp_args = dp_args
        self.obsque = collections.deque(maxlen=config.dp.obs_horizon)
        # self.dp.load(ckpt_path)
        self.action_queue = collections.deque(maxlen=config.dp.act_horizon)
        self.max_length = 100
        self.count = 0

        self.num_diffusion_iters = config.dp.diffusion_iters
        self.trigger_state = {"l": True, "r": True}

    def compile_inference(self, example_obs, num_diffusion_iters):
        message = pickle.dumps(
            {"example_obs": example_obs, "num_diffusion_iters": num_diffusion_iters}
        )
        self._socket.send(message)

        message = self._socket.recv()
        # assert message == b"success", message

    def act(self, obs: Dict[str, Any]) -> np.ndarray:
        act = super().act(obs)
        return act

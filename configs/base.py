from dataclasses import dataclass
from typing import Literal, Tuple, Dict


@dataclass(frozen=True)
class EncoderConfig:
    # each modality defines own encoder and output
    joint_positions_input_dim: int = 12
    joint_positions_net_dim: Tuple[int, ...] = (256, 256, 64)

    joint_velocities_input_dim: int = 12
    joint_velocities_net_dim: Tuple[int, ...] = (256, 256, 64)

    eef_speed_input_dim: int = 12
    eef_speed_net_dim: Tuple[int, ...] = (256, 256, 64)

    ee_pos_quat_input_dim: int = 12
    ee_pos_quat_net_dim: Tuple[int, ...] = (256, 256, 64)

    xhand_pos_input_dim: int = 12
    xhand_pos_net_dim: Tuple[int, ...] = (256, 256, 64)

    xhand_tactile_input_dim: int = 1800
    xhand_tactile_net_dim: Tuple[int, ...] = (256, 256, 64)

    # image encoder and input
    im_net_output_dim: int = 32
    im_encoder_frozen: bool = False
    im_encoder_reduce_lr: bool = False


@dataclass(frozen=True)
class DPConfig:
    encoder: EncoderConfig = EncoderConfig()
    obs_horizon: int = 1
    act_horizon: int = 8
    pre_horizon: int = 16
    diffusion_iters: int = 100
    diffusion_method: Literal["ddim", "ddpm"] = "ddpm"
    num_proposal: int = 5
    action_decoder: Literal["mlp", "hourglass", "cond_hourglass"] = "mlp"
    last_dropout: float = 0.0
    cond_dropout: float = 0.0
    clip_actions: bool = False
    clip_action_scores: bool = False
    clip_score_loss_max: float = 10.0
    clip_score_loss: bool = False


@dataclass(frozen=True)
class OptimConfig:
    batch_size: int = 128
    num_epoch: int = 30
    weight_decay: float = 0.01
    learning_rate: float = 0.0002


@dataclass(frozen=True)
class DataConfig:
    INPUT_DIM = {
        "joint_positions": 12,
        "joint_velocities": 12,
        "eef_speed": 12,
        "ee_pos_quat": 12,
        "xhand_pos": 12,
        "xhand_tactile": 1800,
        "gr1_upper": 20,  # fourier GR-1, 3 head, 3 waist, 2 arms
        "hand_qpos": 12,  # 2 ability hands
    }
    # Data modalities to use
    # Example with image: ("img", "joint_positions", "xhand_pos")
    # Example without image: ("joint_positions", "joint_velocities", "eef_speed", "ee_pos_quat", "xhand_pos", "xhand_tactile")
    # data_key: Tuple[str, ...] = ("img", "joint_positions", "xhand_pos")
    data_key: Tuple[str, ...] = (
        "joint_positions",
        "xhand_pos",
        "xhand_tactile",
    )
    # Image encoder (only needed if "img" in data_key)
    im_encoder: Literal["scratch", "DINO", "CLIP", "DINOv3"] = "DINOv3"
    im_key: Tuple[str, ...] = ("base_rgb")
    im_channel: int = 3
    im_height: int = 240
    im_width: int = 320
    
    # DINOv3 model paths (only needed if im_encoder == "DINOv3" and "img" in data_key)
    # Set to empty string if not using DINOv3
    dinov3_model_dir: str = '/home/wangyenjen/dinov3'
    dinov3_weights_path: str = '/home/wangyenjen/dinov3/dinov3.ckpt'

    # Output action space
    pred_head_act: bool = False
    pred_waist_act: bool = False
    base_action_dim: int = 24

    @property
    def data_dim(self) -> Dict[str, int]:
        data_dim = {}
        for key in self.data_key:
            if key != "img":
                data_dim[key] = self.INPUT_DIM[key]
        return data_dim

    @property
    def action_dim(self) -> int:
        dim = self.base_action_dim
        if self.pred_head_act:
            dim += 3
        if self.pred_waist_act:
            dim += 3
        return dim


@dataclass(frozen=True)
class MinBCConfig:
    seed: int = 0
    gpu: str = '3'
    data_dir: str = 'data/'
    multi_gpu: bool = False
    output_dir: str = 'outputs/'
    train_data: str = 'screw_driver_1104_modified/train/'
    test_data: str = 'screw_driver_1104_modified/test/'
    output_name: str = 'debug'
    policy_type: Literal["dp", "bc"] = "bc"
    dp: DPConfig = DPConfig()
    optim: OptimConfig = OptimConfig()
    data: DataConfig = DataConfig()

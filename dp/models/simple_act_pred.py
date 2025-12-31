import math

import torch
from torch import nn
from configs.base import MinBCConfig
from dp.models.vision.resnet import ResnetEncoder
from dp.models.block import TemporalTransformer
from dp.models.exp.action_decoder import MLPDecoder, HourglassDecoder, CondHourglassDecoder


class MLP(nn.Module):
    def __init__(self, units, input_size):
        super(MLP, self).__init__()
        layers = []
        for output_size in units:
            layers.append(nn.Linear(input_size, output_size))
            layers.append(nn.ReLU())
            input_size = output_size
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class StateEncoder(nn.Module):
    def __init__(self, input_dim, net_dim):
        super(StateEncoder, self).__init__()
        self.mlp = MLP(net_dim, input_dim)

    def forward(self, x):
        return self.mlp(x)


# Diffusion policy


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class VanillaBC(nn.Module):
    def __init__(
        self, config: MinBCConfig, input_dim, num_proposals=3, device="cpu"
    ):
        super().__init__()
        self.obs_horizon = config.dp.obs_horizon
        self.pre_horizon = config.dp.pre_horizon
        self.action_dim = input_dim
        self.num_proposal = num_proposals
        self.clip_actions = config.dp.clip_actions
        self.clip_action_scores = config.dp.clip_action_scores
        self.device = device
        self.data_key = config.data.data_key
        self.image_num = len(config.data.im_key)
        possible_input_type = ["img", "joint_positions","joint_velocities", "eef_speed", "ee_pos_quat", "xhand_pos", "xhand_tactile"]
        self.im_encoder = config.data.im_encoder
        self.policy_input_types = [
            rt for rt in possible_input_type if rt in self.data_key
        ]

        # temporal aggregation option
        # naive implementation is concat timestep features
        self.temporal_aggregation_function = "concat"
        if self.temporal_aggregation_function == "transformer":
            self.temporal_aggregation_net = TemporalTransformer(
                embedding_dim=160, n_head=4, depth=2, dim_feedforward=128,
                output_dim=128, use_pe=True, dense_output=False, dropout=0.0,
            )

        self.encoders = nn.ModuleDict({})
        encoder_config = config.dp.encoder
        obs_dim = 0
        if "img" in self.data_key:
            if config.data.im_encoder == 'scratch':
                image_encoder = nn.ModuleList(
                    [
                        ResnetEncoder(encoder_config.im_net_output_dim,
                                      config.data.im_channel)
                        for i in range(self.image_num)
                    ]
                )
            elif config.data.im_encoder == 'DINO':
                model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
                proj = nn.ModuleList([
                    nn.Linear(384, encoder_config.im_net_output_dim) for i in
                    range(self.image_num)
                ])
                image_encoder = nn.ModuleList([
                    model for i in range(self.image_num)
                ])
                self.encoders["img_encoder_proj"] = proj
            elif config.data.im_encoder == 'DINOv3':
                model = torch.hub.load(
                    config.data.dinov3_model_dir, 'dinov3_vits16',
                    source='local', weights=config.data.dinov3_weights_path
                )
                proj = nn.ModuleList([
                    nn.Linear(384, encoder_config.im_net_output_dim) for i in
                    range(self.image_num)
		])
                image_encoder = nn.ModuleList([
                    model for i in range(self.image_num)
                ])
                self.encoders["img_encoder_proj"] = proj
            elif config.data.im_encoder == 'CLIP':
                import clip
                model, _ = clip.load("ViT-B/32", device=device)
                model = model.float()
                proj = nn.ModuleList([
                    nn.Linear(512, encoder_config.im_net_output_dim) for i in
                    range(self.image_num)
                ])
                image_encoder = nn.ModuleList([
                    model for i in range(self.image_num)
                ])
                self.encoders["img_encoder_proj"] = proj
            else:
                raise NotImplementedError
            image_dim = encoder_config.im_net_output_dim * self.image_num
            self.encoders["img_encoder"] = image_encoder
            obs_dim += image_dim
        if 'joint_positions' in self.data_key:
            self.encoders["joint_positions_encoder"] = StateEncoder(
                encoder_config.joint_positions_input_dim,
                encoder_config.joint_positions_net_dim,
            )
            obs_dim += encoder_config.joint_positions_net_dim[-1]
        if 'joint_velocities' in self.data_key:
            self.encoders["joint_velocities_encoder"] = StateEncoder(
                encoder_config.joint_velocities_input_dim,
                encoder_config.joint_velocities_net_dim,
            )
            obs_dim += encoder_config.joint_velocities_net_dim[-1]
        if 'eef_speed' in self.data_key:
            self.encoders["eef_speed_encoder"] = StateEncoder(
                encoder_config.eef_speed_input_dim,
                encoder_config.eef_speed_net_dim,
            )
            obs_dim += encoder_config.eef_speed_net_dim[-1]
        if 'ee_pos_quat' in self.data_key:
            self.encoders["ee_pos_quat_encoder"] = StateEncoder(
                encoder_config.ee_pos_quat_input_dim,
                encoder_config.ee_pos_quat_net_dim,
            )
            obs_dim += encoder_config.ee_pos_quat_net_dim[-1]
        if 'xhand_pos' in self.data_key:
            self.encoders["xhand_pos_encoder"] = StateEncoder(
                encoder_config.xhand_pos_input_dim,
                encoder_config.xhand_pos_net_dim,
            )
            obs_dim += encoder_config.xhand_pos_net_dim[-1]
        if 'xhand_tactile' in self.data_key:
            self.encoders["xhand_tactile_encoder"] = StateEncoder(
                encoder_config.xhand_tactile_input_dim,
                encoder_config.xhand_tactile_net_dim,
            )
            obs_dim += encoder_config.xhand_tactile_net_dim[-1]

        global_cond_dim = obs_dim * self.obs_horizon
        if self.temporal_aggregation_function == "transformer":
            global_cond_dim = 128

        self.decoder_type = config.dp.action_decoder
        if self.decoder_type == "mlp":
            self.act_decoder = MLPDecoder(
                global_cond_dim, self.action_dim,
                self.pre_horizon, self.num_proposal
            )
        elif self.decoder_type == "hourglass":
            self.act_decoder = HourglassDecoder(
                global_cond_dim, self.action_dim,
                self.pre_horizon, self.num_proposal,
                last_dropout=config.dp.last_dropout,
                cond_dropout=config.dp.cond_dropout,
            )
        elif self.decoder_type == "cond_hourglass":
            self.act_decoder = CondHourglassDecoder(
                global_cond_dim, self.action_dim,
                self.pre_horizon, self.num_proposal,
                last_dropout=config.dp.last_dropout,
                cond_dropout=config.dp.cond_dropout,
            )

        all_params = sum(p.numel() for p in self.parameters())
        print(f"Model Parameters: {all_params // 1e6:.1f}M")

    def forward_encoder(self, data):
        features = []
        for data_key in self.policy_input_types:
            nsample = data[data_key][:, :self.obs_horizon].to(self.device)
            if data_key == "img":
                images = [
                    nsample[:, :, i] for i in range(nsample.shape[2])
                ]  # [B, obs_horizon, M, C, H, W]
                if self.im_encoder == 'CLIP':
                    image_features = [
                        self.encoders[f"{data_key}_encoder"][i].encode_image(
                            image.flatten(end_dim=1)
                        )
                        for i, image in enumerate(images)
                    ]
                    image_features = [
                        self.encoders["img_encoder_proj"][i](im_feat.float())
                        for i, im_feat in enumerate(image_features)
                    ]
                elif self.im_encoder == 'DINO' or self.im_encoder == 'DINOv3':
                    image_features = [
                        self.encoders[f"{data_key}_encoder"][i](
                            image.flatten(end_dim=1)
                        )
                        for i, image in enumerate(images)
                    ]
                    image_features = [
                        self.encoders["img_encoder_proj"][i](im_feat.float())
                        for i, im_feat in enumerate(image_features)
                    ]
                else:
                    image_features = [
                        self.encoders[f"{data_key}_encoder"][i](
                            image.flatten(end_dim=1)
                        )
                        for i, image in enumerate(images)
                    ]
                image_features = torch.stack(image_features, dim=2)
                image_features = image_features.reshape(
                    *nsample.shape[:2], -1
                )
                features.append(image_features)
            elif data_key == "gr1_loco":
                nsample = torch.cat([
                    data["gr1_loco"][:, :self.obs_horizon].to(self.device),
                    data["last_actions"][:, :self.obs_horizon].to(self.device),
                    data["joint_vel"][:, :self.obs_horizon].to(self.device),
                    data["base_angvel"][:, :self.obs_horizon].to(self.device),
                    data["base_imu"][:, :self.obs_horizon].to(self.device) * 10,
                    data["wave"][:, :self.obs_horizon].to(self.device),
                ], dim=2)
                nfeat = self.encoders[f"{data_key}_encoder"](
                    nsample.flatten(end_dim=1)
                )
                nfeat = nfeat.reshape(*nsample.shape[:2], -1)
                features.append(nfeat)
            else:
                nfeat = self.encoders[f"{data_key}_encoder"](
                    nsample.flatten(end_dim=1)
                )
                nfeat = nfeat.reshape(*nsample.shape[:2], -1)
                features.append(nfeat)
                # (B, obs_horizon, obs_dim)
        obs_features = torch.cat(features, dim=-1)
        # (batch x t x dim)
        if self.temporal_aggregation_function == "transformer":
            obs_cond = self.temporal_aggregation_net(obs_features)
        else:
            # (B, obs_horizon * obs_dim)
            obs_cond = obs_features.flatten(start_dim=1)
        return obs_cond

    def bc_model(self, x):
        # (B, feature_dim)
        act, score = self.act_decoder(x)
        if self.clip_actions:
            act = torch.clamp(act, min=-1, max=1)
        if self.clip_action_scores:
            score = torch.clamp(score, min=0)
        return act, score

    def forward(self, x):
        x = self.forward_encoder(x)
        x = self.bc_model(x)
        return x

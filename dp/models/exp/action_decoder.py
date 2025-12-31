import torch
from torch import nn
from .block import ResidualBlock1D, Upsample1d, Downsample1d, ConditionalResidualBlock1D


class MLPDecoder(nn.Module):
    def __init__(self, in_channels, act_dim, pre_horizon, num_proposals, feature_dim=1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channels, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
        )
        self.action_pred = nn.Linear(256, act_dim * pre_horizon * num_proposals)
        if num_proposals > 1:
            self.score_pred = nn.Linear(256, num_proposals)
        else:
            self.score_pred = nn.Identity()
        self.num_proposals = num_proposals
        self.pre_horizon = pre_horizon
        self.action_dim = act_dim

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.net(x)
        act = self.action_pred(x)
        score = self.score_pred(x)
        if self.num_proposals > 1:
            act = act.reshape(
                batch_size, self.num_proposals, self.pre_horizon, self.action_dim
            )
        else:
            act = act.reshape(
                batch_size, self.pre_horizon, self.action_dim
            )
        return act, score


class HourglassDecoder(nn.Module):
    def __init__(self, in_channels, act_dim, pre_horizon, num_proposals, feature_dim=1024,
                 last_dropout=0.0, cond_dropout=0.0):
        super().__init__()
        kernel_size = 5
        n_groups = 8
        down_dims = [256, 512, 1024]
        all_dims = [in_channels] + list(down_dims)
        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        self.ffn = nn.Linear(in_channels, in_channels * pre_horizon)
        self.pre_horizon = pre_horizon
        self.action_dim = act_dim
        self.num_proposals = num_proposals
        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([
            ResidualBlock1D(
                mid_dim, mid_dim, kernel_size=kernel_size, n_groups=n_groups,
            ),
            ResidualBlock1D(
                mid_dim, mid_dim, kernel_size=kernel_size, n_groups=n_groups,
            ),
        ])

        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(
                nn.ModuleList([
                    ResidualBlock1D(
                        dim_in, dim_out, kernel_size=kernel_size, n_groups=n_groups,
                    ),
                    ResidualBlock1D(
                        dim_out, dim_out, kernel_size=kernel_size, n_groups=n_groups,
                    ),
                    Downsample1d(dim_out) if not is_last else nn.Identity(),
                ])
            )

        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(
                nn.ModuleList([
                    ResidualBlock1D(
                        dim_out * 2, dim_in, kernel_size=kernel_size, n_groups=n_groups,
                    ),
                    ResidualBlock1D(
                        dim_in, dim_in, kernel_size=kernel_size, n_groups=n_groups,
                    ),
                    Upsample1d(dim_in) if not is_last else nn.Identity(),
                ])
            )
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.score_pred = nn.Linear(256 * self.pre_horizon, self.num_proposals)
        self.act_pred = nn.Linear(256, self.num_proposals * act_dim)
        if last_dropout > 0:
            self.last_dropout = nn.Dropout(p=last_dropout, inplace=True)
        else:
            self.last_dropout = nn.Identity()
        if cond_dropout > 0:
            self.cond_dropout = nn.Dropout(p=cond_dropout, inplace=True)
        else:
            self.cond_dropout = nn.Identity()

    def forward(self, x):
        # x: batch x feature_dim
        x = self.cond_dropout(x)
        batch_size, feat_dim = x.shape
        x = self.ffn(x)
        x = x.reshape(batch_size, feat_dim, self.pre_horizon)
        # x: (batch x #proposal) x feature_dim x horizon
        h = []
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            x = resnet(x)
            x = resnet2(x)
            h.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x)

        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x)
            x = resnet2(x)
            x = upsample(x)
        # x: (batch, feature_dim) x horizon
        x = x.permute(0, 2, 1)
        # x: (batch, horizon x feature
        x = self.last_dropout(x)
        act = self.act_pred(x)
        score = self.score_pred(x.reshape(batch_size, -1))
        if self.num_proposals > 1:
            act = act.reshape(batch_size, self.pre_horizon, self.num_proposals, self.action_dim)
            act = act.permute(0, 2, 1, 3)
            score = score.reshape(batch_size, self.num_proposals)
        else:
            act = act.reshape(
                batch_size, self.pre_horizon, self.action_dim
            )
        return act, score


class CondHourglassDecoder(nn.Module):
    def __init__(self, in_channels, act_dim, pre_horizon, num_proposals, feature_dim=1024,
                 last_dropout=0.0, cond_dropout=0.0):
        super().__init__()
        kernel_size = 5
        n_groups = 8
        down_dims = [256, 512, 1024]
        all_dims = [act_dim] + list(down_dims)
        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        self.pre_horizon = pre_horizon
        self.action_dim = act_dim
        self.num_proposals = num_proposals
        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=in_channels, kernel_size=kernel_size, n_groups=n_groups,
            ),
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=in_channels, kernel_size=kernel_size, n_groups=n_groups,
            ),
        ])

        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(
                nn.ModuleList([
                    ConditionalResidualBlock1D(
                        dim_in, dim_out, cond_dim=in_channels, kernel_size=kernel_size, n_groups=n_groups,
                    ),
                    ConditionalResidualBlock1D(
                        dim_out, dim_out, cond_dim=in_channels, kernel_size=kernel_size, n_groups=n_groups,
                    ),
                    Downsample1d(dim_out) if not is_last else nn.Identity(),
                ])
            )

        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(
                nn.ModuleList([
                    ConditionalResidualBlock1D(
                        dim_out * 2, dim_in, cond_dim=in_channels, kernel_size=kernel_size, n_groups=n_groups,
                    ),
                    ConditionalResidualBlock1D(
                        dim_in, dim_in, cond_dim=in_channels, kernel_size=kernel_size, n_groups=n_groups,
                    ),
                    Upsample1d(dim_in) if not is_last else nn.Identity(),
                ])
            )
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.score_pred = nn.Linear(256 * self.pre_horizon, 1)
        self.act_pred = nn.Linear(256, act_dim)
        self.cls_token = torch.nn.Parameter(
            torch.randn(1, self.num_proposals, act_dim, self.pre_horizon)
        )  # "global information"
        torch.nn.init.normal_(self.cls_token, std=0.02)
        if last_dropout > 0:
            self.last_dropout = nn.Dropout(p=last_dropout, inplace=True)
        else:
            self.last_dropout = nn.Identity()
        if cond_dropout > 0:
            self.cond_dropout = nn.Dropout(p=cond_dropout, inplace=True)
        else:
            self.cond_dropout = nn.Identity()

    def forward(self, cond):
        h = []
        batch_size = cond.shape[0]
        cond = self.cond_dropout(cond)
        x = self.cls_token.repeat(cond.shape[0], 1, 1, 1)
        x = x.reshape(cond.shape[0] * self.num_proposals, self.action_dim, self.pre_horizon)
        cond = cond[:, None].repeat(1, self.num_proposals, 1).reshape(cond.shape[0] * self.num_proposals, -1)
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            x = resnet(x, cond)
            x = resnet2(x, cond)
            h.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, cond)

        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, cond)
            x = resnet2(x, cond)
            x = upsample(x)
        # x: (batch, feature_dim) x horizon
        x = x.permute(0, 2, 1)
        x = x.reshape(batch_size, self.num_proposals, self.pre_horizon, -1)
        x = self.last_dropout(x)
        # x: (batch, horizon x feature
        act = self.act_pred(x)
        score = self.score_pred(x.reshape(batch_size, self.num_proposals, -1))
        if self.num_proposals > 1:
            score = score.reshape(batch_size, self.num_proposals)
        return act, score

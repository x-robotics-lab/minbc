import os
import copy
import time
import torch
import numpy as np
import torch.distributed as dist

from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from dp.models.simple_act_pred import VanillaBC
from torch import nn
from torch.nn.functional import mse_loss
from torch.utils.tensorboard import SummaryWriter
from configs.base import MinBCConfig
from utils.obs import minmax_norm_data, minmax_unnorm_data
from utils.misc import tprint, pprint


class VanillaBCPolicy:
    def __init__(
        self,
        config: MinBCConfig,
        obs_horizon,
        obs_dim,
        pred_horizon,
        action_horizon,
        action_dim,
        weight_decay=1e-6,
        learning_rate=1e-4,
        binarize_touch=False,
        device='cpu',
    ):
        self.multi_gpu = config.multi_gpu
        self.config = config
        if self.multi_gpu:
            self.rank = int(os.getenv('LOCAL_RANK', '0'))
            self.rank_size = int(os.getenv('WORLD_SIZE', '1'))
            dist.init_process_group("nccl", rank=self.rank, world_size=self.rank_size)
            # self.device = 'cuda:' + str(self.rank)
            print(f'current rank: {self.rank} and use device {device}')
        else:
            self.rank = -1

        encoders = {}
        self.encoders = encoders
        self.obs_horizon = obs_horizon
        self.obs_dim = obs_dim
        self.pred_horizon = pred_horizon
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.data_stat = None
        self.writer = None
        self.binarize_touch = binarize_touch
        # create network object
        self.device = device
        self.num_proposals = config.dp.num_proposal
        self.clip_score_loss_max = config.dp.clip_score_loss_max
        self.clip_score_loss = config.dp.clip_score_loss
        self.model = VanillaBC(config, input_dim=action_dim, num_proposals=self.num_proposals, device=self.device)
        # Exponential Moving Average of the model weights
        self.ema = EMAModel(parameters=self.model.parameters(), power=0.75)
        self.ema_nets = copy.deepcopy(self.model)
        self.optimizer = torch.optim.AdamW(
            params=self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        if self.model.im_encoder == 'CLIP' or self.model.im_encoder == 'DINO' or self.model.im_encoder == 'DINOv3':
            if config.dp.encoder.im_encoder_frozen:
                for p in self.model.encoders.img_encoder.parameters():
                    p.requires_grad = False
                self.optimizer = torch.optim.AdamW(
                    params=[p for p in self.model.parameters() if p.requires_grad],
                    lr=learning_rate, weight_decay=weight_decay,
                )
            if config.dp.encoder.im_encoder_reduce_lr:
                img_ids = {id(p) for p in self.model.encoders.img_encoder.parameters()}

                # Split params; avoid equality comparison on tensors
                other_params = [p for p in self.model.parameters()
                                if p.requires_grad and id(p) not in img_ids]
                img_encoder_params = [p for p in self.model.encoders.img_encoder.parameters()
                                      if p.requires_grad]
                self.optimizer = torch.optim.AdamW(
                    [
                        {"params": other_params, "lr": learning_rate},
                        {"params": img_encoder_params, "lr": learning_rate / 10},
                    ],
                    weight_decay=weight_decay,
                )

    def set_lr_scheduler(self, num_warmup_steps, num_training_steps):
        # Cosine LR schedule with linear warmup
        self.lr_scheduler = get_scheduler(
            name="cosine",
            optimizer=self.optimizer,
            num_warmup_steps=500,
            num_training_steps=num_training_steps,
        )

    def to(self, device):
        self.device = device
        self.model.to(device)
        self.ema.to(device)
        self.ema_nets.to(device)

    def train(
        self,
        num_epochs,
        train_loader,
        test_loaders,
        save_path=None,
        save_freq=10,
        eval_freq=10,
        wandb_logger=None,
    ):
        if self.multi_gpu:
            # torch.cuda.set_device(self.rank)
            model_params = [self.model.state_dict()]
            dist.broadcast_object_list(model_params, 0)
            self.model.load_state_dict(model_params[0])

        best_mse = 1e10
        best_train_mse = 1e10
        global_training_step = 0
        self.model.train()
        if self.writer is None:
            self.writer = SummaryWriter(save_path)
        # epoch loop
        _init_t = time.time()
        for epoch_idx in range(num_epochs):
            epoch_loss = list()
            # batch loop
            _t = time.time()
            for data in train_loader:
                # data normalized in dataset
                gt_action = data["action"].to(self.device)
                # # visualize image
                # a = data['img'][0, 0, 0]  # batch, horizon, num_image
                # b = data['img'][0, 0, 1]
                # a = a * 255
                # b = b * 255
                # from matplotlib import pyplot as plt
                # a = a.permute(1, 2, 0).cpu().numpy().astype(np.int32)
                # b = b.permute(1, 2, 0).cpu().numpy().astype(np.int32)
                # a = np.concatenate([a, b], axis=1)
                # a = a[:, :, ::-1]
                # plt.imshow(a)
                # plt.show()

                ### IMPT: make sure input is always in this order
                # eef, hand_pos, img, pos, touch

                # predict the noise residual
                pred_action, pred_score = self.model(data)
                if self.num_proposals > 1:
                    _gt = gt_action[:, None].repeat(1, self.num_proposals, 1, 1)
                    loss = nn.functional.mse_loss(_gt, pred_action, reduction='none').mean(dim=(2, 3))
                    score_loss = ((pred_score - loss.detach()) ** 2)
                    if self.clip_score_loss:
                        score_loss = torch.clip(score_loss, max=self.clip_score_loss_max)
                    score_loss = score_loss.mean()
                    loss_mask = loss.argmin(dim=1)
                    loss = loss[torch.arange(loss.shape[0]), loss_mask]
                    loss = loss.mean() + score_loss
                else:
                    pred_action = pred_action.reshape(-1, self.pred_horizon, self.action_dim)
                    loss = nn.functional.mse_loss(gt_action, pred_action)

                # optimize
                loss.backward()

                if self.multi_gpu:
                    # batch all_reduce ops: see https://github.com/entity-neural-network/incubator/pull/220
                    all_grads_list = []
                    for param in self.model.parameters():
                        if param.grad is not None:
                            all_grads_list.append(param.grad.view(-1))
                    all_grads = torch.cat(all_grads_list)
                    dist.all_reduce(all_grads, op=dist.ReduceOp.SUM)
                    offset = 0
                    for param in self.model.parameters():
                        if param.grad is not None:
                            param.grad.data.copy_(
                                all_grads[offset: offset + param.numel()].view_as(param.grad.data) / self.rank_size
                            )
                            offset += param.numel()

                # per-step log
                self.writer.add_scalar(f'lr', self.optimizer.param_groups[0]['lr'], global_training_step)
                grad_norm = torch.norm(torch.stack(
                    [p.grad.detach().norm(2) for p in self.model.parameters() if p.grad is not None]
                ), 2)
                self.writer.add_scalar(f'grad_norm', grad_norm.item(), global_training_step)
                weight_norm = torch.norm(torch.stack(
                    [p.detach().norm(2) for p in self.model.parameters() if p is not None]
                ), 2)
                self.writer.add_scalar(f'weight_norm', weight_norm.item(), global_training_step)
                pred_action_max = torch.abs(pred_action).max()
                self.writer.add_scalar(f'pred_act_max', pred_action_max.item(), global_training_step)
                self.writer.add_scalar(f'loss', loss.item(), global_training_step)
                if self.num_proposals > 1:
                    self.writer.add_scalar(f'score_loss', score_loss.item(), global_training_step)
                global_training_step += 1

                self.optimizer.step()
                self.optimizer.zero_grad()
                # step lr scheduler every batch
                # this is different from standard pytorch behavior
                self.lr_scheduler.step()

                # update Exponential Moving Average of the model weights
                self.ema.step(self.model.parameters())
                epoch_loss.append(loss.item())

            if self.rank <= 0:
                eta_s = (time.time() - _init_t) / (epoch_idx + 1) * (num_epochs - epoch_idx - 1)
                eta_h = eta_s / 3600
                tprint(f"Epoch {epoch_idx} | Loss: {np.mean(epoch_loss):.4f} | "
                       f"Epoch Time: {time.time() - _t:.2f}s | ETA: {eta_h:.1f}h")
                self.writer.add_scalar("Epoch Loss", np.mean(epoch_loss), epoch_idx)

                if wandb_logger is not None:
                    wandb_logger.step()
                    wandb_logger.log({"Epoch Loss": np.mean(epoch_loss), "epoch": epoch_idx})
                if save_path is not None:
                    model_path = os.path.join(save_path, f"model_last.ckpt")
                    self.save(model_path)

            if epoch_idx % eval_freq == 0 or epoch_idx == num_epochs - 1:
                self.to_ema()
                self.ema_nets.eval()
                _eval_t = time.time()
                mses, normalized_mses = [], []
                for test_loader in test_loaders:
                    mse, normalized_mse = self.eval(test_loader)
                    mses.append(mse)
                    normalized_mses.append(normalized_mse)
                log_mse = mses[1:]
                mse = mses[0]
                normalized_mse = normalized_mses[0]

                if self.rank <= 0:
                    if mse < best_mse:
                        self.save(os.path.join(save_path, f"model_best.ckpt"))
                        best_mse = mse

                    for j, (_m, _n) in enumerate(zip(mses[1:], normalized_mses[1:])):
                        self.writer.add_scalar(f"Action_MSE_{j}", _m, epoch_idx)
                    self.writer.add_scalar("Action_MSE", mse, epoch_idx)
                    self.writer.add_scalar("Normalized_MSE", normalized_mse, epoch_idx)
                    pprint(f"{self.config.output_name} | Epoch {epoch_idx} | Test MSE: {mse:.4f} | Best Test MSE: {best_mse:.4f}")

                if (epoch_idx > 0 and epoch_idx % (eval_freq * 5) == 0
                        or epoch_idx == num_epochs - 1):
                    _eval_t = time.time()
                    _mse, _normalized_mse = self.eval(train_loader)
                    if self.rank <= 0:
                        if _mse < best_train_mse:
                            best_train_mse = _mse
                        self.writer.add_scalar("Train Action_MSE", _mse, epoch_idx)
                        self.writer.add_scalar("Train Normalized_MSE", _normalized_mse, epoch_idx)
                        pprint(f"{self.config.output_name} | {epoch_idx} | Train MSE: {_mse:.4f} | Best Train MSE: {best_train_mse:.4f}")

                if self.rank <= 0:
                    if wandb_logger is not None:
                        wandb_logger.log({"Action_MSE": mse})
                        wandb_logger.log({"Normalized_MSE": normalized_mse})
                self.ema_nets.train()

    @torch.no_grad()
    def eval(self, test_loader):
        mse = 0
        normalized_mse = 0
        cnt = 0
        for data in test_loader:
            gt_action = data["action"].to(self.device)

            pred_action, pred_score = self.ema_nets(data)
            if self.num_proposals > 1:
                score_mask = pred_score.argmin(dim=1)
                pred_action = pred_action[torch.arange(pred_action.shape[0]), score_mask]
            else:
                pred_action = pred_action.reshape(-1, self.pred_horizon, self.action_dim)

            # unnormalize action
            noisy_action = pred_action.detach().to("cpu").numpy()
            gt_action = gt_action.detach().to("cpu").numpy()
            action_pred = minmax_unnorm_data(noisy_action, dmin=self.data_stat["action"]["min"], dmax=self.data_stat["action"]["max"])
            gt_action = minmax_unnorm_data(gt_action, dmin=self.data_stat["action"]["min"], dmax=self.data_stat["action"]["max"])
            actions_pred = np.array(action_pred)
            action = np.array(gt_action)
            _mse = mse_loss(
                torch.tensor(actions_pred), torch.tensor(action), reduction='none'
            ).mean(-1).mean(-1).sum()
            normalized_action = minmax_norm_data(action, dmin=self.data_stat["action"]["min"], dmax=self.data_stat["action"]["max"])
            normalized_action_pred = minmax_norm_data(actions_pred, dmin=self.data_stat["action"]["min"], dmax=self.data_stat["action"]["max"])
            _normalized_mse = mse_loss(
                torch.tensor(normalized_action_pred),
                torch.tensor(normalized_action[:len(actions_pred)]),
                reduction='none',
            ).mean(-1).mean(-1).sum()
            mse += _mse.item()
            normalized_mse += _normalized_mse.item()
            cnt += gt_action.shape[0]

        mse /= cnt
        normalized_mse /= cnt
        return mse, normalized_mse

    def to_ema(self):
        # Weights of the EMA model
        # is used for inference
        self.ema.copy_to(self.ema_nets.parameters())

    def load(self, path):
        state_dict = torch.load(path, map_location="cuda", weights_only=True)
        self.ema_nets = self.model
        self.ema_nets.load_state_dict(state_dict)

    def save(self, path):
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        self.ema_nets = self.model
        torch.save(self.ema_nets.state_dict(), path)

    def _get_data_forward(self, stats, percentiles, obs_deque, data_key):
        sample = np.stack([x[data_key] for x in obs_deque])
        if data_key != "img":
            # image is already normalized
            p2 = percentiles[data_key]['lower']
            p98 = percentiles[data_key]['upper']
            mid = 0.5 * (p2 + p98)
            span = (p98 - p2)
            # Avoid divide-by-zero if a dim is (near) constant in the 2–98% range
            eps = 1e-12
            span_safe = np.where(span < eps, 1.0, span)
            y = 2.0 * (sample - mid[None, None]) / span_safe[None, None]  # 2–98% -> [-1, 1]
            y = np.clip(y, -1.5, 1.5)  # cap to [-1.5, 1.5]
            sample = y
        sample = (
            torch.from_numpy(sample).to(self.device, dtype=torch.float32).unsqueeze(0)
        )
        return sample

    def forward(self, stats, percentiles, obs_deque, num_diffusion_iters=None):
        self.ema_nets.eval()
        with torch.no_grad():
            data = {}
            # eef, hand_pos, img, pos, touch
            for data_key in self.config.data.data_key:
                sample = self._get_data_forward(stats, percentiles, obs_deque, data_key)
                data[data_key] = sample
            pred_action, pred_score = self.ema_nets(data)
            if self.num_proposals > 1:
                score_mask = pred_score.argmin(dim=1)
                pred_action = pred_action[torch.arange(pred_action.shape[0]), score_mask]
            else:
                pred_action = pred_action.reshape(-1, self.pred_horizon, self.action_dim)

        # unnormalize action
        action = pred_action.detach().to("cpu").numpy()
        # (B, pred_horizon, action_dim)
        action = action[0]
        action_pred = minmax_unnorm_data(action, dmin=stats["action"]["min"], dmax=stats["action"]["max"])

        # only take action_horizon number of actions
        start = self.obs_horizon - 1
        end = start + self.action_horizon
        action = action_pred[start:end, :]

        return action

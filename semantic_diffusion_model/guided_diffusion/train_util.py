# this script has been modified to incorporate w&b
# 07/11/2023

import copy
import functools
import os

import torch
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW

from .fp16_util import MixedPrecisionTrainer
from .resample import LossAwareSampler, UniformSampler

import matplotlib.pyplot as plt

# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0


def update_ema(target_params, source_params, rate=0.99):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.

    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)


class TrainLoop:
    def __init__(
        self,
        *,
        model,
        diffusion,
        data,
        num_classes,
        batch_size,
        lr,
        ema_rate,
        drop_rate,
        log_interval,
        save_interval,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=1e-3,
        lr_anneal_steps=0,
        output_dir,
        device="cuda",
    ):
        self.model = model
        self.diffusion = diffusion
        self.data = data
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.device = device

        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.drop_rate = drop_rate
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps
        self.output_dir = output_dir

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size * dist.get_world_size()

        self.sync_cuda = torch.cuda.is_available()

        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            fp16_scale_growth=self.fp16_scale_growth,
        )

        self.opt = AdamW(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        )
        self.ema_params = [
            copy.deepcopy(self.mp_trainer.master_params)
            for _ in range(len(self.ema_rate))
        ]

        # Clear CUDA cache before starting the training
        torch.cuda.empty_cache()

    def run_loop(self):
        while (
            not self.lr_anneal_steps
            or self.step + self.resume_step < self.lr_anneal_steps
        ):
            print(f"step: {self.step + self.resume_step}")
            batch, cond = next(self.data)

            cond = self.preprocess_input(cond)

            self.run_step(batch, cond)

            if self.step % self.save_interval == 0 and self.step > 0:
                print("Saving step")
                print("Saving step")
                self.save()
                self.sanity_test(batch=batch, device=self.device, cond=cond)
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return

            self.step += 1
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_step(self, batch, cond):
        self.forward_backward(batch, cond)

        took_step = self.mp_trainer.optimize(self.opt)
        if took_step:
            self._update_ema()
        self._anneal_lr()
        self.log_step()

        # Clear CUDA cache after each step
        torch.cuda.empty_cache()

    def forward_backward(self, batch, cond):
        self.mp_trainer.zero_grad()
        for i in range(0, batch.shape[0], self.batch_size):
            micro = batch[i : i + self.batch_size].to(self.device)
            micro_cond = {
                k: v[i : i + self.batch_size].to(self.device) for k, v in cond.items()
            }
            last_batch = (i + self.batch_size) >= batch.shape[0]

            t, weights = self.schedule_sampler.sample(micro.shape[0], self.device)

            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro,
                t,
                model_kwargs=micro_cond,
            )

            if last_batch or not self.use_ddp:
                losses = compute_losses()

            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            loss = (losses["loss"] * weights).mean()
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                loss = loss.clamp(-1e6, 1e6)

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean()
            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )
            self.mp_trainer.backward(loss)

    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.mp_trainer.master_params, rate=rate)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        # logger.logkv("step", self.step + self.resume_step)
        # logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)
        # logger.logkv("lr", self.opt.param_groups[0]["lr"])
        # logger.logkv("lr_anneal_steps", self.lr_anneal_steps)
        # logger.logkv("memory_usage", torch.cuda.memory_allocated())
        print("Please write logging code HERE!")
        pass

    def save(self):
        def save_checkpoint(rate, params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            if dist.get_rank() == 0:
                print(f"saving model {rate}...")
                if not rate:
                    filename = f"model{(self.step + self.resume_step):06d}.pt"
                else:
                    filename = f"ema_{rate}_{(self.step + self.resume_step):06d}.pt"
                save_path = os.path.join(self.output_dir, filename)
                torch.save(state_dict, save_path)
                print(f"saved model {rate} to {save_path}")

        save_checkpoint(0, self.mp_trainer.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        if dist.get_rank() == 0:
            optimizer_filename = f"opt{(self.step + self.resume_step):06d}.pt"
            optimizer_path = os.path.join(self.output_dir, optimizer_filename)
            torch.save(self.opt.state_dict(), optimizer_path)
        dist.barrier()

    def sanity_test(self, batch, device, cond):
        src_img = ((batch + 1.0) / 2.0).to(device)
        model_kwargs = cond

        with torch.no_grad():
            self.model.eval()
            inference_img, snapshots = self.diffusion.p_sample_loop_with_snapshot(
                self.model,
                (batch.shape[0], 3, batch.shape[2], batch.shape[3]),
                model_kwargs=model_kwargs,
                progress=True,
            )
            self.model.train()

        inference_img = (inference_img + 1) / 2.0
        log_images(
            inference_img=inference_img,
            src_img=src_img,
            snapshots=snapshots,
            output_dir=self.output_dir,
            self=self,
        )

    def preprocess_input(self, data):
        data["label"] = data["label"].long()

        label_map = data["label"]
        bs, _, h, w = label_map.size()
        nc = self.num_classes
        input_label = torch.FloatTensor(bs, nc, h, w).zero_()
        input_semantics = input_label.scatter_(1, label_map, 1.0)

        if "instance" in data:
            inst_map = data["instance"]
            instance_edge_map = self.get_edges(inst_map)
            input_semantics = torch.cat((input_semantics, instance_edge_map), dim=1)

        if self.drop_rate > 0.0:
            mask = (
                torch.rand([input_semantics.shape[0], 1, 1, 1]) > self.drop_rate
            ).float()
            input_semantics = input_semantics * mask

        cond = {
            key: value
            for key, value in data.items()
            if key not in ["label", "instance", "path", "label_ori"]
        }
        cond["y"] = input_semantics

        return cond

    def get_edges(self, t):
        edge = torch.ByteTensor(t.size()).zero_()
        edge[:, :, :, 1:] = edge[:, :, :, 1:] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, :, :-1] = edge[:, :, :, :-1] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, 1:, :] = edge[:, :, 1:, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        edge[:, :, :-1, :] = edge[:, :, :-1, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        return edge.float()


def log_loss_dict(diffusion, ts, losses):
    print("Please write logging code HERE!")
    # for key, values in losses.items():
    #     logger.logkv_mean(key, values.mean().item())
    #     for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
    #         quartile = int(4 * sub_t / diffusion.num_timesteps)
    #         logger.logkv_mean(f"{key}_q{quartile}", sub_loss)


def log_images(inference_img, src_img, snapshots, output_dir, self):
    num_rows = 2 + len(snapshots)
    num_cols = inference_img.shape[0]
    base_width = 4
    base_height = 4
    fig_width = num_cols * base_width + 2
    fig_height = num_rows * base_height

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(fig_width, fig_height))
    fig.suptitle("Diffusion Model Results", fontsize=16)

    for k in range(num_cols):
        axs[0, k].imshow(src_img[k, 0, ...].cpu().detach().numpy(), cmap="gray")
        axs[0, k].axis("off")

        axs[1, k].imshow(inference_img[k, 0, ...].cpu().detach().numpy(), cmap="gray")
        axs[1, k].axis("off")

        for i, snap in enumerate(snapshots):
            axs[i + 2, k].imshow(
                snapshots[snap][k, 0, ...].cpu().detach().numpy(), cmap="gray"
            )
            axs[i + 2, k].axis("off")

    axs[0, 0].set_title("Source Image")
    axs[1, 0].set_title("Inference Image")
    for i, snap in enumerate(snapshots):
        axs[i + 2, 0].set_title(f"Snapshot {snap}")

    plt.tight_layout(rect=[0, 0.0, 1, 0.95])
    plt.savefig(f"{output_dir}/diffusion_results_{str(self.step).zfill(6)}.png")
    plt.close()

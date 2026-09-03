"""Training loop for EchoDiffusion.

Standard AdamW + cosine schedule + EMA + AMP, with validation that reports
*trajectory* error in metres rather than only the denoising loss.  That
distinction matters: the diffusion MSE is measured on noise at a random
timestep and barely moves once training is underway, so it is nearly useless
for spotting a policy that has learned to drive straight ahead regardless of
where the sound is.  The metrics that catch that are ADE/FDE and the
final-heading error against the true source bearing.
"""

from __future__ import annotations

import copy
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..audio.odas import dominant_bearing_from_tokens

#: Minimum predicted endpoint displacement (metres) for the bearing metric to
#: mean anything -- below this the endpoint is effectively at the origin.
MIN_ENDPOINT_M = 0.05


class EMA:
    """Exponential moving average of model weights, used for evaluation.

    The decay is warmed up as ``min(decay, (1 + step) / (10 + step))``.  Without
    it, a short run validates a shadow that is still mostly random
    initialisation -- at decay 0.999 the init still carries ~50% weight after
    700 steps -- which makes validation metrics look catastrophic and
    completely disconnected from the training loss.
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.999,
                 warmup: bool = True):
        self.decay = decay
        self.warmup = warmup
        self.step = 0
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    def current_decay(self) -> float:
        if not self.warmup:
            return self.decay
        return min(self.decay, (1.0 + self.step) / (10.0 + self.step))

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        d = self.current_decay()
        for s, p in zip(self.shadow.parameters(), model.parameters()):
            s.mul_(d).add_(p.detach(), alpha=1.0 - d)
        for s, b in zip(self.shadow.buffers(), model.buffers()):
            s.copy_(b)
        self.step += 1


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict,
        logger=None,
        device: torch.device | None = None,
        resume: str | None = None,
    ):
        self.config = config
        tc = config.get("training", {})
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.logger = logger

        self.num_epochs = int(tc.get("num_epochs", 100))
        self.grad_clip = float(tc.get("grad_clip", 1.0))
        self.log_every = int(tc.get("log_every", 20))
        self.val_every = int(tc.get("val_every", 1))
        self.viz_every = int(tc.get("viz_every", 5))
        self.save_every = int(tc.get("save_every", 10))

        self.output_dir = Path(config.get("paths", {}).get(
            "output_dir", "checkpoints/run"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=float(tc.get("learning_rate", 1e-4)),
            weight_decay=float(tc.get("weight_decay", 1e-6)),
            betas=tuple(tc.get("betas", (0.95, 0.999))),
        )

        self.warmup_steps = int(tc.get("warmup_steps", 500))
        self.total_steps = max(self.num_epochs * max(len(train_loader), 1), 1)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, self._lr_lambda)

        self.use_amp = bool(tc.get("amp", True)) and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        self.ema = EMA(self.model, float(tc.get("ema_decay", 0.999)),
                       warmup=bool(tc.get("ema_warmup", True))) \
            if bool(tc.get("use_ema", True)) else None

        self.epoch = 0
        self.global_step = 0
        self.best_ade = math.inf

        # Metres per normalised unit -- needed to report errors in metres.
        self.traj_scale = float(getattr(train_loader.dataset, "traj_scale", 1.0))
        self.goal_scale = float(
            getattr(train_loader.dataset.cfg, "goal_scale", 8.0))
        self.field_range = float(train_loader.dataset.bev_cfg.range_m)

        if resume:
            self.load_checkpoint(resume)

    def _lr_lambda(self, step: int) -> float:
        if step < self.warmup_steps:
            return step / max(self.warmup_steps, 1)
        progress = (step - self.warmup_steps) / max(
            self.total_steps - self.warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    # ── loops ─────────────────────────────────────────────────────────────

    def _to_device(self, batch: dict) -> dict:
        return {k: v.to(self.device, non_blocking=True) if torch.is_tensor(v) else v
                for k, v in batch.items()}

    def train_epoch(self) -> dict:
        self.model.train()
        totals: dict[str, float] = {}
        n = 0

        pbar = tqdm(self.train_loader, desc=f"epoch {self.epoch}", leave=False)
        for batch in pbar:
            batch = self._to_device(batch)

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                out = self.model.compute_loss(batch)
                loss = out["loss"]

            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            if self.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                               self.grad_clip)
            # AMP may skip the optimizer step when it detects inf/nan and
            # lowers the loss scale.  Stepping the LR schedule anyway both
            # desynchronises it and triggers a torch warning, so gate on
            # whether the scale actually held.
            scale_before = self.scaler.get_scale()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            if self.scaler.get_scale() >= scale_before:
                self.scheduler.step()

            if self.ema is not None:
                self.ema.update(self.model)

            for k, v in out.items():
                totals[k] = totals.get(k, 0.0) + float(v.detach())
            n += 1
            self.global_step += 1

            if self.global_step % self.log_every == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                pbar.set_postfix(loss=f"{float(loss.detach()):.4f}", lr=f"{lr:.2e}")
                if self.logger:
                    self.logger.log_metrics(
                        {**{k: float(v.detach()) for k, v in out.items()},
                         "lr": lr},
                        step=self.global_step, epoch=self.epoch, prefix="train/")

        return {k: v / max(n, 1) for k, v in totals.items()}

    @torch.no_grad()
    def validate(self, visualize: bool = False) -> dict:
        """Denoising loss plus sampled-trajectory metrics in metres."""
        model = self.ema.shadow if self.ema is not None else self.model
        model.eval()

        totals: dict[str, float] = {}
        n_batches = 0
        ade_sum = fde_sum = bearing_sum = 0.0
        n_traj = 0
        n_bearing = 0
        first: dict | None = None

        for batch in tqdm(self.val_loader, desc="val", leave=False):
            batch = self._to_device(batch)

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                out = model.compute_loss(batch)
            for k, v in out.items():
                totals[k] = totals.get(k, 0.0) + float(v.detach())
            n_batches += 1

            pred = model.predict(batch).float()
            target = batch["traj"].float()
            valid = batch["traj_valid"].bool()
            if valid.any():
                err = (pred - target)[valid] * self.traj_scale  # -> metres
                dist = err.norm(dim=-1)                          # (Nv, horizon)
                ade_sum += float(dist.mean(dim=-1).sum())
                fde_sum += float(dist[:, -1].sum())
                n_traj += int(valid.sum())

                # Does the endpoint actually point at the source?  This is the
                # metric that separates "learned the audio" from "learned the
                # average training path".
                sv = batch["source_valid"].bool() & valid
                if sv.any():
                    endpoint = pred[sv][:, -1] * self.traj_scale
                    src = batch["source_xy"][sv].float() * self.goal_scale
                    # Skip near-stationary predictions: their endpoint sits at
                    # the origin, so the bearing is noise and would swamp the
                    # mean without saying anything about the audio.
                    moving = endpoint.norm(dim=-1) > MIN_ENDPOINT_M
                    if moving.any():
                        a = torch.atan2(endpoint[moving, 1], endpoint[moving, 0])
                        b = torch.atan2(src[moving, 1], src[moving, 0])
                        diff = torch.atan2(torch.sin(a - b), torch.cos(a - b)).abs()
                        bearing_sum += float(diff.sum())
                        n_bearing += int(moving.sum())

            if first is None:
                k = min(4, batch["bev"].shape[0])
                # Frame 0 of the DoA stack is the anchor step: what the policy
                # heard at the moment it produced this trajectory.
                doa_theta, doa_r = dominant_bearing_from_tokens(
                    batch["doa"][:k, 0].float().cpu().numpy(),
                    batch["doa_mask"][:k, 0].float().cpu().numpy())
                first = {
                    "bev": batch["bev"][:k].float().cpu().numpy(),
                    "pred": pred[:k].cpu().numpy(),
                    "target": target[:k].cpu().numpy(),
                    "source_xy": batch["source_xy"][:k].float().cpu().numpy()
                    if batch["source_valid"][:k].any() else None,
                    "doa_bearing": doa_theta,
                    "doa_strength": doa_r,
                }

        metrics = {k: v / max(n_batches, 1) for k, v in totals.items()}
        metrics["ade_m"] = ade_sum / max(n_traj, 1)
        metrics["fde_m"] = fde_sum / max(n_traj, 1)
        if n_bearing:
            metrics["endpoint_bearing_err_deg"] = math.degrees(
                bearing_sum / n_bearing)

        if visualize and self.logger and first is not None:
            self.logger.log_trajectory_samples(
                first["bev"], first["pred"], first["target"],
                traj_scale=self.traj_scale, field_range=self.field_range,
                step=self.global_step, source_xy=first["source_xy"],
                goal_scale=self.goal_scale,
                doa_bearing=first["doa_bearing"],
                doa_strength=first["doa_strength"],
                name=f"val_traj_vs_doa_epoch{self.epoch:03d}")

        return metrics

    def train(self) -> None:
        print(f"Training on {self.device} for {self.num_epochs} epochs "
              f"({len(self.train_loader)} batches/epoch)")

        for epoch in range(self.epoch, self.num_epochs):
            self.epoch = epoch
            t0 = time.time()
            train_metrics = self.train_epoch()

            line = (f"epoch {epoch:3d}  "
                    + "  ".join(f"{k}={v:.4f}" for k, v in train_metrics.items())
                    + f"  ({time.time() - t0:.1f}s)")

            if (epoch + 1) % self.val_every == 0:
                val_metrics = self.validate(
                    visualize=(epoch + 1) % self.viz_every == 0)
                line += "  |  " + "  ".join(
                    f"val_{k}={v:.4f}" for k, v in val_metrics.items())
                if self.logger:
                    self.logger.log_metrics(val_metrics, step=self.global_step,
                                            epoch=epoch, prefix="val/")
                if val_metrics["ade_m"] < self.best_ade:
                    self.best_ade = val_metrics["ade_m"]
                    self.save_checkpoint("best.pt")
                    line += "  *best*"

            print(line)
            if self.logger:
                self.logger.log_metrics(train_metrics, step=self.global_step,
                                        epoch=epoch, prefix="train_epoch/")

            if (epoch + 1) % self.save_every == 0:
                self.save_checkpoint("last.pt")

        self.save_checkpoint("last.pt")
        print(f"Done. Best val ADE: {self.best_ade:.4f} m")

    # ── checkpoints ───────────────────────────────────────────────────────

    def save_checkpoint(self, filename: str) -> Path:
        path = self.output_dir / filename
        torch.save({
            "epoch": self.epoch,
            "global_step": self.global_step,
            "best_ade": self.best_ade,
            "model": self.model.state_dict(),
            "ema": self.ema.shadow.state_dict() if self.ema else None,
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "config": self.config,
            # Predictions are normalised; without this a checkpoint cannot be
            # turned back into metres.
            "traj_scale": self.traj_scale,
        }, path)
        return path

    def load_checkpoint(self, path: str | Path) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model"])
        if self.ema is not None and ckpt.get("ema"):
            self.ema.shadow.load_state_dict(ckpt["ema"])
        if "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            self.scheduler.load_state_dict(ckpt["scheduler"])
        self.epoch = int(ckpt.get("epoch", 0)) + 1
        self.global_step = int(ckpt.get("global_step", 0))
        self.best_ade = float(ckpt.get("best_ade", math.inf))
        print(f"Resumed from {path} at epoch {self.epoch}")

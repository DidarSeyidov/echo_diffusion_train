"""DDPM training objective and DDIM sampler.

Implemented directly rather than via ``diffusers`` so the noise schedule and
prediction target stay visible and hackable, and so the repo does not pin a
fast-moving external API.  Trajectory diffusion is small enough that there is
nothing to gain from a heavier library.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def linear_beta_schedule(timesteps: int, beta_start: float = 1e-4,
                         beta_end: float = 0.02) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """Nichol & Dhariwal cosine schedule.

    Preferred for short sequences: the linear schedule destroys the signal too
    early for a 20-step trajectory, leaving most timesteps uninformative.
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class GaussianDiffusion(nn.Module):
    """Buffers and math for DDPM training / DDIM sampling.

    Args:
        num_timesteps: training diffusion steps.
        beta_schedule: ``"cosine"`` or ``"linear"``.
        prediction_type: ``"epsilon"`` (predict the noise) or ``"sample"``
            (predict the clean trajectory).  ``"sample"`` is often steadier for
            low-dimensional trajectory data, where the signal-to-noise ratio at
            high t makes epsilon-prediction targets nearly arbitrary.
        clip_sample: clamp predicted trajectories to ``[-clip_range, clip_range]``
            during sampling.  Targets are normalised, so this is a meaningful
            bound rather than an arbitrary one.
    """

    def __init__(
        self,
        num_timesteps: int = 100,
        beta_schedule: str = "cosine",
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        prediction_type: str = "epsilon",
        clip_sample: bool = True,
        clip_range: float = 1.5,
    ):
        super().__init__()
        if prediction_type not in ("epsilon", "sample"):
            raise ValueError(f"unknown prediction_type {prediction_type!r}")
        self.num_timesteps = int(num_timesteps)
        self.prediction_type = prediction_type
        self.clip_sample = clip_sample
        self.clip_range = clip_range

        if beta_schedule == "cosine":
            betas = cosine_beta_schedule(self.num_timesteps)
        elif beta_schedule == "linear":
            betas = linear_beta_schedule(self.num_timesteps, beta_start, beta_end)
        else:
            raise ValueError(f"unknown beta_schedule {beta_schedule!r}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas.float())
        self.register_buffer("alphas_cumprod", alphas_cumprod.float())
        self.register_buffer("sqrt_alphas_cumprod", alphas_cumprod.sqrt().float())
        self.register_buffer("sqrt_one_minus_alphas_cumprod",
                             (1.0 - alphas_cumprod).sqrt().float())

    # ── forward process ───────────────────────────────────────────────────

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor,
                 noise: torch.Tensor) -> torch.Tensor:
        """Add ``t`` steps of noise to a clean trajectory."""
        a = self.sqrt_alphas_cumprod[t].view(-1, *([1] * (x_start.ndim - 1)))
        b = self.sqrt_one_minus_alphas_cumprod[t].view(-1, *([1] * (x_start.ndim - 1)))
        return a * x_start + b * noise

    def target_for(self, x_start: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """What the network should regress, given the prediction type."""
        return noise if self.prediction_type == "epsilon" else x_start

    def sample_timesteps(self, batch: int, device: torch.device) -> torch.Tensor:
        return torch.randint(0, self.num_timesteps, (batch,), device=device)

    # ── reverse process ───────────────────────────────────────────────────

    def _to_x0(self, model_out: torch.Tensor, x_t: torch.Tensor,
               t: torch.Tensor) -> torch.Tensor:
        if self.prediction_type == "sample":
            x0 = model_out
        else:
            a = self.sqrt_alphas_cumprod[t].view(-1, *([1] * (x_t.ndim - 1)))
            b = self.sqrt_one_minus_alphas_cumprod[t].view(
                -1, *([1] * (x_t.ndim - 1)))
            x0 = (x_t - b * model_out) / a.clamp(min=1e-8)
        if self.clip_sample:
            x0 = x0.clamp(-self.clip_range, self.clip_range)
        return x0

    @torch.no_grad()
    def ddim_sample(
        self,
        model,
        shape: tuple[int, ...],
        global_cond: torch.Tensor | None = None,
        num_steps: int = 16,
        eta: float = 0.0,
        device: torch.device | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Sample trajectories with DDIM.

        ``eta = 0`` is the deterministic sampler: identical conditioning gives
        identical output, which is what you want for a controller.  Raise it
        to draw a spread of plausible paths (useful for measuring the policy's
        multimodality when the bearing is ambiguous).
        """
        device = device or next(model.parameters()).device
        x = torch.randn(shape, device=device, generator=generator)

        # Uniformly spaced subsequence, ending at t = 0.
        step_ratio = max(self.num_timesteps // num_steps, 1)
        timesteps = list(range(0, self.num_timesteps, step_ratio))[::-1]

        for i, t in enumerate(timesteps):
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
            model_out = model(x, t_batch, global_cond)
            x0 = self._to_x0(model_out, x, t_batch)

            alpha_t = self.alphas_cumprod[t]
            t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else -1
            alpha_prev = self.alphas_cumprod[t_prev] if t_prev >= 0 \
                else torch.tensor(1.0, device=device)

            eps = (x - alpha_t.sqrt() * x0) / (1 - alpha_t).sqrt().clamp(min=1e-8)
            sigma = eta * ((1 - alpha_prev) / (1 - alpha_t)).sqrt() \
                * (1 - alpha_t / alpha_prev).sqrt()
            dir_xt = (1 - alpha_prev - sigma ** 2).clamp(min=0).sqrt() * eps

            x = alpha_prev.sqrt() * x0 + dir_xt
            if eta > 0 and t_prev >= 0:
                x = x + sigma * torch.randn(shape, device=device, generator=generator)

        return x

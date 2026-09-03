"""Conditional 1-D U-Net over the trajectory sequence.

This is the denoising backbone of the diffusion policy: it maps a noised
trajectory ``(B, horizon, 2)`` plus a conditioning vector to the noise (or
clean trajectory) prediction.  Conditioning enters through FiLM -- every
residual block is scaled and shifted by a projection of
``[timestep_embedding, global_cond]`` -- which keeps the temporal convolutions
free to model waypoint-to-waypoint smoothness while the audio evidence sets
*where* the path goes.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class SinusoidalPosEmb(nn.Module):
    """Transformer-style sinusoidal embedding of the diffusion timestep."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=x.device) / (half - 1))
        args = x[:, None].float() * freqs[None]
        return torch.cat([args.sin(), args.cos()], dim=-1)


class Conv1dBlock(nn.Module):
    """Conv1d -> GroupNorm -> Mish."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, n_groups: int = 8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(min(n_groups, out_ch), out_ch),
            nn.Mish(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ConditionalResidualBlock1D(nn.Module):
    """Two conv blocks with FiLM conditioning applied between them."""

    def __init__(self, in_ch: int, out_ch: int, cond_dim: int,
                 kernel_size: int = 5, n_groups: int = 8):
        super().__init__()
        self.blocks = nn.ModuleList([
            Conv1dBlock(in_ch, out_ch, kernel_size, n_groups),
            Conv1dBlock(out_ch, out_ch, kernel_size, n_groups),
        ])
        # Predicts per-channel (scale, bias) -- FiLM.
        self.cond_encoder = nn.Sequential(
            nn.Mish(), nn.Linear(cond_dim, out_ch * 2))
        self.out_ch = out_ch
        self.residual_conv = (nn.Conv1d(in_ch, out_ch, 1)
                              if in_ch != out_ch else nn.Identity())

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond).reshape(-1, 2, self.out_ch, 1)
        out = embed[:, 0] * out + embed[:, 1]
        out = self.blocks[1](out)
        return out + self.residual_conv(x)


class ConditionalUnet1D(nn.Module):
    """U-Net over the waypoint axis with global FiLM conditioning.

    Args:
        input_dim: channels per waypoint (2 for x/y, 3 with heading).
        global_cond_dim: width of the fused conditioning vector.
        diffusion_step_embed_dim: timestep embedding width.
        down_dims: channel widths per resolution level.
        kernel_size: temporal conv kernel.
    """

    def __init__(
        self,
        input_dim: int = 2,
        global_cond_dim: int = 256,
        diffusion_step_embed_dim: int = 128,
        down_dims: tuple[int, ...] = (128, 256, 512),
        kernel_size: int = 5,
        n_groups: int = 8,
    ):
        super().__init__()
        dsed = diffusion_step_embed_dim
        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4), nn.Mish(), nn.Linear(dsed * 4, dsed),
        )
        cond_dim = dsed + global_cond_dim

        dims = [input_dim] + list(down_dims)
        in_out = list(zip(dims[:-1], dims[1:]))

        self.down_modules = nn.ModuleList()
        for i, (din, dout) in enumerate(in_out):
            is_last = i >= len(in_out) - 1
            self.down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(din, dout, cond_dim, kernel_size, n_groups),
                ConditionalResidualBlock1D(dout, dout, cond_dim, kernel_size, n_groups),
                nn.Conv1d(dout, dout, 3, 2, 1) if not is_last else nn.Identity(),
            ]))

        mid = down_dims[-1]
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(mid, mid, cond_dim, kernel_size, n_groups),
            ConditionalResidualBlock1D(mid, mid, cond_dim, kernel_size, n_groups),
        ])

        # One up module per downsample, and every one of them upsamples -- the
        # down path applies len(in_out) - 1 stride-2 convs (its last level is
        # Identity), so the counts balance and the output length matches the
        # input horizon exactly.
        self.up_modules = nn.ModuleList()
        for din, dout in reversed(in_out[1:]):
            self.up_modules.append(nn.ModuleList([
                # ``dout * 2`` because the skip connection is concatenated.
                ConditionalResidualBlock1D(dout * 2, din, cond_dim, kernel_size, n_groups),
                ConditionalResidualBlock1D(din, din, cond_dim, kernel_size, n_groups),
                nn.ConvTranspose1d(din, din, 4, 2, 1),
            ]))

        self.n_downsamples = len(in_out) - 1

        self.final_conv = nn.Sequential(
            Conv1dBlock(down_dims[0], down_dims[0], kernel_size, n_groups),
            nn.Conv1d(down_dims[0], input_dim, 1),
        )

    def forward(self, sample: torch.Tensor, timestep: torch.Tensor,
                global_cond: torch.Tensor | None = None) -> torch.Tensor:
        """``sample``: (B, horizon, input_dim) -> same shape.

        ``horizon`` must be divisible by ``2 ** n_downsamples`` (8 with the
        default three levels), otherwise the up path cannot restore the
        original length.
        """
        horizon = sample.shape[1]
        stride = 2 ** self.n_downsamples
        if horizon % stride:
            raise ValueError(
                f"horizon={horizon} must be divisible by {stride} for "
                f"down_dims with {self.n_downsamples} downsamples; pick a "
                f"different data.horizon or shorten model.unet.down_dims")

        x = sample.transpose(1, 2)                       # (B, C, T)

        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], device=sample.device)
        timestep = timestep.expand(sample.shape[0]).to(sample.device)

        cond = self.diffusion_step_encoder(timestep)
        if global_cond is not None:
            cond = torch.cat([cond, global_cond], dim=-1)

        skips = []
        for res1, res2, downsample in self.down_modules:
            x = res1(x, cond)
            x = res2(x, cond)
            skips.append(x)
            x = downsample(x)

        for module in self.mid_modules:
            x = module(x, cond)

        for res1, res2, upsample in self.up_modules:
            x = torch.cat([x, skips.pop()], dim=1)
            x = res1(x, cond)
            x = res2(x, cond)
            x = upsample(x)

        return self.final_conv(x).transpose(1, 2)        # (B, T, C)

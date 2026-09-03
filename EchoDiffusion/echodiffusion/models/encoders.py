"""Conditioning encoders: BEV field, raw DoA tokens, and ego state.

Three complementary views of the same audio evidence feed the policy:

* :class:`BEVFieldEncoder` reads the *fused* spatial posterior -- where the
  filter currently believes the source is, including range once motion has
  triangulated it.
* :class:`DoATokenEncoder` reads the *raw recent bearings*.  The field is a
  lossy summary with a multi-second half-life, so the tokens are what let the
  policy react to a new detection immediately.
* :class:`EgoEncoder` reads recent self-motion plus the filter's scalar
  readout, which tells the policy how much its own movement has been buying in
  localisation certainty.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class BEVFieldEncoder(nn.Module):
    """Small CNN over the ``(1 + history_len, H, W)`` sound-probability stack.

    Deliberately shallow: the input is already a probability map in metric
    space, not raw pixels, so there is little hierarchy to discover and a
    heavy backbone would just overfit.  Coordinates are preserved by ending in
    an adaptive pool to a small grid rather than a global pool -- collapsing to
    a single vector would throw away *where* the mass sits, which is the only
    thing the map is carrying.
    """

    def __init__(self, in_channels: int = 5, out_dim: int = 256,
                 widths: tuple[int, ...] = (32, 64, 128), pool: int = 4,
                 dropout: float = 0.0):
        super().__init__()
        layers: list[nn.Module] = []
        c_in = in_channels
        for c_out in widths:
            layers += [
                nn.Conv2d(c_in, c_out, 3, stride=2, padding=1),
                nn.GroupNorm(min(8, c_out), c_out),
                nn.Mish(),
            ]
            c_in = c_out
        self.conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(pool)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c_in * pool * pool, out_dim),
            nn.Mish(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(out_dim, out_dim),
        )
        self.out_dim = out_dim

    def forward(self, bev: torch.Tensor) -> torch.Tensor:
        """``bev``: (B, C, H, W) -> (B, out_dim)."""
        return self.head(self.pool(self.conv(bev)))


class DoATokenEncoder(nn.Module):
    """Encoder over ``(B, T, K, F)`` detection tokens with a validity mask.

    Detections within a frame are an unordered set of varying size, so they are
    pooled permutation-invariantly (masked mean + masked max) rather than
    flattened.  Max-pooling matters here: mean alone washes out a single strong
    detection sitting among several weak clutter peaks, which is precisely the
    situation in the reference recordings.
    """

    def __init__(self, feature_dim: int = 6, token_dim: int = 128,
                 out_dim: int = 128, n_frames: int = 4):
        super().__init__()
        self.token_mlp = nn.Sequential(
            nn.Linear(feature_dim, token_dim), nn.Mish(),
            nn.Linear(token_dim, token_dim), nn.Mish(),
        )
        self.frame_mlp = nn.Sequential(
            nn.Linear(token_dim * 2, token_dim), nn.Mish())
        self.temporal = nn.Sequential(
            nn.Flatten(),
            nn.Linear(token_dim * n_frames, out_dim), nn.Mish(),
            nn.Linear(out_dim, out_dim),
        )
        self.out_dim = out_dim

    def forward(self, doa: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """``doa``: (B, T, K, F), ``mask``: (B, T, K) -> (B, out_dim)."""
        h = self.token_mlp(doa)                          # (B, T, K, D)
        m = mask.unsqueeze(-1)                           # (B, T, K, 1)

        denom = m.sum(dim=2).clamp(min=1.0)
        mean = (h * m).sum(dim=2) / denom
        # Masked-out slots must not win the max; -inf would produce NaNs for an
        # all-empty frame, so use a large finite sentinel and zero it after.
        masked = h.masked_fill(m == 0, -1e4)
        peak = masked.max(dim=2).values
        peak = peak * (m.sum(dim=2) > 0).float()

        frame = self.frame_mlp(torch.cat([mean, peak], dim=-1))   # (B, T, D)
        return self.temporal(frame)


class EgoEncoder(nn.Module):
    """MLP over flattened past ego poses plus the filter's scalar readout."""

    def __init__(self, past_len: int = 10, past_dim: int = 3,
                 extra_dim: int = 4, out_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(past_len * past_dim + extra_dim, out_dim), nn.Mish(),
            nn.Linear(out_dim, out_dim),
        )
        self.out_dim = out_dim

    def forward(self, past: torch.Tensor, extra: torch.Tensor) -> torch.Tensor:
        """``past``: (B, P, 3), ``extra``: (B, E) -> (B, out_dim)."""
        return self.net(torch.cat([past.flatten(1), extra], dim=-1))


class ImagePool(nn.Module):
    """Attention-pool ViT patch tokens into one conditioning vector.

    A learned query attends over the patches, so the model can select the
    patches that matter (e.g. the region the audio points at) instead of
    averaging the whole frame.
    """

    def __init__(self, embed_dim: int, out_dim: int = 256, num_heads: int = 8):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, out_dim), nn.Mish(), nn.Linear(out_dim, out_dim))
        self.out_dim = out_dim

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """``tokens``: (B, N, D) -> (B, out_dim)."""
        q = self.query.expand(tokens.shape[0], -1, -1)
        pooled, _ = self.attn(q, tokens, tokens, need_weights=False)
        return self.proj(self.norm(pooled.squeeze(1)))

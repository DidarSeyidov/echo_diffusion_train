"""DINOv2 backbone with lightweight adapters and temporal fusion.

Ported from HorizonDiffusion, trimmed to what EchoDiffusion needs.  The design
is unchanged and deliberately so: a frozen self-supervised DINOv2 trunk keeps
the visual features general (important when the image branch will only ever see
a few hours of one robot's footage), while zero-initialised bottleneck adapters
in the last transformer blocks give just enough capacity to specialise.  Only
the adapters and the temporal fusion train.

This module is imported **only** when ``data.use_image`` is true, so the
audio-only path never needs ``timm``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange, repeat
from torch.utils.checkpoint import checkpoint as grad_checkpoint


class AdapterLayer(nn.Module):
    """Bottleneck adapter: down -> GELU -> up, with a residual.

    The up-projection is zero-initialised so the adapter starts as an exact
    identity -- training then begins from the pretrained DINOv2 behaviour
    rather than from a perturbed version of it.
    """

    def __init__(self, dim: int, bottleneck_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, dim),
        )
        nn.init.zeros_(self.net[3].weight)
        nn.init.zeros_(self.net[3].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class TemporalAttention(nn.Module):
    """Each patch position attends across the T frames, then averages.

    Input ``(B, T, N, D)`` -> output ``(B, N, D)``.
    """

    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.num_heads = num_heads

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, N, D = x.shape
        x_flat = rearrange(self.norm(x), "b t n d -> (b n) t d")
        qkv = self.qkv(x_flat).chunk(3, dim=-1)
        q, k, v = (rearrange(t, "bn t (h d) -> bn h t d", h=self.num_heads)
                   for t in qkv)
        scale = (D // self.num_heads) ** -0.5
        attn = (q @ k.transpose(-2, -1) * scale).softmax(dim=-1)
        out = rearrange(attn @ v, "bn h t d -> bn t (h d)")
        out = self.proj(out).mean(dim=1)
        return rearrange(out, "(b n) d -> b n d", b=B)


class ViTAdapter(nn.Module):
    """Frozen DINOv2 ViT + trainable adapters + temporal fusion."""

    def __init__(
        self,
        vit_model: str = "vit_base_patch14_dinov2.lvd142m",
        num_frames: int = 1,
        use_adapter: bool = True,
        adapter_dim: int = 64,
        adapter_layers: list[int] | None = None,
        freeze_backbone: bool = True,
        img_size: tuple[int, int] = (224, 392),
        grad_checkpointing: bool = True,
        pretrained: bool = True,
    ):
        super().__init__()
        import timm

        self.num_frames = num_frames
        self.grad_checkpointing = grad_checkpointing

        self.vit = timm.create_model(
            vit_model, pretrained=pretrained, num_classes=0, dynamic_img_size=True)
        if freeze_backbone:
            for p in self.vit.parameters():
                p.requires_grad = False

        self.embed_dim = self.vit.embed_dim
        patch = getattr(self.vit.patch_embed, "patch_size", (14, 14))[0]
        self.patches_h = img_size[0] // patch
        self.patches_w = img_size[1] // patch
        self.num_patches = self.patches_h * self.patches_w

        if use_adapter:
            if adapter_layers is None:
                n_blocks = len(self.vit.blocks)
                adapter_layers = list(range(n_blocks // 2, n_blocks))
            self.adapters = nn.ModuleDict({
                f"layer_{i}": AdapterLayer(self.embed_dim, adapter_dim)
                for i in adapter_layers
            })
        else:
            self.adapters = None

        self.temporal_fusion = TemporalAttention(self.embed_dim)

    def _forward_with_adapters(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 3, H, W) -> (B, N_patches, D), CLS dropped."""
        x = self.vit.patch_embed(x)

        if hasattr(self.vit, "_pos_embed"):
            x = self.vit._pos_embed(x)
        else:
            cls = repeat(self.vit.cls_token, "1 1 d -> b 1 d", b=x.shape[0])
            x = torch.cat([cls, x], dim=1) + self.vit.pos_embed

        if hasattr(self.vit, "patch_drop"):
            x = self.vit.patch_drop(x)
        if hasattr(self.vit, "norm_pre"):
            x = self.vit.norm_pre(x)

        n_prefix = getattr(self.vit, "num_prefix_tokens", 1)
        for i, block in enumerate(self.vit.blocks):
            if self.grad_checkpointing and torch.is_grad_enabled():
                x = grad_checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
            key = f"layer_{i}"
            if self.adapters is not None and key in self.adapters:
                prefix, patches = x[:, :n_prefix], x[:, n_prefix:]
                x = torch.cat([prefix, self.adapters[key](patches)], dim=1)

        x = self.vit.norm(x)
        return x[:, n_prefix:, :]

    def forward(self, frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``frames``: (B, T, 3, H, W).

        Returns ``(fused, per_frame)`` with shapes ``(B, N, D)`` and
        ``(B, T, N, D)``.
        """
        B, T = frames.shape[:2]
        flat = rearrange(frames, "b t c h w -> (b t) c h w")
        feats = self._forward_with_adapters(flat)
        per_frame = rearrange(feats, "(b t) n d -> b t n d", b=B, t=T)
        return self.temporal_fusion(per_frame), per_frame

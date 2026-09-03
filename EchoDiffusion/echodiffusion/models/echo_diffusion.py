"""EchoDiffusion: audio-guided (optionally audio+vision) trajectory diffusion.

Conditioning pipeline::

    BEV sound field  (B, C, H, W) ┐
    DoA tokens       (B, T, K, F) ├─> concat -> fusion MLP -> global_cond ─┐
    ego / filter readout          ┤                                        │
    image frames     (B, T, 3, H, W) ┘  (optional, DINOv2 + ViT adapter)   │
                                                                          v
                        noised trajectory (B, horizon, 2) -> ConditionalUnet1D
                                                                          │
                                                                 eps / x0 ┘

The image branch is constructed only when ``data.use_image`` is set, so the
audio-only configuration neither imports ``timm`` nor allocates a ViT.  Both
branches feed the *same* fusion layer, which means an audio-only checkpoint and
an audio+image checkpoint differ only in the width of that layer -- switching
modality is a config change, not a code path.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .diffusion import GaussianDiffusion
from .encoders import BEVFieldEncoder, DoATokenEncoder, EgoEncoder, ImagePool
from .unet1d import ConditionalUnet1D


class EchoDiffusionPolicy(nn.Module):
    """Diffusion policy conditioned on sound-source evidence."""

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        mc = config.get("model", {})
        dc = config.get("data", {})
        bev_cfg = config.get("bev", {})

        self.horizon = int(dc.get("horizon", 20))
        self.traj_dim = int(mc.get("traj_dim", 2))
        self.use_image = bool(dc.get("use_image", False))

        # ── conditioning encoders ─────────────────────────────────────────
        history_len = int(bev_cfg.get("history_len", 4))
        enc = mc.get("encoders", {})

        self.bev_encoder = BEVFieldEncoder(
            in_channels=1 + history_len,
            out_dim=int(enc.get("bev_dim", 256)),
            widths=tuple(enc.get("bev_widths", (32, 64, 128))),
            pool=int(enc.get("bev_pool", 4)),
            dropout=float(enc.get("dropout", 0.0)),
        )
        self.doa_encoder = DoATokenEncoder(
            feature_dim=6,
            token_dim=int(enc.get("doa_token_dim", 128)),
            out_dim=int(enc.get("doa_dim", 128)),
            n_frames=int(dc.get("doa_frames", 4)),
        )
        self.ego_encoder = EgoEncoder(
            past_len=int(dc.get("past_len", 10)),
            past_dim=3,
            extra_dim=4,                      # field estimate: x, y, spread, conf
            out_dim=int(enc.get("ego_dim", 64)),
        )

        cond_in = (self.bev_encoder.out_dim + self.doa_encoder.out_dim
                   + self.ego_encoder.out_dim)

        self.vit = None
        self.image_pool = None
        if self.use_image:
            from .vit_adapter import ViTAdapter
            vit_cfg = mc.get("vit", {})
            self.vit = ViTAdapter(
                vit_model=vit_cfg.get("backbone", "vit_base_patch14_dinov2.lvd142m"),
                num_frames=int(dc.get("image_frames", 1)),
                use_adapter=bool(vit_cfg.get("use_adapter", True)),
                adapter_dim=int(vit_cfg.get("adapter_dim", 64)),
                adapter_layers=vit_cfg.get("adapter_layers"),
                freeze_backbone=bool(vit_cfg.get("freeze_backbone", True)),
                img_size=tuple(dc.get("image_size", (224, 392))),
                grad_checkpointing=bool(vit_cfg.get("grad_checkpointing", True)),
                pretrained=bool(vit_cfg.get("pretrained", True)),
            )
            self.image_pool = ImagePool(
                self.vit.embed_dim, out_dim=int(enc.get("image_dim", 256)))
            cond_in += self.image_pool.out_dim

        cond_dim = int(mc.get("cond_dim", 256))
        self.cond_fusion = nn.Sequential(
            nn.Linear(cond_in, cond_dim), nn.Mish(),
            nn.Linear(cond_dim, cond_dim),
        )
        self.cond_dim = cond_dim

        # ── auxiliary head ────────────────────────────────────────────────
        # Regressing the source position from the same conditioning vector is
        # a cheap, well-posed signal that forces the encoders to actually
        # localise rather than memorise "drive forward".  Off by default only
        # when no GT source pose exists.
        self.predict_source = bool(mc.get("predict_source", True))
        self.source_head = nn.Sequential(
            nn.Linear(cond_dim, 128), nn.Mish(), nn.Linear(128, 2),
        ) if self.predict_source else None

        # ── denoiser + schedule ───────────────────────────────────────────
        unet_cfg = mc.get("unet", {})
        self.unet = ConditionalUnet1D(
            input_dim=self.traj_dim,
            global_cond_dim=cond_dim,
            diffusion_step_embed_dim=int(unet_cfg.get("step_embed_dim", 128)),
            down_dims=tuple(unet_cfg.get("down_dims", (128, 256, 512))),
            kernel_size=int(unet_cfg.get("kernel_size", 5)),
            n_groups=int(unet_cfg.get("n_groups", 8)),
        )

        diff_cfg = config.get("diffusion", {})
        self.diffusion = GaussianDiffusion(
            num_timesteps=int(diff_cfg.get("num_timesteps", 100)),
            beta_schedule=diff_cfg.get("beta_schedule", "cosine"),
            beta_start=float(diff_cfg.get("beta_start", 1e-4)),
            beta_end=float(diff_cfg.get("beta_end", 0.02)),
            prediction_type=diff_cfg.get("prediction_type", "epsilon"),
            clip_sample=bool(diff_cfg.get("clip_sample", True)),
            clip_range=float(diff_cfg.get("clip_range", 1.5)),
        )
        self.num_inference_steps = int(diff_cfg.get("num_inference_steps", 16))
        self.ddim_eta = float(diff_cfg.get("ddim_eta", 0.0))

        # Classifier-free guidance: with this probability the conditioning is
        # replaced by a learned null embedding during training, which lets
        # sampling amplify the audio evidence at inference time.
        self.cond_dropout = float(mc.get("cond_dropout", 0.0))
        self.guidance_scale = float(mc.get("guidance_scale", 1.0))
        self.null_cond = nn.Parameter(torch.zeros(1, cond_dim)) \
            if self.cond_dropout > 0 else None

    # ── conditioning ──────────────────────────────────────────────────────

    def encode_conditioning(self, batch: dict) -> torch.Tensor:
        """Fuse every enabled modality into ``(B, cond_dim)``."""
        parts = [
            self.bev_encoder(batch["bev"]),
            self.doa_encoder(batch["doa"], batch["doa_mask"]),
            self.ego_encoder(batch["past"], batch["field_estimate"]),
        ]
        if self.use_image:
            if "image" not in batch:
                raise KeyError(
                    "data.use_image is true but the batch has no 'image' key -- "
                    "the episodes were prepared without camera frames")
            fused, _ = self.vit(batch["image"])
            parts.append(self.image_pool(fused))

        cond = self.cond_fusion(torch.cat(parts, dim=-1))

        if self.training and self.null_cond is not None:
            drop = (torch.rand(cond.shape[0], 1, device=cond.device)
                    < self.cond_dropout).float()
            cond = drop * self.null_cond.expand_as(cond) + (1 - drop) * cond
        return cond

    # ── training ──────────────────────────────────────────────────────────

    def compute_loss(self, batch: dict) -> dict:
        """DDPM denoising loss (+ auxiliary source regression).

        Windows without a valid future trajectory are masked out rather than
        dropped, so the batch shape stays static.
        """
        cond = self.encode_conditioning(batch)
        traj = batch["traj"]
        B = traj.shape[0]

        noise = torch.randn_like(traj)
        t = self.diffusion.sample_timesteps(B, traj.device)
        noisy = self.diffusion.q_sample(traj, t, noise)
        pred = self.unet(noisy, t, cond)
        target = self.diffusion.target_for(traj, noise)

        valid = batch.get("traj_valid")
        w = torch.ones(B, device=traj.device) if valid is None else valid.float()
        per_sample = ((pred - target) ** 2).mean(dim=(1, 2))
        denom = w.sum().clamp(min=1.0)
        diffusion_loss = (per_sample * w).sum() / denom

        out = {"diffusion_loss": diffusion_loss, "loss": diffusion_loss}

        if self.source_head is not None and "source_xy" in batch:
            sv = batch["source_valid"].float()
            src_pred = self.source_head(cond)
            src_err = ((src_pred - batch["source_xy"]) ** 2).mean(dim=-1)
            src_loss = (src_err * sv).sum() / sv.sum().clamp(min=1.0)
            weight = float(self.config.get("training", {}).get("source_loss_weight", 0.1))
            out["source_loss"] = src_loss
            out["loss"] = out["loss"] + weight * src_loss

        return out

    def forward(self, batch: dict) -> dict:
        return self.compute_loss(batch)

    # ── inference ─────────────────────────────────────────────────────────

    @torch.no_grad()
    def predict(self, batch: dict, num_steps: int | None = None,
                eta: float | None = None, n_samples: int = 1) -> torch.Tensor:
        """Sample trajectories.

        Returns ``(B, horizon, traj_dim)`` for ``n_samples == 1``, otherwise
        ``(n_samples, B, horizon, traj_dim)`` -- drawing several samples with
        ``eta > 0`` is how you inspect the policy's multimodality when the
        bearing alone is ambiguous.  Values are in normalised units; multiply
        by the dataset's ``traj_scale`` for metres.
        """
        self.eval()
        cond = self.encode_conditioning(batch)
        B = cond.shape[0]
        shape = (B, self.horizon, self.traj_dim)

        if self._cfg_active():
            # The guided wrapper closes over ``cond`` and the null embedding,
            # so the sampler passes no conditioning of its own.
            denoiser, cond_arg = self._guided_denoiser(cond), None
        else:
            denoiser, cond_arg = self.unet, cond

        outs = [
            self.diffusion.ddim_sample(
                denoiser, shape,
                global_cond=cond_arg,
                num_steps=num_steps or self.num_inference_steps,
                eta=self.ddim_eta if eta is None else eta,
                device=cond.device,
            )
            for _ in range(n_samples)
        ]
        return outs[0] if n_samples == 1 else torch.stack(outs)

    def _cfg_active(self) -> bool:
        return self.null_cond is not None and self.guidance_scale != 1.0

    def _guided_denoiser(self, cond: torch.Tensor):
        """Wrap the U-Net with classifier-free guidance.

        Exposes ``.parameters()`` so the sampler can still infer the device.
        """
        null = self.null_cond.expand_as(cond)
        scale = self.guidance_scale
        unet = self.unet

        class _Guided:
            def __call__(self, x, t, _ignored=None):
                uncond = unet(x, t, null)
                condit = unet(x, t, cond)
                return uncond + scale * (condit - uncond)

            @staticmethod
            def parameters():
                return unet.parameters()

        return _Guided()

    # ── introspection ─────────────────────────────────────────────────────

    def param_counts(self) -> dict[str, int]:
        def count(m) -> int:
            return 0 if m is None else sum(
                p.numel() for p in m.parameters() if p.requires_grad)
        return {
            "trainable_total": sum(p.numel() for p in self.parameters()
                                   if p.requires_grad),
            "total": sum(p.numel() for p in self.parameters()),
            "bev_encoder": count(self.bev_encoder),
            "doa_encoder": count(self.doa_encoder),
            "ego_encoder": count(self.ego_encoder),
            "vit_trainable": count(self.vit),
            "unet": count(self.unet),
        }

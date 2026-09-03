"""Tests for the denoiser, the diffusion schedule, and the full policy."""

from __future__ import annotations

import pytest
import torch

from echodiffusion.models.diffusion import GaussianDiffusion
from echodiffusion.models.echo_diffusion import EchoDiffusionPolicy
from echodiffusion.models.encoders import DoATokenEncoder
from echodiffusion.models.unet1d import ConditionalUnet1D

HORIZON = 20
BEV_CH = 5


def base_config(**overrides) -> dict:
    cfg = {
        "data": {"horizon": HORIZON, "past_len": 10, "doa_frames": 4,
                 "use_image": False},
        "bev": {"history_len": BEV_CH - 1},
        "model": {"cond_dim": 128,
                  "encoders": {"bev_dim": 128, "doa_dim": 64, "ego_dim": 32},
                  "unet": {"down_dims": [64, 128]}},
        "diffusion": {"num_timesteps": 50, "num_inference_steps": 5},
    }
    for key, value in overrides.items():
        cfg.setdefault(key, {}).update(value)
    return cfg


def make_batch(b: int = 3, horizon: int = HORIZON) -> dict:
    return {
        "bev": torch.rand(b, BEV_CH, 40, 40),
        "doa": torch.randn(b, 4, 8, 6),
        "doa_mask": torch.ones(b, 4, 8),
        "past": torch.randn(b, 10, 3),
        "field_estimate": torch.randn(b, 4),
        "traj": torch.randn(b, horizon, 2) * 0.3,
        "traj_valid": torch.ones(b),
        "source_xy": torch.randn(b, 2) * 0.3,
        "source_valid": torch.ones(b),
    }


# ── U-Net ─────────────────────────────────────────────────────────────────

def test_unet_preserves_shape():
    net = ConditionalUnet1D(input_dim=2, global_cond_dim=32, down_dims=(32, 64))
    x = torch.randn(2, HORIZON, 2)
    out = net(x, torch.tensor([3, 7]), torch.randn(2, 32))
    assert out.shape == x.shape


def test_unet_rejects_indivisible_horizon():
    """Three levels means two stride-2 convs, so the horizon must divide by 4."""
    net = ConditionalUnet1D(input_dim=2, global_cond_dim=16,
                            down_dims=(32, 64, 128))
    with pytest.raises(ValueError, match="divisible"):
        net(torch.randn(2, 18, 2), torch.tensor([1, 1]), torch.randn(2, 16))


def test_unet_conditioning_changes_the_output():
    net = ConditionalUnet1D(input_dim=2, global_cond_dim=32, down_dims=(32, 64))
    x = torch.randn(2, HORIZON, 2)
    t = torch.tensor([5, 5])
    a = net(x, t, torch.zeros(2, 32))
    b = net(x, t, torch.ones(2, 32))
    assert not torch.allclose(a, b)


# ── diffusion schedule ────────────────────────────────────────────────────

@pytest.mark.parametrize("schedule", ["cosine", "linear"])
def test_alphas_are_monotonic(schedule):
    d = GaussianDiffusion(num_timesteps=100, beta_schedule=schedule)
    ac = d.alphas_cumprod
    assert torch.all(ac[1:] <= ac[:-1] + 1e-6)   # non-increasing
    assert ac[0] < 1.0 and ac[-1] > 0.0


def test_q_sample_endpoints():
    d = GaussianDiffusion(num_timesteps=100, beta_schedule="cosine")
    x0 = torch.randn(4, HORIZON, 2)
    noise = torch.randn_like(x0)

    early = d.q_sample(x0, torch.zeros(4, dtype=torch.long), noise)
    late = d.q_sample(x0, torch.full((4,), 99, dtype=torch.long), noise)
    # t=0 stays close to the signal; t=T is dominated by noise.
    assert (early - x0).abs().mean() < (late - x0).abs().mean()


def test_prediction_type_selects_the_target():
    x0, noise = torch.randn(2, HORIZON, 2), torch.randn(2, HORIZON, 2)
    eps_d = GaussianDiffusion(prediction_type="epsilon")
    sam_d = GaussianDiffusion(prediction_type="sample")
    assert torch.equal(eps_d.target_for(x0, noise), noise)
    assert torch.equal(sam_d.target_for(x0, noise), x0)


def test_ddim_recovers_x0_from_a_perfect_denoiser():
    """With an oracle that always returns the true trajectory, the deterministic
    sampler must converge to it."""
    target = torch.randn(2, HORIZON, 2) * 0.4
    d = GaussianDiffusion(num_timesteps=100, prediction_type="sample",
                          clip_sample=False)

    class Oracle(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy = torch.nn.Parameter(torch.zeros(1))

        def forward(self, x, t, cond=None):
            return target.to(x.device)

    out = d.ddim_sample(Oracle(), target.shape, num_steps=20,
                        eta=0.0, device=torch.device("cpu"))
    assert torch.allclose(out, target, atol=1e-3)


# ── encoders ──────────────────────────────────────────────────────────────

def test_doa_encoder_ignores_masked_slots():
    enc = DoATokenEncoder(feature_dim=6, token_dim=16, out_dim=16, n_frames=2)
    enc.eval()
    doa = torch.randn(1, 2, 4, 6)
    mask = torch.tensor([[[1.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]])

    with torch.no_grad():
        a = enc(doa, mask)
        perturbed = doa.clone()
        perturbed[:, :, 2:] = 99.0      # only masked-out slots change
        b = enc(perturbed, mask)
    assert torch.allclose(a, b, atol=1e-5)


def test_doa_encoder_survives_an_all_empty_frame():
    enc = DoATokenEncoder(feature_dim=6, token_dim=16, out_dim=16, n_frames=2)
    out = enc(torch.randn(2, 2, 4, 6), torch.zeros(2, 2, 4))
    assert torch.isfinite(out).all()


# ── policy ────────────────────────────────────────────────────────────────

def test_loss_and_backward():
    model = EchoDiffusionPolicy(base_config())
    model.train()
    out = model.compute_loss(make_batch())
    assert {"loss", "diffusion_loss", "source_loss"} <= set(out)
    out["loss"].backward()
    assert any(p.grad is not None for p in model.parameters() if p.requires_grad)


def test_invalid_windows_are_masked_out():
    """A batch where nothing is valid must produce a finite, zero-ish loss."""
    model = EchoDiffusionPolicy(base_config())
    batch = make_batch()
    batch["traj_valid"] = torch.zeros(3)
    batch["source_valid"] = torch.zeros(3)
    out = model.compute_loss(batch)
    assert torch.isfinite(out["loss"]) and out["loss"].item() == pytest.approx(0.0)


def test_predict_shapes_and_determinism():
    model = EchoDiffusionPolicy(base_config())
    batch = make_batch()

    torch.manual_seed(0)
    a = model.predict(batch)
    torch.manual_seed(0)
    b = model.predict(batch)
    assert a.shape == (3, HORIZON, 2)
    # eta=0 is deterministic given the same initial noise.
    assert torch.allclose(a, b)

    multi = model.predict(batch, n_samples=3, eta=1.0)
    assert multi.shape == (3, 3, HORIZON, 2)


def test_classifier_free_guidance_path_runs():
    cfg = base_config()
    cfg["model"]["cond_dropout"] = 0.2
    cfg["model"]["guidance_scale"] = 2.0
    model = EchoDiffusionPolicy(cfg)
    assert model.null_cond is not None and model._cfg_active()

    model.train()
    model.compute_loss(make_batch())["loss"].backward()

    out = model.predict(make_batch())
    assert out.shape == (3, HORIZON, 2) and torch.isfinite(out).all()


def test_missing_image_raises_a_clear_error():
    cfg = base_config()
    cfg["data"]["use_image"] = True
    pytest.importorskip("timm")
    cfg["model"]["vit"] = {"pretrained": False, "adapter_layers": [11]}
    model = EchoDiffusionPolicy(cfg)
    with pytest.raises(KeyError, match="use_image"):
        model.compute_loss(make_batch())

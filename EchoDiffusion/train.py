#!/usr/bin/env python3
"""Train EchoDiffusion.

    python train.py --config configs/audio_only.yaml
    python train.py --config configs/audio_image.yaml --resume checkpoints/.../last.pt
"""

from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")

# comet_ml installs its auto-logging hooks at import time and can only patch
# frameworks imported after it, so it has to come before torch -- otherwise it
# warns and silently logs less.  Optional: training runs fine without it.
try:
    import comet_ml  # noqa: F401
except ImportError:
    pass

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml

from echodiffusion.data.dataset import create_dataloaders
from echodiffusion.models.echo_diffusion import EchoDiffusionPolicy
from echodiffusion.training.trainer import Trainer
from echodiffusion.utils.comet_logger import CometLogger


def parse_args():
    p = argparse.ArgumentParser(description="Train EchoDiffusion")
    p.add_argument("--config", default="configs/audio_only.yaml")
    p.add_argument("--resume", default=None, help="checkpoint to resume from")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--comet-api-key", default=None)
    p.add_argument("--comet-project", default=None)
    p.add_argument("--no-comet", action="store_true",
                   help="disable Comet even if logging.use_comet is true")
    p.add_argument("--epochs", type=int, default=None,
                   help="override training.num_epochs (handy for smoke tests)")
    p.add_argument("--name", default=None, help="override the run name")
    p.add_argument("--train-dir", default=None, help="override paths.train_dir")
    p.add_argument("--val-dir", default=None, help="override paths.val_dir")
    p.add_argument("--output-dir", default=None, help="override paths.output_dir")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    return p.parse_args()


def apply_overrides(config: dict, args) -> None:
    """Apply CLI overrides onto the loaded config, in place."""
    paths = config.setdefault("paths", {})
    tc = config.setdefault("training", {})
    for value, section, key in (
        (args.train_dir, paths, "train_dir"),
        (args.val_dir, paths, "val_dir"),
        (args.output_dir, paths, "output_dir"),
        (args.epochs, tc, "num_epochs"),
        (args.batch_size, tc, "batch_size"),
        (args.num_workers, tc, "num_workers"),
    ):
        if value is not None:
            section[key] = value


def build_run_name(config: dict, config_path: str) -> tuple[str, list[str]]:
    """Derive a self-describing run name and filter tags from the config.

    The name encodes the knobs that actually change results, so two runs are
    distinguishable in the Comet UI without opening either:

        audio-only_h20_bev6.0m-hl6.0_eps-T100_vicon_s42

    i.e. modality, prediction horizon, BEV extent and evidence half-life,
    diffusion prediction type and step count, pose source, seed.
    """
    dc = config.get("data", {})
    bev = config.get("bev", {})
    diff = config.get("diffusion", {})
    model = config.get("model", {})

    modality = "audio+image" if dc.get("use_image") else "audio-only"
    horizon = dc.get("horizon", 20)
    rng_m = bev.get("range_m", 6.0)
    half_life = bev.get("decay_half_life_s", 6.0)
    pred = diff.get("prediction_type", "epsilon")[:3]
    steps = diff.get("num_timesteps", 100)
    pose = config.get("poses", {}).get("source", "auto")
    seed = config.get("seed", 0)

    parts = [modality, f"h{horizon}", f"bev{rng_m}m-hl{half_life}",
             f"{pred}-T{steps}", pose, f"s{seed}"]
    if model.get("cond_dropout", 0) > 0:
        parts.insert(-1, f"cfg{model.get('guidance_scale', 1.0)}")
    name = "_".join(str(p) for p in parts)

    tags = [
        Path(config_path).stem, modality, f"pose_{pose}", f"seed{seed}",
        f"horizon{horizon}", pred,
        f"bev_range{rng_m}", f"half_life{half_life}",
        f"elev_gate{bev.get('max_elevation_deg', 78.0)}",
    ]
    if model.get("cond_dropout", 0) > 0:
        tags.append("cfg-guidance")
    if not model.get("predict_source", True):
        tags.append("no-source-head")
    return name, tags


def main():
    args = parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    print(f"Config: {args.config}")
    apply_overrides(config, args)

    seed = int(config.get("seed", 0))
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── logging ───────────────────────────────────────────────────────────
    logger = None
    log_cfg = config.get("logging", {})
    if log_cfg.get("use_comet", False) and not args.no_comet:
        run_name, tags = build_run_name(config, args.config)
        if args.name:
            run_name = args.name
        logger = CometLogger(
            api_key=args.comet_api_key or log_cfg.get("comet_api_key"),
            project_name=args.comet_project or log_cfg.get(
                "comet_project", "echodiffusion"),
            workspace=log_cfg.get("comet_workspace"),
            config=config,
            experiment_name=run_name,
            tags=tags,
            viz_dir=Path(config.get("paths", {}).get(
                "output_dir", "checkpoints/run")) / "viz",
        )
        if logger.enabled:
            print(f"Comet run '{run_name}'  tags={tags}")

    # ── data ──────────────────────────────────────────────────────────────
    print("Loading episodes ...")
    train_loader, val_loader = create_dataloaders(config)
    print(f"train: {len(train_loader.dataset)} windows "
          f"({len(train_loader)} batches)  |  "
          f"val: {len(val_loader.dataset)} windows "
          f"({len(val_loader)} batches)")
    print(f"traj_scale: {train_loader.dataset.traj_scale:.3f} m per unit")

    # ── model ─────────────────────────────────────────────────────────────
    model = EchoDiffusionPolicy(config)
    counts = model.param_counts()
    print(f"Model: {counts['trainable_total'] / 1e6:.2f}M trainable "
          f"/ {counts['total'] / 1e6:.2f}M total")
    for k, v in counts.items():
        if k not in ("trainable_total", "total") and v:
            print(f"    {k:<16} {v / 1e6:6.2f}M")
    if logger:
        logger.log_parameters({"params": counts,
                               "traj_scale": train_loader.dataset.traj_scale})

    # ── train ─────────────────────────────────────────────────────────────
    trainer = Trainer(model, train_loader, val_loader, config,
                      logger=logger, device=device, resume=args.resume)
    try:
        trainer.train()
    finally:
        if logger:
            logger.end()


if __name__ == "__main__":
    main()

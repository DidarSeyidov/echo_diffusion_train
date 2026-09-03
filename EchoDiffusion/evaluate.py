#!/usr/bin/env python3
"""Evaluate a trained EchoDiffusion checkpoint.

    python evaluate.py --checkpoint checkpoints/audio_only/best.pt
    python evaluate.py --checkpoint best.pt --episodes data/episodes_synth/val --viz out/

Reports, in metres:

* **ADE / FDE** -- average and final displacement error against the expert.
* **endpoint bearing error** -- angle between the predicted endpoint and the
  true source direction.  This is the metric that actually tests whether the
  audio is being used: a policy that ignores the sound and drives straight can
  post a respectable ADE while failing this badly.
* **progress ratio** -- how much of the distance to the source the predicted
  path closes, relative to the expert.  1.0 means it approaches as decisively.
* **field spread** -- the filter's own localisation uncertainty, reported
  separately for near-stationary and moving windows to show whether motion is
  buying certainty.
"""

from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from echodiffusion.data.dataset import DataConfig, EchoTrajectoryDataset
from echodiffusion.data.episode import find_episodes
from echodiffusion.models.echo_diffusion import EchoDiffusionPolicy
from echodiffusion.utils.geometry import wrap_angle

#: Minimum predicted endpoint displacement (metres) for the bearing and
#: progress metrics to be meaningful.  Below this the endpoint is effectively
#: at the origin and its direction carries no information.
MIN_ENDPOINT_M = 0.05


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", default=None,
                   help="defaults to the config stored in the checkpoint")
    p.add_argument("--episodes", default=None,
                   help="episode root to evaluate on (default: paths.val_dir)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--num-steps", type=int, default=None,
                   help="DDIM steps (default: the config's)")
    p.add_argument("--n-samples", type=int, default=1,
                   help=">1 with --eta>0 measures multimodality")
    p.add_argument("--eta", type=float, default=None)
    p.add_argument("--use-ema", action="store_true", default=True)
    p.add_argument("--no-ema", dest="use_ema", action="store_false")
    p.add_argument("--viz", default=None, help="directory for sample plots")
    p.add_argument("--viz-count", type=int, default=8)
    p.add_argument("--gpu", type=int, default=0)
    return p.parse_args()


@torch.no_grad()
def evaluate(model, loader, device, traj_scale, goal_scale,
             num_steps=None, eta=None, n_samples=1):
    model.eval()
    ade, fde, bearing, progress, spreads = [], [], [], [], []
    moved, diversity = [], []
    collected = []

    for batch in tqdm(loader, desc="eval"):
        batch = {k: v.to(device) if torch.is_tensor(v) else v
                 for k, v in batch.items()}

        pred = model.predict(batch, num_steps=num_steps, eta=eta,
                             n_samples=n_samples).float()
        if n_samples > 1:
            # Spread across samples measures how multimodal the policy is.
            diversity.append(float(pred.std(0).mean()) * traj_scale)
            pred = pred.mean(0)

        target = batch["traj"].float()
        valid = batch["traj_valid"].bool()
        if not valid.any():
            continue

        p_m = pred[valid] * traj_scale                 # metres
        t_m = target[valid] * traj_scale
        dist = (p_m - t_m).norm(dim=-1)
        ade.append(dist.mean(dim=-1).cpu().numpy())
        fde.append(dist[:, -1].cpu().numpy())

        # How far the window travelled -- used to split static vs. moving.
        moved.append(batch["past"][valid][:, 0, :2].norm(dim=-1).cpu().numpy())
        spreads.append((batch["field_estimate"][valid][:, 2]
                        * goal_scale).cpu().numpy())

        sv = batch["source_valid"].bool() & valid
        if sv.any():
            src = batch["source_xy"][sv].float() * goal_scale
            end = (pred[sv] * traj_scale)[:, -1]

            # A near-stationary prediction has its endpoint at the origin, so
            # its bearing is pure numerical noise.  Those windows (the robot
            # has already arrived, or is pausing) would otherwise dominate the
            # mean while telling us nothing about whether the audio was used.
            moving = end.norm(dim=-1) > MIN_ENDPOINT_M
            if moving.any():
                a = torch.atan2(end[moving, 1], end[moving, 0])
                b = torch.atan2(src[moving, 1], src[moving, 0])
                bearing.append(np.degrees(np.abs(wrap_angle(
                    (a - b).cpu().numpy()))))

            # Progress: distance closed toward the source, predicted vs expert.
            d0 = src.norm(dim=-1)
            d_pred = (src - end).norm(dim=-1)
            d_gt = (src - (target[sv].float() * traj_scale)[:, -1]).norm(dim=-1)
            closed_pred = (d0 - d_pred).cpu().numpy()
            closed_gt = (d0 - d_gt).cpu().numpy()
            # Only meaningful where the expert itself made real progress.
            ok = np.abs(closed_gt) > MIN_ENDPOINT_M
            if ok.any():
                progress.append(closed_pred[ok] / closed_gt[ok])

        if len(collected) < 1:
            doa_theta, doa_r = dominant_bearing_from_tokens(
                batch["doa"][:, 0].float().cpu().numpy(),
                batch["doa_mask"][:, 0].float().cpu().numpy())
            collected.append({
                "bev": batch["bev"].float().cpu().numpy(),
                "pred": pred.cpu().numpy(),
                "target": target.cpu().numpy(),
                "source_xy": batch["source_xy"].float().cpu().numpy(),
                "source_valid": batch["source_valid"].cpu().numpy(),
                "doa_bearing": doa_theta,
                "doa_strength": doa_r,
            })

    def cat(x):
        return np.concatenate(x) if x else np.zeros(0)

    ade, fde = cat(ade), cat(fde)
    bearing, progress = cat(bearing), cat(progress)
    moved, spreads = cat(moved), cat(spreads)

    metrics = {
        "n_windows": int(ade.size),
        "ade_m": float(ade.mean()) if ade.size else float("nan"),
        "fde_m": float(fde.mean()) if fde.size else float("nan"),
        "ade_median_m": float(np.median(ade)) if ade.size else float("nan"),
    }
    if bearing.size:
        metrics["bearing_n"] = int(bearing.size)
        metrics["endpoint_bearing_err_deg"] = float(bearing.mean())
        metrics["endpoint_bearing_median_deg"] = float(np.median(bearing))
        metrics["bearing_within_30deg_pct"] = float((bearing < 30).mean() * 100)
    if progress.size:
        metrics["progress_ratio"] = float(np.median(progress))
    if spreads.size and moved.size:
        # Windows whose recent motion was small vs. large: if triangulation
        # works, the moving group should show a tighter posterior.
        thresh = float(np.median(moved))
        static, mobile = moved <= thresh, moved > thresh
        if static.any():
            metrics["field_spread_static_m"] = float(spreads[static].mean())
        if mobile.any():
            metrics["field_spread_moving_m"] = float(spreads[mobile].mean())
    if diversity:
        metrics["sample_diversity_m"] = float(np.mean(diversity))

    return metrics, (collected[0] if collected else None)


def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if args.config:
        with open(args.config) as f:
            config = yaml.safe_load(f)
    else:
        config = ckpt["config"]
    traj_scale = float(ckpt.get("traj_scale", 1.0))
    print(f"Checkpoint: {args.checkpoint}  (epoch {ckpt.get('epoch')}, "
          f"traj_scale {traj_scale:.3f} m)")

    cfg = DataConfig.from_config(config)
    root = args.episodes or config["paths"]["val_dir"]
    episodes = find_episodes(root)
    if not episodes:
        raise SystemExit(f"no episodes under {root}")

    dataset = EchoTrajectoryDataset(episodes, cfg, name="eval",
                                    traj_scale=traj_scale)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers)
    print(f"{len(dataset)} windows from {len(episodes)} episodes")

    model = EchoDiffusionPolicy(config).to(device)
    state = ckpt["ema"] if (args.use_ema and ckpt.get("ema")) else ckpt["model"]
    model.load_state_dict(state)
    print(f"Weights: {'EMA' if (args.use_ema and ckpt.get('ema')) else 'raw'}")

    metrics, samples = evaluate(
        model, loader, device, traj_scale, cfg.goal_scale,
        num_steps=args.num_steps, eta=args.eta, n_samples=args.n_samples)

    print("\n" + "=" * 52)
    for k, v in metrics.items():
        print(f"  {k:<32} {v:10.4f}" if isinstance(v, float)
              else f"  {k:<32} {v:>10}")
    print("=" * 52)

    out_dir = Path(args.viz) if args.viz else Path(args.checkpoint).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "eval_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nmetrics -> {out_dir / 'eval_metrics.json'}")

    if args.viz and samples is not None:
        from echodiffusion.utils.comet_logger import CometLogger
        logger = CometLogger(api_key="", viz_dir=out_dir)     # disk-only
        logger.log_trajectory_samples(
            samples["bev"], samples["pred"], samples["target"],
            traj_scale=traj_scale, field_range=dataset.bev_cfg.range_m,
            step=0, name="eval_traj_vs_doa", max_items=args.viz_count,
            source_xy=samples["source_xy"] if samples["source_valid"].any() else None,
            goal_scale=cfg.goal_scale,
            doa_bearing=samples["doa_bearing"],
            doa_strength=samples["doa_strength"])
        print(f"plots  -> {out_dir}")


if __name__ == "__main__":
    main()

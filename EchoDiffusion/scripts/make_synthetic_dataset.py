#!/usr/bin/env python3
"""Generate synthetic episodes so the full stack is runnable before the real
recordings carry odometry / Vicon / camera.

    python scripts/make_synthetic_dataset.py --out data/episodes_synth
    python train.py --config configs/audio_only.yaml   # after pointing paths at it

The DoA statistics mirror ``session_audio/session_01`` (clutter peaks, a
near-vertical phantom SST track, high source elevation), so a model that
trains here is exercising the same failure modes the real data will present.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from echodiffusion.data.synthetic import generate_dataset


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="data/episodes_synth")
    p.add_argument("--n-train", type=int, default=120)
    p.add_argument("--n-val", type=int, default=24)
    p.add_argument("--duration", type=float, default=20.0, help="seconds/episode")
    p.add_argument("--rate", type=float, default=10.0, help="sample rate (Hz)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--bearing-noise-deg", type=float, default=4.0)
    p.add_argument("--dropout-prob", type=float, default=0.05,
                   help="per-step chance the source is silent")
    args = p.parse_args()

    print(f"Generating {args.n_train} train + {args.n_val} val episodes "
          f"({args.duration}s @ {args.rate} Hz) -> {args.out}")
    generate_dataset(
        args.out,
        n_train=args.n_train,
        n_val=args.n_val,
        seed=args.seed,
        duration_s=args.duration,
        sample_rate_hz=args.rate,
        bearing_noise_deg=args.bearing_noise_deg,
        dropout_prob=args.dropout_prob,
    )
    print("\nNow point the training config at it:")
    print(f"  paths.train_dir: {args.out}/train")
    print(f"  paths.val_dir:   {args.out}/val")


if __name__ == "__main__":
    main()

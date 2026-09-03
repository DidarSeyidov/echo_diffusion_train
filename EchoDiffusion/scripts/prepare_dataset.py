#!/usr/bin/env python3
"""Convert rosbag2 sessions into training episodes.

    python scripts/prepare_dataset.py --config configs/prepare.yaml
    python scripts/prepare_dataset.py --bag-root /path/to/bags --out data/episodes

Sessions are split into train/val **by session**, never by window, so no
trajectory can appear on both sides of the split.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from echodiffusion.data.bag_to_episode import build_episode_from_bag
from echodiffusion.data.rosbag_reader import find_sessions


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/prepare.yaml")
    p.add_argument("--bag-root", default=None, help="override dataset.bag_root")
    p.add_argument("--out", default=None, help="override dataset.out_root")
    p.add_argument("--rate", type=float, default=None,
                   help="override dataset.sample_rate_hz")
    p.add_argument("--no-images", action="store_true")
    args = p.parse_args()

    cfg = {}
    if Path(args.config).exists():
        with open(args.config) as f:
            cfg = yaml.safe_load(f) or {}
    dc = cfg.get("dataset", {})

    bag_root = Path(args.bag_root or dc.get("bag_root", "."))
    out_root = Path(args.out or dc.get("out_root", "data/episodes"))
    rate = float(args.rate or dc.get("sample_rate_hz", 10.0))
    save_images = not args.no_images and bool(dc.get("save_images", True))

    static_src = dc.get("static_source_pose")
    if static_src is not None:
        static_src = tuple(float(v) for v in static_src)

    sessions = find_sessions(bag_root)
    if not sessions:
        raise SystemExit(f"no rosbag2 sessions found under {bag_root}")
    print(f"Found {len(sessions)} session(s) under {bag_root}\n")

    # Session-level split.
    frac = float(dc.get("val_fraction", 0.2))
    rng = np.random.default_rng(int(dc.get("seed", 42)))
    order = rng.permutation(len(sessions))
    n_val = int(round(len(sessions) * frac)) if len(sessions) > 1 else 0
    val_ids = set(order[:n_val].tolist())

    ok, failed = 0, []
    for i, session in enumerate(sessions):
        split = "val" if i in val_ids else "train"
        out_dir = out_root / split / session.name
        try:
            build_episode_from_bag(
                session, out_dir,
                sample_rate_hz=rate,
                topics=dc.get("topics"),
                pose_source=dc.get("pose_source", "auto"),
                static_source_pose=static_src,
                save_images=save_images,
                image_quality=int(dc.get("image_quality", 92)),
                name=session.name,
            )
            ok += 1
        except Exception as exc:                          # noqa: BLE001
            # One malformed session should not abandon the rest of the batch.
            print(f"  !! {session.name} failed: {type(exc).__name__}: {exc}")
            failed.append(session.name)
        print()

    print(f"Prepared {ok}/{len(sessions)} sessions -> {out_root}")
    if failed:
        print(f"Failed: {', '.join(failed)}")
    if n_val == 0 and len(sessions) >= 1:
        print("\nNOTE: only one session, so everything went to train/. Record "
              "more sessions, or point paths.train_dir and paths.val_dir at "
              "the same directory for a smoke test.")


if __name__ == "__main__":
    main()

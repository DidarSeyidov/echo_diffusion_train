#!/usr/bin/env python3
"""Inspect a rosbag2 session: topics, rates, and ODAS bearing statistics.

    python scripts/inspect_bag.py /home/zhura/datasets/echodiffusion
    python scripts/inspect_bag.py <session_dir> --plot bearings.png

Run this first on any new recording.  The elevation histogram in particular
tells you whether the array extrinsics in the config are right: for a flat
(z-up) mount and a source at robot height, elevations should cluster near 0.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from echodiffusion.audio.odas import (ArrayExtrinsics, ssl_to_observation,
                                      sst_to_observation)
from echodiffusion.data.rosbag_reader import Rosbag2Reader, find_sessions


def pct(a: np.ndarray, label: str, unit: str = "") -> str:
    if a.size == 0:
        return f"    {label:<12} (none)"
    q = np.percentile(a, [0, 5, 25, 50, 75, 95, 100])
    return (f"    {label:<12} min={q[0]:7.2f} p5={q[1]:7.2f} p25={q[2]:7.2f} "
            f"med={q[3]:7.2f} p75={q[4]:7.2f} p95={q[5]:7.2f} "
            f"max={q[6]:7.2f} {unit}")


def analyse(session: Path, extrinsics: ArrayExtrinsics, plot: str | None = None):
    reader = Rosbag2Reader(session)
    print("=" * 78)
    print(reader.summary())

    ssl_az, ssl_el, ssl_w = [], [], []
    sst_rows: dict[int, list] = {}

    for m in reader.read(topics=["/ssl", "/sst"]):
        if m.msg_type.endswith("OdasSslArrayStamped"):
            obs = ssl_to_observation(m.msg["stamp"], m.msg["sources"], extrinsics)
            ssl_az.append(obs.azimuth)
            ssl_el.append(obs.elevation)
            ssl_w.append(obs.weight)
        else:
            obs = sst_to_observation(m.msg["stamp"], m.msg["sources"], extrinsics)
            for tid, az, el, w in zip(obs.track_id, obs.azimuth,
                                      obs.elevation, obs.weight):
                sst_rows.setdefault(int(tid), []).append((az, el, w))

    if ssl_az:
        az = np.degrees(np.concatenate(ssl_az))
        el = np.degrees(np.concatenate(ssl_el))
        w = np.concatenate(ssl_w)
        print(f"\n  SSL potentials: {az.size}")
        print(pct(az, "azimuth", "deg"))
        print(pct(el, "elevation", "deg"))
        print(pct(w, "energy"))
        strong = w > np.percentile(w, 75)
        if strong.any():
            print(pct(az[strong], "az (top-25%)", "deg"))
            print(pct(el[strong], "el (top-25%)", "deg"))

    if sst_rows:
        print(f"\n  SST tracks: {len(sst_rows)}")
        for tid, rows in sorted(sst_rows.items(), key=lambda kv: -len(kv[1])):
            arr = np.array(rows)
            az, el, act = np.degrees(arr[:, 0]), np.degrees(arr[:, 1]), arr[:, 2]
            live = act > 0.1
            print(f"    id={tid:<6} n={len(arr):<6} active={live.mean() * 100:5.1f}%  "
                  f"az[{az.min():7.1f},{az.max():7.1f}] "
                  f"el[{el.min():6.1f},{el.max():6.1f}] "
                  f"act_mean={act.mean():.2f}")
            if live.sum() > 10:
                print(f"           while active: az med={np.median(az[live]):7.1f} "
                      f"el med={np.median(el[live]):6.1f}")

    if plot and ssl_az:
        _plot(np.degrees(np.concatenate(ssl_az)),
              np.degrees(np.concatenate(ssl_el)),
              np.concatenate(ssl_w), plot, session.name)


def _plot(az, el, w, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    axes[0].hist(az, bins=72, color="#38bdf8")
    axes[0].set_xlabel("azimuth (deg)"); axes[0].set_ylabel("count")
    axes[1].hist(el, bins=72, color="#f472b6")
    axes[1].set_xlabel("elevation (deg)")
    sc = axes[2].scatter(az, el, c=w, s=3, cmap="magma", alpha=0.5)
    axes[2].set_xlabel("azimuth (deg)"); axes[2].set_ylabel("elevation (deg)")
    fig.colorbar(sc, ax=axes[2], label="energy")
    fig.suptitle(f"ODAS SSL bearings -- {title}")
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    print(f"\n  plot -> {path}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("path", help="bag session directory, or a root to search")
    p.add_argument("--plot", default=None, help="write a bearing-histogram PNG")
    p.add_argument("--azimuth-offset-deg", type=float, default=0.0)
    p.add_argument("--rotation-rpy-deg", type=float, nargs=3, default=(0, 0, 0))
    args = p.parse_args()

    extrinsics = ArrayExtrinsics(
        rotation_rpy_deg=tuple(args.rotation_rpy_deg),
        azimuth_offset_deg=args.azimuth_offset_deg)

    root = Path(args.path)
    sessions = [root] if (root / "metadata.yaml").exists() else find_sessions(root)
    if not sessions:
        raise SystemExit(f"no rosbag2 sessions found under {root}")

    for s in sessions:
        analyse(s, extrinsics, args.plot)


if __name__ == "__main__":
    main()

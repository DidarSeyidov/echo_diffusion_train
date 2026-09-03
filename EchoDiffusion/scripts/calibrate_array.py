#!/usr/bin/env python3
"""Solve for the microphone array's azimuth offset against a known bearing.

Two modes:

**Known static bearing** -- put the speaker at a measured bearing from the
robot, record a stationary session, then::

    python scripts/calibrate_array.py <session_dir> --true-bearing-deg 45

The reported offset is what to paste into ``array.azimuth_offset_deg``.

**Against Vicon** -- once the GT source pose and robot pose are in the bag,
the true bearing is known at every instant::

    python scripts/calibrate_array.py <episode_dir> --from-episode

This fits offset *and* checks whether the azimuth needs mirroring, by
comparing the circular residual for both handedness hypotheses.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from echodiffusion.audio.odas import (ArrayExtrinsics, ssl_to_observation,
                                      sst_to_observation)
from echodiffusion.data.episode import Episode
from echodiffusion.data.rosbag_reader import Rosbag2Reader
from echodiffusion.utils.geometry import wrap_angle, world_to_body


def circular_fit(measured: np.ndarray, truth: np.ndarray,
                 weights: np.ndarray) -> tuple[float, float]:
    """Weighted circular mean of (truth - measured), plus the residual spread.

    Returns ``(offset_rad, residual_rad)``.  Averaging the *difference* on the
    unit circle avoids the wrap-around bias a plain arithmetic mean would
    introduce near +/-pi.
    """
    d = wrap_angle(truth - measured)
    c = float(np.sum(weights * np.cos(d)))
    s = float(np.sum(weights * np.sin(d)))
    offset = float(np.arctan2(s, c))
    r = np.hypot(c, s) / max(weights.sum(), 1e-9)
    # Circular standard deviation from the resultant length.
    residual = float(np.sqrt(-2.0 * np.log(np.clip(r, 1e-9, 1.0))))
    return offset, residual


def collect_from_bag(session: Path, min_weight: float, max_elev: float):
    reader = Rosbag2Reader(session)
    identity = ArrayExtrinsics()
    az, w = [], []
    for m in reader.read(topics=["/ssl", "/sst"]):
        to_obs = (ssl_to_observation if m.msg_type.endswith("SslArrayStamped")
                  else sst_to_observation)
        obs = to_obs(m.msg["stamp"], m.msg["sources"], identity)
        obs = obs.filtered(min_weight, max_elev)
        if len(obs):
            az.append(obs.azimuth)
            w.append(obs.weight)
    if not az:
        raise SystemExit("no detections passed the weight/elevation filter")
    return np.concatenate(az), np.concatenate(w)


def collect_from_episode(ep_dir: Path, min_weight: float, max_elev: float,
                         pose_source: str):
    ep = Episode.load(ep_dir)
    poses = ep.pose(pose_source)
    identity = ArrayExtrinsics()

    measured, truth, weights = [], [], []
    for i in range(len(ep)):
        if not (np.isfinite(poses[i]).all() and ep.source_valid[i]):
            continue
        rel = world_to_body(ep.source_pose[i], poses[i])
        true_bearing = float(np.arctan2(rel[1], rel[0]))

        parts = []
        k = int(ep.ssl_n[i])
        if k:
            parts.append(ssl_to_observation(float(ep.t[i]), ep.ssl[i, :k], identity))
        k = int(ep.sst_n[i])
        if k:
            parts.append(sst_to_observation(float(ep.t[i]), ep.sst[i, :k], identity))
        for obs in parts:
            obs = obs.filtered(min_weight, max_elev)
            for a, wt in zip(obs.azimuth, obs.weight):
                measured.append(a)
                truth.append(true_bearing)
                weights.append(wt)

    if not measured:
        raise SystemExit(
            "no samples with both a valid pose and a GT source position -- "
            "this episode needs odometry/Vicon and vicon_source recorded")
    return (np.array(measured), np.array(truth), np.array(weights))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("path", help="bag session directory, or an episode directory")
    p.add_argument("--true-bearing-deg", type=float, default=None,
                   help="known constant source bearing (robot frame, CCW from fwd)")
    p.add_argument("--from-episode", action="store_true",
                   help="derive the true bearing from the episode's GT poses")
    p.add_argument("--pose-source", default="auto")
    p.add_argument("--min-weight", type=float, default=0.15,
                   help="ignore weak detections -- clutter has no bearing to fit")
    p.add_argument("--max-elevation-deg", type=float, default=85.0)
    args = p.parse_args()

    path = Path(args.path)

    if args.from_episode:
        measured, truth, weights = collect_from_episode(
            path, args.min_weight, args.max_elevation_deg, args.pose_source)
    elif args.true_bearing_deg is not None:
        measured, weights = collect_from_bag(
            path, args.min_weight, args.max_elevation_deg)
        truth = np.full_like(measured, np.radians(args.true_bearing_deg))
    else:
        raise SystemExit("pass --true-bearing-deg or --from-episode")

    print(f"samples: {measured.size}  (weight >= {args.min_weight}, "
          f"elevation <= {args.max_elevation_deg} deg)\n")

    best = None
    for flip in (False, True):
        m = -measured if flip else measured
        offset, residual = circular_fit(m, truth, weights)
        print(f"  flip_azimuth={str(flip):<5}  "
              f"azimuth_offset_deg={np.degrees(offset):8.2f}  "
              f"residual={np.degrees(residual):6.2f} deg")
        if best is None or residual < best[2]:
            best = (flip, offset, residual)

    flip, offset, residual = best
    print("\nBest fit -- paste into the training config:\n")
    print("array:")
    print("  rotation_rpy_deg: [0.0, 0.0, 0.0]")
    print(f"  azimuth_offset_deg: {np.degrees(offset):.2f}")
    print(f"  flip_azimuth: {str(flip).lower()}")
    if residual > np.radians(30):
        print(f"\n! residual is {np.degrees(residual):.1f} deg -- that is a poor "
              "fit. The array is probably not flat/z-up, or the detections are "
              "dominated by reflections. Check the elevation histogram from "
              "scripts/inspect_bag.py before trusting this.")


if __name__ == "__main__":
    main()

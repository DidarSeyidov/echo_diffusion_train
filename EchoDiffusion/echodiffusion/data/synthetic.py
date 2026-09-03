"""Synthetic episode generator.

The reference bags currently carry audio only -- no odometry, no Vicon, no
camera -- so there is nothing to supervise a trajectory against yet.  This
module fabricates episodes in the exact on-disk format that
``bag_to_episode`` produces, which keeps the whole stack (dataset, model,
trainer, logging, evaluation) runnable and testable end to end until the real
recordings are complete.

The DoA statistics are matched to ``session_audio/session_01`` rather than
invented, because those quirks are what the encoder has to be robust to:

* four SSL potentials per hop, one near the true bearing and the rest
  reverberation-like clutter at lower energy;
* two SST tracks, one following the source and one pinned near vertical --
  the real bag has exactly such a track (``id=2887``, elevation 77-90 deg),
  which is why ``max_elevation_deg`` gating exists;
* high elevations overall (the reference speaker sat well above the array).

The "expert" that produces the supervision is a proportional unicycle
controller homing on the source, with per-episode gain/speed jitter and a
lateral detour so the trajectories are not all straight lines.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..utils.geometry import wrap_angle
from .episode import MAX_SSL, MAX_SST, Episode, EpisodeMeta


def _unit_from_az_el(az: np.ndarray, el: np.ndarray) -> np.ndarray:
    """(azimuth, elevation) in radians -> (N, 3) unit vectors, x fwd / y left / z up."""
    ce = np.cos(el)
    return np.stack([ce * np.cos(az), ce * np.sin(az), np.sin(el)], axis=-1)


def simulate_episode(
    seed: int = 0,
    duration_s: float = 20.0,
    sample_rate_hz: float = 10.0,
    source_range: tuple[float, float] = (3.0, 8.0),
    v_max: float = 0.45,
    omega_max: float = 1.2,
    stop_radius: float = 0.5,
    bearing_noise_deg: float = 4.0,
    clutter_prob: float = 0.8,
    dropout_prob: float = 0.05,
    elevation_deg: tuple[float, float] = (35.0, 75.0),
    name: str = "synthetic",
) -> Episode:
    """Simulate one approach-the-speaker episode.

    Args:
        source_range: (min, max) initial robot-source distance in metres.
        v_max / omega_max: expert speed limits (unicycle).
        stop_radius: expert stops once this close to the source.
        bearing_noise_deg: std of the azimuth error on the true detection.
        clutter_prob: per-step chance of emitting reverberation-like SSL peaks.
        dropout_prob: per-step chance the source is not detected at all
            (silence between utterances) -- the field must coast through these.
        elevation_deg: (min, max) elevation the source is seen at.
    """
    rng = np.random.default_rng(seed)
    n = int(round(duration_s * sample_rate_hz))
    dt = 1.0 / sample_rate_hz

    # ── scene ─────────────────────────────────────────────────────────────
    dist = rng.uniform(*source_range)
    src_bearing = rng.uniform(-np.pi, np.pi)
    source = np.array([dist * np.cos(src_bearing), dist * np.sin(src_bearing)])
    # Start heading is deliberately unrelated to the source bearing: many
    # episodes therefore begin with the source behind the robot, which is the
    # case that audio-only guidance has to solve and vision cannot.
    pose = np.array([0.0, 0.0, rng.uniform(-np.pi, np.pi)])

    k_omega = rng.uniform(1.2, 2.2)
    speed = v_max * rng.uniform(0.7, 1.0)
    # A slowly-varying lateral bias bends the path, so the expert is not a
    # pure straight line the model could fit from the first bearing alone.
    detour_amp = rng.uniform(0.0, 0.5)
    detour_freq = rng.uniform(0.1, 0.35)

    el_lo, el_hi = np.radians(elevation_deg)
    src_elev = rng.uniform(el_lo, el_hi)
    sst_track_id = int(rng.integers(1, 4000))

    ep = Episode.empty(n, EpisodeMeta(
        name=name, source="synthetic", sample_rate_hz=sample_rate_hz,
        extra={"seed": seed, "source_xy": source.tolist(),
               "sst_track_id": sst_track_id},
    ))
    ep.t = np.arange(n) * dt
    ep.source_pose = np.tile(source.astype(np.float32), (n, 1))

    # Vicon is the exact simulated pose; the odometry stream adds a slowly
    # accumulating drift so that switching ``poses.source`` in the config
    # exercises a genuinely different (and noisier) signal.
    drift = rng.normal(0.0, 0.002, size=3) * np.array([1.0, 1.0, 0.5])

    for i in range(n):
        ep.pose_vicon[i] = pose.astype(np.float32)
        ep.pose_odometry[i] = (pose + drift * i).astype(np.float32)

        delta = source - pose[:2]
        rng_to_src = float(np.hypot(*delta))
        true_bearing = wrap_angle(np.arctan2(delta[1], delta[0]) - pose[2])

        # ── measurements ──────────────────────────────────────────────────
        detected = rng.random() > dropout_prob
        ssl_rows, sst_rows = [], []

        if detected:
            az = true_bearing + rng.normal(0.0, np.radians(bearing_noise_deg))
            el = src_elev + rng.normal(0.0, np.radians(3.0))
            # Energy falls off with range but stays in ODAS's [0, 1] band.
            energy = float(np.clip(0.45 * (3.0 / max(rng_to_src, 1.0)) ** 0.5
                                   + rng.normal(0, 0.05), 0.05, 1.0))
            ssl_rows.append([*_unit_from_az_el(np.array(az), np.array(el)), energy])
            sst_rows.append([sst_track_id,
                             *_unit_from_az_el(np.array(az), np.array(el)),
                             float(np.clip(rng.uniform(0.6, 1.0), 0, 1))])
        else:
            # Dormant SST track: keeps its last bearing at ~zero activity,
            # exactly as ODAS behaves between utterances.
            sst_rows.append([sst_track_id,
                             *_unit_from_az_el(np.array(true_bearing),
                                               np.array(src_elev)), 0.0])

        if rng.random() < clutter_prob:
            for _ in range(int(rng.integers(1, 4))):
                az = rng.uniform(-np.pi, np.pi)
                el = rng.uniform(np.radians(20.0), np.radians(88.0))
                e = float(np.clip(rng.uniform(0.01, 0.14), 0.0, 1.0))
                ssl_rows.append([*_unit_from_az_el(np.array(az), np.array(el)), e])

        # The persistent near-vertical phantom track seen in the real bag.
        sst_rows.append([2887, *_unit_from_az_el(
            np.array(rng.uniform(-np.pi, np.pi)),
            np.array(np.radians(rng.uniform(80.0, 90.0)))),
            float(rng.uniform(0.1, 1.0))])

        k = min(len(ssl_rows), MAX_SSL)
        if k:
            ep.ssl[i, :k] = np.asarray(ssl_rows[:k], dtype=np.float32)
            ep.ssl_n[i] = k
        k = min(len(sst_rows), MAX_SST)
        if k:
            ep.sst[i, :k] = np.asarray(sst_rows[:k], dtype=np.float32)
            ep.sst_n[i] = k

        # ── expert step ───────────────────────────────────────────────────
        if rng_to_src > stop_radius:
            detour = detour_amp * np.sin(2 * np.pi * detour_freq * ep.t[i])
            omega = np.clip(k_omega * true_bearing + detour, -omega_max, omega_max)
            # Slow down while turning hard and on the final approach.
            v = speed * float(np.clip(np.cos(true_bearing), 0.0, 1.0))
            v *= float(np.clip(rng_to_src / 1.5, 0.15, 1.0))
        else:
            omega, v = 0.0, 0.0

        pose = np.array([
            pose[0] + v * np.cos(pose[2]) * dt,
            pose[1] + v * np.sin(pose[2]) * dt,
            wrap_angle(pose[2] + omega * dt),
        ])

    return ep


def generate_dataset(
    out_root: str | Path,
    n_train: int = 80,
    n_val: int = 16,
    seed: int = 0,
    verbose: bool = True,
    **episode_kwargs,
) -> dict[str, list[Path]]:
    """Write ``n_train`` + ``n_val`` synthetic episodes under ``out_root``.

    Train and val seeds come from disjoint ranges, so the split holds even if
    the counts change later.
    """
    out_root = Path(out_root)
    written: dict[str, list[Path]] = {"train": [], "val": []}

    for split, count, base in (("train", n_train, seed),
                               ("val", n_val, seed + 100_000)):
        for i in range(count):
            ep_dir = out_root / split / f"{split}_{i:04d}"
            ep = simulate_episode(seed=base + i, name=f"{split}_{i:04d}",
                                  **episode_kwargs)
            ep.save(ep_dir)
            written[split].append(ep_dir)
        if verbose:
            print(f"  {split}: {count} episodes -> {out_root / split}")

    return written

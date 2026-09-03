"""Torch dataset over prepared episodes.

Each sample is a window ending at some anchor step ``i``:

* **conditioning** -- the Bayesian BEV field rolled forward over the
  ``warmup`` steps up to ``i``, a set of raw DoA tokens for the same span, the
  robot's recent ego-motion, and (optionally) camera frames;
* **target** -- the next ``horizon`` poses expressed in the body frame at
  ``i``, normalised.

The BEV field is recomputed per window rather than cached for the whole
episode.  A window costs a few hundred microseconds of numpy, while caching
``(N, C, H, W)`` snapshots would run to gigabytes per split; recomputing also
mirrors exactly what a live ROS node does, so there is no train/deploy skew in
how the field is built.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ..audio.bev_field import BEVFieldConfig, SoundBEVField
from ..audio.odas import (ArrayExtrinsics, DoAObservation, merge_observations,
                          ssl_to_observation, sst_to_observation)
from ..utils.geometry import relative_pose, world_to_body, wrap_angle
from .episode import Episode, find_episodes

#: Per-detection token layout fed to the DoA encoder.
DOA_FEATURES = 6      # sin(az), cos(az), sin(el), cos(el), weight, is_tracked


@dataclass
class DataConfig:
    """Everything :class:`EchoTrajectoryDataset` needs, parsed from YAML."""

    # windowing
    horizon: int = 20              # predicted waypoints
    warmup: int = 30               # filter roll-forward before the anchor step
    past_len: int = 10             # past ego poses given to the model
    stride: int = 1                # anchor-step stride when enumerating windows

    # pose selection
    pose_source: str = "auto"      # "odometry" | "vicon" | "auto"
    pose_fallback: bool = True     # allow the other stream when the chosen one is absent

    # observations
    use_ssl: bool = True
    use_sst: bool = True
    max_doa_tokens: int = 8
    doa_frames: int = 4            # DoA token history depth
    min_sst_activity: float = 0.1  # dormant SST tracks carry a stale bearing

    # targets
    traj_scale: float = 2.0        # metres mapped to 1.0; "auto" fits it from data
    goal_scale: float = 8.0        # normalisation for the aux source-position head

    # images
    use_image: bool = False
    image_size: tuple[int, int] = (224, 392)
    image_frames: int = 1

    # nested configs
    array: dict = field(default_factory=dict)
    bev: dict = field(default_factory=dict)

    @classmethod
    def from_config(cls, config: dict) -> "DataConfig":
        dc = dict(config.get("data", {}))
        dc.setdefault("pose_source", config.get("poses", {}).get("source", "auto"))
        dc.setdefault("pose_fallback", config.get("poses", {}).get("fallback", True))
        dc.setdefault("array", config.get("array", {}))
        dc.setdefault("bev", config.get("bev", {}))
        if "image_size" in dc:
            dc["image_size"] = tuple(dc["image_size"])
        known = {k: v for k, v in dc.items() if k in cls.__dataclass_fields__}
        return cls(**known)


class EchoTrajectoryDataset(Dataset):
    """Windows of (BEV field + DoA tokens [+ image]) -> future trajectory."""

    def __init__(
        self,
        episode_dirs: list[str | Path],
        cfg: DataConfig,
        require_targets: bool = True,
        name: str = "dataset",
        traj_scale: float | None = None,
    ):
        self.cfg = cfg
        self.name = name
        self.require_targets = require_targets
        self.extrinsics = ArrayExtrinsics.from_config(cfg.array)
        self.bev_cfg = BEVFieldConfig.from_config(cfg.bev)

        self.episodes: list[Episode] = []
        for d in episode_dirs:
            ep = Episode.load(d)
            self.bev_cfg.dt_default = 1.0 / max(ep.meta.sample_rate_hz, 1e-6)
            self.episodes.append(ep)
        if not self.episodes:
            raise ValueError(f"[{name}] no episodes found")

        self.index = self._build_index()
        if not self.index:
            raise ValueError(
                f"[{name}] every episode was rejected -- no window has both a "
                f"valid pose stream (requested {cfg.pose_source!r}) and "
                f"{cfg.horizon} future steps. Prepared episodes without "
                f"odometry/Vicon can only be used with require_targets=False.")

        # An explicit scale wins: val must be normalised on the *train* scale,
        # otherwise its errors are measured on a different axis and are not
        # comparable across splits.
        if traj_scale is not None:
            self.traj_scale = float(traj_scale)
        elif cfg.traj_scale == "auto":
            self.traj_scale = self._fit_traj_scale()
            print(f"[{name}] traj_scale=auto -> {self.traj_scale:.3f} m")
        else:
            self.traj_scale = float(cfg.traj_scale)

    # ── indexing ──────────────────────────────────────────────────────────

    def _build_index(self) -> list[tuple[int, int]]:
        """Enumerate ``(episode, anchor_step)`` pairs that yield a full window.

        Anchors are restricted to the longest contiguous run of valid poses so
        no window straddles a pose gap; ``warmup`` steps before the anchor are
        allowed to fall outside it, since the filter simply starts later.
        """
        cfg = self.cfg
        index: list[tuple[int, int]] = []
        for e, ep in enumerate(self.episodes):
            if not self.require_targets:
                index += [(e, i) for i in range(0, len(ep), cfg.stride)]
                continue

            if ep.resolve_pose_source(cfg.pose_source, cfg.pose_fallback) is None:
                print(f"[{self.name}] skipping {ep.meta.name}: no "
                      f"{cfg.pose_source!r} pose stream")
                continue

            lo, hi = ep.valid_range(cfg.pose_source)
            # The anchor needs ``horizon`` future steps inside the valid run.
            last = hi - cfg.horizon
            if last <= lo:
                print(f"[{self.name}] skipping {ep.meta.name}: valid pose run "
                      f"[{lo}:{hi}] shorter than horizon {cfg.horizon}")
                continue
            index += [(e, i) for i in range(lo, last, cfg.stride)]
        return index

    def _fit_traj_scale(self, quantile: float = 0.99) -> float:
        """Pick the normalisation scale from the data itself.

        Uses a high quantile rather than the max so one anomalous window does
        not squash the whole target distribution toward zero.
        """
        mags = []
        for e, i in self.index[::max(len(self.index) // 2000, 1)]:
            traj = self._trajectory(self.episodes[e], i)
            if traj is not None:
                mags.append(np.abs(traj).max())
        if not mags:
            return 1.0
        return float(max(np.quantile(mags, quantile), 1e-3))

    def __len__(self) -> int:
        return len(self.index)

    # ── per-sample pieces ─────────────────────────────────────────────────

    def _observation(self, ep: Episode, i: int) -> DoAObservation:
        """Merge this step's SSL and SST detections into one bearing set."""
        parts = []
        if self.cfg.use_ssl:
            k = int(ep.ssl_n[i])
            if k:
                parts.append(ssl_to_observation(
                    float(ep.t[i]), ep.ssl[i, :k], self.extrinsics))
        if self.cfg.use_sst:
            k = int(ep.sst_n[i])
            if k:
                rows = ep.sst[i, :k]
                # Dormant tracks keep their last bearing at ~zero activity;
                # folding those in would keep re-asserting stale evidence.
                rows = rows[rows[:, 4] >= self.cfg.min_sst_activity]
                if len(rows):
                    parts.append(sst_to_observation(
                        float(ep.t[i]), rows, self.extrinsics))
        return merge_observations(parts, t=float(ep.t[i]))

    def _run_filter(self, ep: Episode, i: int) -> SoundBEVField:
        """Roll the Bayesian field forward over ``[i - warmup, i]``."""
        cfg = self.cfg
        field_ = SoundBEVField(self.bev_cfg)
        start = max(i - cfg.warmup + 1, 0)
        poses = ep.pose(cfg.pose_source, cfg.pose_fallback)
        dt = 1.0 / max(ep.meta.sample_rate_hz, 1e-6)

        prev_pose = None
        for k in range(start, i + 1):
            pose_k = poses[k]
            if np.isfinite(pose_k).all():
                delta = None if prev_pose is None else relative_pose(prev_pose, pose_k)
                prev_pose = pose_k
            else:
                # Without a pose we cannot warp; the field then behaves like a
                # stationary observer for that step, which is the honest
                # fallback (it just will not triangulate).
                delta = None
            field_.step(self._observation(ep, k), delta, dt=dt)
        return field_

    def _doa_tokens(self, ep: Episode, i: int) -> tuple[np.ndarray, np.ndarray]:
        """Recent raw detections as ``(doa_frames, max_doa_tokens, F)`` + mask.

        Frame 0 is the anchor step and later rows go back in time.  These carry
        the un-fused evidence: the BEV field is a lossy summary, and giving the
        policy the raw bearings too lets it react to a detection faster than
        the filter's half-life allows.
        """
        cfg = self.cfg
        tokens = np.zeros((cfg.doa_frames, cfg.max_doa_tokens, DOA_FEATURES),
                          dtype=np.float32)
        mask = np.zeros((cfg.doa_frames, cfg.max_doa_tokens), dtype=np.float32)

        for f in range(cfg.doa_frames):
            k = i - f
            if k < 0:
                break
            obs = self._observation(ep, k).filtered(
                self.bev_cfg.min_weight, self.bev_cfg.max_elevation_deg)
            obs = obs.top_k(cfg.max_doa_tokens)
            n = len(obs)
            if n == 0:
                continue
            tokens[f, :n, 0] = np.sin(obs.azimuth)
            tokens[f, :n, 1] = np.cos(obs.azimuth)
            tokens[f, :n, 2] = np.sin(obs.elevation)
            tokens[f, :n, 3] = np.cos(obs.elevation)
            tokens[f, :n, 4] = obs.weight
            tokens[f, :n, 5] = (obs.track_id >= 0).astype(np.float32)
            mask[f, :n] = 1.0
        return tokens, mask

    def _trajectory(self, ep: Episode, i: int) -> np.ndarray | None:
        """Future poses in the anchor body frame, as ``(horizon, 2)`` metres."""
        cfg = self.cfg
        poses = ep.pose(cfg.pose_source, cfg.pose_fallback)
        future = poses[i + 1: i + 1 + cfg.horizon]
        if len(future) < cfg.horizon or not np.isfinite(future).all():
            return None
        if not np.isfinite(poses[i]).all():
            return None
        return world_to_body(future[:, :2], poses[i]).astype(np.float32)

    def _past(self, ep: Episode, i: int) -> np.ndarray:
        """Past poses in the anchor body frame, ``(past_len, 3)``.

        Columns are (x, y, yaw-delta); rows run backwards from the anchor and
        are zero-padded (i.e. "stationary") at the start of an episode.
        """
        cfg = self.cfg
        out = np.zeros((cfg.past_len, 3), dtype=np.float32)
        poses = ep.pose(cfg.pose_source, cfg.pose_fallback)
        if not np.isfinite(poses[i]).all():
            return out
        for k in range(cfg.past_len):
            j = i - k - 1
            if j < 0 or not np.isfinite(poses[j]).all():
                break
            xy = world_to_body(poses[j, :2], poses[i])
            out[k] = [xy[0], xy[1], wrap_angle(poses[j, 2] - poses[i, 2])]
        return out

    def _images(self, ep: Episode, i: int) -> np.ndarray:
        """``(image_frames, 3, H, W)`` normalised RGB, newest first."""
        from .image_io import load_image, normalize_image

        cfg = self.cfg
        h, w = cfg.image_size
        out = np.zeros((cfg.image_frames, 3, h, w), dtype=np.float32)
        for f in range(cfg.image_frames):
            k = max(i - f, 0)
            path = ep.image_path(k)
            if path is None or not Path(path).exists():
                continue
            out[f] = normalize_image(load_image(path, size=(h, w)))
        return out

    # ── sample ────────────────────────────────────────────────────────────

    def __getitem__(self, idx: int) -> dict:
        e, i = self.index[idx]
        ep = self.episodes[e]
        cfg = self.cfg

        field_ = self._run_filter(ep, i)
        tokens, mask = self._doa_tokens(ep, i)

        fx, fy, spread = field_.expected_position()
        sample = {
            "bev": torch.from_numpy(field_.channels()),
            "doa": torch.from_numpy(tokens),
            "doa_mask": torch.from_numpy(mask),
            "past": torch.from_numpy(self._past(ep, i)),
            # Filter readout as an explicit low-dimensional cue: the CNN can in
            # principle extract this from the BEV stack, but handing it over
            # directly makes the audio-only model converge noticeably faster.
            "field_estimate": torch.tensor(
                [fx / cfg.goal_scale, fy / cfg.goal_scale,
                 spread / cfg.goal_scale, field_.confidence()],
                dtype=torch.float32),
            "episode": torch.tensor(e, dtype=torch.long),
            "step": torch.tensor(i, dtype=torch.long),
        }

        traj = self._trajectory(ep, i)
        if traj is None:
            sample["traj"] = torch.zeros(cfg.horizon, 2)
            sample["traj_valid"] = torch.tensor(0.0)
        else:
            sample["traj"] = torch.from_numpy(traj / self.traj_scale)
            sample["traj_valid"] = torch.tensor(1.0)

        # Auxiliary head: GT source position in the anchor body frame.
        poses = ep.pose(cfg.pose_source, cfg.pose_fallback)
        if ep.source_valid[i] and np.isfinite(poses[i]).all():
            rel = world_to_body(ep.source_pose[i], poses[i])
            sample["source_xy"] = torch.from_numpy(
                (rel / cfg.goal_scale).astype(np.float32))
            sample["source_valid"] = torch.tensor(1.0)
        else:
            sample["source_xy"] = torch.zeros(2)
            sample["source_valid"] = torch.tensor(0.0)

        if cfg.use_image:
            sample["image"] = torch.from_numpy(self._images(ep, i))

        return sample


def create_dataloaders(config: dict) -> tuple[DataLoader, DataLoader]:
    """Build train / val loaders from a parsed YAML config.

    Episodes are taken from ``paths.train_dir`` / ``paths.val_dir``.  When only
    ``paths.episode_dir`` is given, episodes are split by *episode* (never by
    window) so no trajectory leaks across the split.
    """
    cfg = DataConfig.from_config(config)
    paths = config.get("paths", {})
    tc = config.get("training", {})

    train_dirs, val_dirs = _resolve_splits(paths, config)
    train_ds = EchoTrajectoryDataset(train_dirs, cfg, name="train")
    val_ds = EchoTrajectoryDataset(val_dirs, cfg, name="val",
                                   traj_scale=train_ds.traj_scale)

    common = dict(
        batch_size=int(tc.get("batch_size", 32)),
        num_workers=int(tc.get("num_workers", 4)),
        pin_memory=bool(tc.get("pin_memory", True)),
        persistent_workers=bool(tc.get("num_workers", 4)) > 0,
        drop_last=False,
    )
    train_loader = DataLoader(train_ds, shuffle=True, **common)
    val_loader = DataLoader(val_ds, shuffle=False, **common)
    return train_loader, val_loader


def _resolve_splits(paths: dict, config: dict) -> tuple[list[Path], list[Path]]:
    if paths.get("train_dir") and paths.get("val_dir"):
        train = find_episodes(paths["train_dir"])
        val = find_episodes(paths["val_dir"])
        if not train:
            raise FileNotFoundError(
                f"no episodes under {paths['train_dir']} -- run "
                f"scripts/prepare_dataset.py first")
        return train, val

    root = paths.get("episode_dir")
    if not root:
        raise KeyError("config needs paths.train_dir + paths.val_dir, "
                       "or paths.episode_dir")
    episodes = find_episodes(root)
    if not episodes:
        raise FileNotFoundError(f"no episodes under {root}")

    frac = float(config.get("data", {}).get("val_fraction", 0.2))
    rng = np.random.default_rng(int(config.get("seed", 0)))
    order = rng.permutation(len(episodes))
    n_val = max(int(round(len(episodes) * frac)), 1) if len(episodes) > 1 else 0
    val_idx = set(order[:n_val].tolist())
    return ([episodes[i] for i in range(len(episodes)) if i not in val_idx],
            [episodes[i] for i in range(len(episodes)) if i in val_idx])

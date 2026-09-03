"""Convert a rosbag2 session into an :class:`~echodiffusion.data.episode.Episode`.

The bag is resampled onto a uniform grid at ``sample_rate_hz``.  ODAS runs far
faster than the policy needs (125 Hz in the reference recordings), so every
detection falling in a grid bin is pooled and the strongest ``MAX_SSL`` /
``MAX_SST`` are kept -- pooling rather than nearest-neighbour sampling, because
throwing away 12 of every 13 SSL frames discards real evidence.

Poses and the ground-truth source position come from whichever topics the
config points at; missing topics leave NaN, and the dataset then simply skips
the affected windows.  That is what makes the current audio-only bags usable
today and the same code correct once odometry, Vicon and camera land.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..utils.geometry import interpolate_poses, quat_to_yaw
from .episode import MAX_SSL, MAX_SST, Episode, EpisodeMeta
from .rosbag_reader import Rosbag2Reader

#: Defaults match the reference ODAS launch; override under ``dataset.topics``.
DEFAULT_TOPICS = {
    "ssl": "/ssl",
    "sst": "/sst",
    "odometry": "/odom",
    "vicon_robot": "/vicon/robot/pose",
    "vicon_source": "/vicon/source/pose",
    "image": "/camera/color/image_raw",
}


def _pose_rows(messages) -> tuple[np.ndarray, np.ndarray]:
    """Decoded pose-ish messages -> (timestamps, (N, 3) x/y/yaw)."""
    ts, rows = [], []
    for m in messages:
        msg = m.msg
        ts.append(msg.get("stamp") or m.t_recv)
        pos, quat = msg["position"], msg["orientation"]
        rows.append([pos[0], pos[1], quat_to_yaw(quat)])
    if not rows:
        return np.zeros(0), np.zeros((0, 3))
    return np.asarray(ts, dtype=np.float64), np.asarray(rows, dtype=np.float64)


def _bin_index(stamps: np.ndarray, t0: float, rate: float, n: int) -> np.ndarray:
    """Map timestamps to grid-bin indices, clipped to ``[0, n)``."""
    idx = np.floor((stamps - t0) * rate + 0.5).astype(np.int64)
    return np.clip(idx, 0, n - 1)


def _pool_odas(records: list[tuple[float, np.ndarray]], n_bins: int, t0: float,
               rate: float, max_rows: int, weight_col: int
               ) -> tuple[np.ndarray, np.ndarray]:
    """Pool ODAS rows into grid bins, keeping the strongest ``max_rows`` each.

    ``records`` is a list of ``(stamp, (K, C) rows)``; the result is
    ``((n_bins, max_rows, C), (n_bins,) counts)``.
    """
    n_cols = records[0][1].shape[1] if records else 4
    out = np.zeros((n_bins, max_rows, n_cols), dtype=np.float32)
    counts = np.zeros(n_bins, dtype=np.int16)
    if not records:
        return out, counts

    stamps = np.array([r[0] for r in records], dtype=np.float64)
    bins = _bin_index(stamps, t0, rate, n_bins)

    buckets: dict[int, list[np.ndarray]] = {}
    for b, (_, rows) in zip(bins, records):
        if rows.size:
            buckets.setdefault(int(b), []).append(rows)

    for b, chunks in buckets.items():
        rows = np.concatenate(chunks, axis=0)
        if rows.shape[0] > max_rows:
            keep = np.argsort(-rows[:, weight_col])[:max_rows]
            rows = rows[keep]
        k = rows.shape[0]
        out[b, :k] = rows
        counts[b] = k
    return out, counts


def build_episode_from_bag(
    session_dir: str | Path,
    out_dir: str | Path,
    sample_rate_hz: float = 10.0,
    topics: dict | None = None,
    pose_source: str = "odometry",
    static_source_pose: tuple[float, float] | None = None,
    save_images: bool = True,
    image_quality: int = 92,
    name: str | None = None,
    verbose: bool = True,
) -> Episode:
    """Read one bag session and write an episode directory.

    Args:
        session_dir: rosbag2 session (the directory holding ``metadata.yaml``).
        out_dir: destination episode directory.
        sample_rate_hz: uniform grid rate; also the policy's control rate.
        topics: overrides for :data:`DEFAULT_TOPICS`.
        pose_source: ``"odometry"`` or ``"vicon"`` -- which topic supplies the
            robot pose.  Falls back to the other with a warning if the chosen
            one is absent, and to NaN if neither exists.
        static_source_pose: world ``(x, y)`` of the speaker for recordings
            where it never moved and no Vicon topic was recorded.
        save_images: extract camera frames to ``<out_dir>/images``.
    """
    session_dir = Path(session_dir)
    out_dir = Path(out_dir)
    topics = {**DEFAULT_TOPICS, **(topics or {})}

    reader = Rosbag2Reader(session_dir)
    available = set(reader.topics)
    if verbose:
        print(reader.summary())

    def want(key: str) -> str | None:
        name_ = topics.get(key)
        return name_ if name_ in available else None

    ssl_topic, sst_topic = want("ssl"), want("sst")
    if ssl_topic is None and sst_topic is None:
        raise ValueError(
            f"{session_dir} has neither {topics['ssl']} nor {topics['sst']}; "
            "nothing to build an episode from")

    # ── collect ───────────────────────────────────────────────────────────
    ssl_records: list[tuple[float, np.ndarray]] = []
    sst_records: list[tuple[float, np.ndarray]] = []
    image_msgs: list = []
    pose_msgs: dict[str, list] = {"odometry": [], "vicon_robot": [], "vicon_source": []}

    wanted = [t for t in (ssl_topic, sst_topic, want("odometry"),
                          want("vicon_robot"), want("vicon_source"),
                          want("image") if save_images else None) if t]
    key_for_topic = {topics[k]: k for k in
                     ("odometry", "vicon_robot", "vicon_source")}

    for m in reader.read(topics=wanted):
        stamp = m.msg.get("stamp") or m.t_recv
        if m.topic == ssl_topic:
            ssl_records.append((stamp, m.msg["sources"]))
        elif m.topic == sst_topic:
            sst_records.append((stamp, m.msg["sources"]))
        elif m.topic in key_for_topic:
            pose_msgs[key_for_topic[m.topic]].append(m)
        else:
            image_msgs.append(m)

    # ── time grid ─────────────────────────────────────────────────────────
    all_stamps = [s for s, _ in ssl_records] + [s for s, _ in sst_records]
    t0, t1 = min(all_stamps), max(all_stamps)
    n = max(int(np.floor((t1 - t0) * sample_rate_hz)) + 1, 1)
    grid = t0 + np.arange(n) / sample_rate_hz

    ep = Episode.empty(n, EpisodeMeta(
        name=name or session_dir.name,
        source="rosbag",
        sample_rate_hz=float(sample_rate_hz),
        extra={"bag": str(session_dir), "topics": topics},
    ))
    ep.t = grid

    ep.ssl, ep.ssl_n = _pool_odas(ssl_records, n, t0, sample_rate_hz,
                                  MAX_SSL, weight_col=3)   # energy
    ep.sst, ep.sst_n = _pool_odas(sst_records, n, t0, sample_rate_hz,
                                  MAX_SST, weight_col=4)   # activity

    # ── robot pose ────────────────────────────────────────────────────────
    # Both streams are stored when both were recorded; ``poses.source`` in the
    # training config picks between them later.  ``pose_source`` here only
    # controls which absence is worth warning about.
    for key, attr in (("odometry", "pose_odometry"), ("vicon_robot", "pose_vicon")):
        if not pose_msgs[key]:
            continue
        ts, rows = _pose_rows(pose_msgs[key])
        poses = interpolate_poses(grid, ts, rows).astype(np.float32)
        # Do not extrapolate: np.interp clamps outside the recorded span, which
        # would silently invent a stationary robot at both ends.
        poses[(grid < ts.min()) | (grid > ts.max())] = np.nan
        setattr(ep, attr, poses)

    got = [s for s in ("odometry", "vicon") if ep.has_pose(s)]
    if verbose:
        if not got:
            print(f"  ! no pose topic ({topics['odometry']} / "
                  f"{topics['vicon_robot']}) -- poses left as NaN")
        elif pose_source not in got and pose_source != "auto":
            print(f"  ! requested pose_source={pose_source!r} not in this bag; "
                  f"available: {got}")

    # ── ground-truth source pose ──────────────────────────────────────────
    if pose_msgs["vicon_source"]:
        ts, rows = _pose_rows(pose_msgs["vicon_source"])
        src = interpolate_poses(grid, ts, rows).astype(np.float32)
        src[(grid < ts.min()) | (grid > ts.max())] = np.nan
        ep.source_pose = src[:, :2]
    elif static_source_pose is not None:
        ep.source_pose = np.tile(
            np.asarray(static_source_pose, dtype=np.float32), (n, 1))
    elif verbose:
        print("  ! no GT source pose (no vicon_source topic, no "
              "static_source_pose) -- source_pose left as NaN")

    # ── images ────────────────────────────────────────────────────────────
    if save_images and image_msgs:
        ep.image_paths, ep.image_index = _write_images(
            image_msgs, out_dir, grid, sample_rate_hz)

    ep.save(out_dir)
    if verbose:
        print(f"  -> {out_dir}\n  {ep.describe()}")
    return ep


def _write_images(image_msgs, out_dir: Path, grid: np.ndarray, rate: float
                  ) -> tuple[list[str], np.ndarray]:
    """Extract one camera frame per grid bin (the nearest in time)."""
    from .image_io import save_image_message

    img_dir = Path(out_dir) / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    stamps = np.array([m.msg.get("stamp") or m.t_recv for m in image_msgs])
    paths: list[str] = []
    index = np.full(len(grid), -1, dtype=np.int32)

    # Nearest camera frame per bin; bins with nothing within half a period
    # keep -1 so the dataset can drop them instead of showing a stale frame.
    tol = 0.5 / rate
    for i, t in enumerate(grid):
        j = int(np.argmin(np.abs(stamps - t)))
        if abs(stamps[j] - t) > tol:
            continue
        rel = f"images/{i:06d}.jpg"
        save_image_message(image_msgs[j].msg, Path(out_dir) / rel)
        index[i] = len(paths)
        paths.append(rel)
    return paths, index

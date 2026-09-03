"""On-disk episode format.

One episode = one continuous recording resampled onto a uniform time grid.
Everything downstream (the torch ``Dataset``, evaluation, visualisation) reads
this format, so bags, the synthetic generator and any future data source only
have to agree on *this* schema rather than on each other.

Layout on disk::

    <episode_dir>/
        episode.npz      # the arrays below
        episode.json     # human-readable metadata (mirrors npz['meta'])
        images/          # only when the recording carried a camera
            000000.jpg ...

Arrays (N = number of samples on the time grid):

=================  =====================  ============================================
key                shape / dtype          meaning
=================  =====================  ============================================
``t``              (N,) float64           seconds since epoch, uniform spacing
``ssl``            (N, K, 4) float32      ODAS SSL, **raw array frame**: x, y, z, energy
``ssl_n``          (N,) int16             valid rows in ``ssl`` (rest are zero padding)
``sst``            (N, M, 5) float32      ODAS SST, raw: id, x, y, z, activity
``sst_n``          (N,) int16             valid rows in ``sst``
``pose_odometry``  (N, 3) float32         robot (x, y, yaw) from odometry; NaN if absent
``pose_vicon``     (N, 3) float32         robot (x, y, yaw) from Vicon;    NaN if absent
``source_pose``    (N, 2) float32         GT source (x, y) in world; NaN if unknown
``image_index``    (N,) int32             row into ``image_paths``; -1 if none
``image_paths``    (P,) <U                paths relative to the episode dir
=================  =====================  ============================================

Two design choices worth stating:

* **Both pose streams are stored**, each possibly all-NaN.  ``poses.source`` in
  the config then selects between them at *train* time, so switching odometry
  vs. Vicon is a config edit rather than a dataset rebuild.
* **DoA vectors are stored unrotated**, exactly as ODAS emitted them, with the
  array extrinsics applied at load time -- so recalibrating the array mounting
  angle is likewise a config edit, which matters while that angle is still
  being pinned down.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np

EPISODE_NPZ = "episode.npz"
EPISODE_JSON = "episode.json"
IMAGE_DIR = "images"

#: Padding widths.  ODAS is configured for 4 SSL potentials and 2-4 SST tracks
#: in the reference recordings; the loader tolerates fewer.
MAX_SSL = 8
MAX_SST = 8


@dataclass
class EpisodeMeta:
    name: str = "episode"
    source: str = "unknown"          # "rosbag" | "synthetic"
    sample_rate_hz: float = 10.0
    duration_s: float = 0.0
    n_samples: int = 0
    #: Which pose streams actually carry data, e.g. ``["odometry", "vicon"]``.
    pose_sources: list = field(default_factory=list)
    has_source_pose: bool = False
    has_images: bool = False
    #: Free-form provenance (bag path, topic names, generator seed, ...).
    extra: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=str)

    @classmethod
    def from_dict(cls, d: dict) -> "EpisodeMeta":
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)


#: Pose streams an episode may carry.  ``"auto"`` in a config means "prefer
#: Vicon when present, else odometry".
POSE_SOURCES = ("odometry", "vicon")


@dataclass
class Episode:
    t: np.ndarray
    ssl: np.ndarray
    ssl_n: np.ndarray
    sst: np.ndarray
    sst_n: np.ndarray
    pose_odometry: np.ndarray
    pose_vicon: np.ndarray
    source_pose: np.ndarray
    image_index: np.ndarray
    image_paths: list[str]
    meta: EpisodeMeta
    #: Set when loaded from disk; used to resolve ``image_paths``.
    root: Path | None = None

    # ── construction ──────────────────────────────────────────────────────

    @classmethod
    def empty(cls, n: int, meta: EpisodeMeta | None = None) -> "Episode":
        return cls(
            t=np.zeros(n, dtype=np.float64),
            ssl=np.zeros((n, MAX_SSL, 4), dtype=np.float32),
            ssl_n=np.zeros(n, dtype=np.int16),
            sst=np.zeros((n, MAX_SST, 5), dtype=np.float32),
            sst_n=np.zeros(n, dtype=np.int16),
            pose_odometry=np.full((n, 3), np.nan, dtype=np.float32),
            pose_vicon=np.full((n, 3), np.nan, dtype=np.float32),
            source_pose=np.full((n, 2), np.nan, dtype=np.float32),
            image_index=np.full(n, -1, dtype=np.int32),
            image_paths=[],
            meta=meta or EpisodeMeta(),
        )

    def __len__(self) -> int:
        return int(self.t.shape[0])

    # ── pose access ───────────────────────────────────────────────────────

    def has_pose(self, source: str) -> bool:
        arr = getattr(self, f"pose_{source}", None)
        return arr is not None and bool(np.isfinite(arr).all(axis=1).any())

    def resolve_pose_source(self, requested: str = "auto",
                            fallback: bool = True) -> str | None:
        """Pick a usable pose stream, honouring ``requested`` where possible.

        ``"auto"`` prefers Vicon (millimetre-accurate, no drift) and falls back
        to odometry.  An explicit request that is unavailable falls back to the
        other stream only when ``fallback`` is set, so a strict config can hard
        fail instead of silently training on the wrong source.
        """
        if requested == "auto":
            order = ["vicon", "odometry"]
        elif fallback:
            order = [requested] + [s for s in POSE_SOURCES if s != requested]
        else:
            order = [requested]
        return next((s for s in order if self.has_pose(s)), None)

    def pose(self, source: str = "auto", fallback: bool = True) -> np.ndarray:
        """(N, 3) robot poses from the selected stream; all-NaN if none exist."""
        chosen = self.resolve_pose_source(source, fallback)
        if chosen is None:
            return np.full((len(self), 3), np.nan, dtype=np.float32)
        return getattr(self, f"pose_{chosen}")

    # ── validity ──────────────────────────────────────────────────────────

    def pose_valid(self, source: str = "auto") -> np.ndarray:
        """(N,) bool -- samples with a usable robot pose."""
        return np.isfinite(self.pose(source)).all(axis=1)

    @property
    def source_valid(self) -> np.ndarray:
        """(N,) bool -- samples with a usable GT source position."""
        return np.isfinite(self.source_pose).all(axis=1)

    def valid_range(self, source: str = "auto") -> tuple[int, int]:
        """Largest contiguous span with a valid pose, as ``[start, stop)``.

        Trajectory targets are differences of poses, so a single NaN gap
        poisons every window that straddles it; windows are therefore drawn
        only from inside this span.
        """
        return longest_true_run(self.pose_valid(source))

    def image_path(self, i: int) -> Path | None:
        idx = int(self.image_index[i])
        if idx < 0 or idx >= len(self.image_paths):
            return None
        root = self.root or Path(".")
        return root / self.image_paths[idx]

    # ── io ────────────────────────────────────────────────────────────────

    def save(self, out_dir: str | Path) -> Path:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        self.meta.n_samples = len(self)
        self.meta.duration_s = float(self.t[-1] - self.t[0]) if len(self) else 0.0
        self.meta.pose_sources = [s for s in POSE_SOURCES if self.has_pose(s)]
        self.meta.has_source_pose = bool(self.source_valid.any())
        self.meta.has_images = bool((self.image_index >= 0).any())

        np.savez_compressed(
            out_dir / EPISODE_NPZ,
            t=self.t, ssl=self.ssl, ssl_n=self.ssl_n,
            sst=self.sst, sst_n=self.sst_n,
            pose_odometry=self.pose_odometry, pose_vicon=self.pose_vicon,
            source_pose=self.source_pose,
            image_index=self.image_index,
            image_paths=np.array(self.image_paths, dtype=object),
            meta=np.array(self.meta.to_json()),
        )
        (out_dir / EPISODE_JSON).write_text(self.meta.to_json())
        return out_dir / EPISODE_NPZ

    @classmethod
    def load(cls, episode_dir: str | Path) -> "Episode":
        episode_dir = Path(episode_dir)
        npz_path = episode_dir if episode_dir.suffix == ".npz" \
            else episode_dir / EPISODE_NPZ
        root = npz_path.parent

        with np.load(npz_path, allow_pickle=True) as z:
            meta = EpisodeMeta.from_dict(json.loads(str(z["meta"])))
            paths = [str(p) for p in z["image_paths"].tolist()]
            n = len(z["t"])
            nan3 = np.full((n, 3), np.nan, dtype=np.float32)
            return cls(
                t=z["t"], ssl=z["ssl"], ssl_n=z["ssl_n"],
                sst=z["sst"], sst_n=z["sst_n"],
                pose_odometry=z["pose_odometry"] if "pose_odometry" in z else nan3,
                pose_vicon=z["pose_vicon"] if "pose_vicon" in z else nan3.copy(),
                source_pose=z["source_pose"],
                image_index=z["image_index"],
                image_paths=paths, meta=meta, root=root,
            )

    def describe(self, source: str = "auto") -> str:
        # Computed from the arrays rather than read from ``meta``, so this is
        # accurate on a freshly built episode too (meta is refreshed on save).
        chosen = self.resolve_pose_source(source) or "none"
        available = [s for s in POSE_SOURCES if self.has_pose(s)] or ["none"]
        valid = self.pose_valid(source)
        lo, hi = self.valid_range(source)
        duration = float(self.t[-1] - self.t[0]) if len(self) else 0.0
        return (
            f"{self.meta.name}: {len(self)} samples @ {self.meta.sample_rate_hz} Hz "
            f"({duration:.1f} s)\n"
            f"  pose={chosen} (available: {', '.join(available)}) "
            f"valid={int(valid.sum())}/{len(self)} contiguous=[{lo}:{hi}]\n"
            f"  source_pose={'yes' if self.source_valid.any() else 'no'}  "
            f"images={'yes' if (self.image_index >= 0).any() else 'no'}  "
            f"ssl={self.ssl_n.mean():.1f}/step  sst={self.sst_n.mean():.1f}/step"
        )


def longest_true_run(mask: np.ndarray) -> tuple[int, int]:
    """Longest contiguous run of True in a 1-D boolean mask, as ``[start, stop)``."""
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return 0, 0
    # Pad with False so every run has an explicit rising and falling edge.
    padded = np.concatenate(([False], mask, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    starts, stops = edges[0::2], edges[1::2]
    k = int(np.argmax(stops - starts))
    return int(starts[k]), int(stops[k])


def find_episodes(root: str | Path) -> list[Path]:
    """Every episode directory under ``root`` (recursively), sorted by name."""
    root = Path(root)
    if not root.exists():
        return []
    return sorted({p.parent for p in root.rglob(EPISODE_NPZ)})

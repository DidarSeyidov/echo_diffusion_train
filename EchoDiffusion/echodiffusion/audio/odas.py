"""Turn ODAS SSL/SST direction-of-arrival vectors into robot-frame bearings.

ODAS reports each source as a **unit vector in the microphone-array frame**.
Two steps get us to something a planar policy can use:

1. rotate the vector into the robot base frame (``ArrayExtrinsics``), and
2. project onto the ground plane and take ``azimuth = atan2(y, x)``.

Step 2 discards elevation, which is the right call for a ground robot: the
source's bearing is what determines where to drive.  Elevation is still carried
through as metadata so it can gate detections (see ``max_elevation_deg``) --
in the reference session the tracked source sits at 50-80 deg elevation, so an
over-tight gate would throw away every real detection.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..utils.geometry import wrap_angle


def euler_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Intrinsic Z-Y-X (yaw-pitch-roll) rotation matrix, radians."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp,     cp * sr,                cp * cr],
    ], dtype=np.float64)


@dataclass
class ArrayExtrinsics:
    """Pose of the microphone array in the robot base frame.

    ``rotation_rpy_deg`` rotates a vector **from the array frame into the base
    frame**.  The default (0, 0, 0) assumes the array lies flat with its z axis
    up and its x axis pointing forward along the robot -- the configuration
    confirmed for the reference recordings.

    ``azimuth_offset_deg`` is a final yaw correction applied after the
    rotation.  Keep it for the common case where the array is flat but clocked
    by some angle about its own z axis; it is what
    ``scripts/calibrate_array.py`` solves for.
    """

    rotation_rpy_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)
    translation: tuple[float, float, float] = (0.0, 0.0, 0.0)
    azimuth_offset_deg: float = 0.0
    #: Mirror the azimuth.  Some ODAS mic-geometry configs end up left-handed
    #: relative to the robot; flipping here is cheaper than re-deriving the
    #: array geometry.
    flip_azimuth: bool = False

    @classmethod
    def from_config(cls, cfg: dict | None) -> "ArrayExtrinsics":
        cfg = cfg or {}
        return cls(
            rotation_rpy_deg=tuple(cfg.get("rotation_rpy_deg", (0.0, 0.0, 0.0))),
            translation=tuple(cfg.get("translation", (0.0, 0.0, 0.0))),
            azimuth_offset_deg=float(cfg.get("azimuth_offset_deg", 0.0)),
            flip_azimuth=bool(cfg.get("flip_azimuth", False)),
        )

    @property
    def rotation(self) -> np.ndarray:
        return euler_to_matrix(*np.radians(self.rotation_rpy_deg))

    def to_base(self, vecs: np.ndarray) -> np.ndarray:
        """Rotate (N, 3) array-frame unit vectors into the base frame."""
        return np.asarray(vecs, dtype=np.float64) @ self.rotation.T


@dataclass
class DoAObservation:
    """One synchronised batch of bearing observations at time ``t``.

    Attributes:
        t: timestamp (seconds).
        azimuth: (K,) bearing in the robot base frame, radians, 0 = forward,
            positive counter-clockwise (to the robot's left).
        elevation: (K,) angle above the ground plane, radians.
        weight: (K,) confidence in [0, 1] -- SSL energy or SST activity.
        track_id: (K,) SST track id, or -1 for untracked SSL potentials.
    """

    t: float
    azimuth: np.ndarray
    elevation: np.ndarray
    weight: np.ndarray
    track_id: np.ndarray

    def __len__(self) -> int:
        return int(self.azimuth.shape[0])

    def filtered(self, min_weight: float = 0.0,
                 max_elevation_deg: float = 90.0) -> "DoAObservation":
        keep = (self.weight >= min_weight) & (
            np.abs(np.degrees(self.elevation)) <= max_elevation_deg)
        return DoAObservation(self.t, self.azimuth[keep], self.elevation[keep],
                              self.weight[keep], self.track_id[keep])

    def top_k(self, k: int) -> "DoAObservation":
        """Keep the ``k`` highest-weight detections (padding is the caller's job)."""
        if len(self) <= k:
            return self
        idx = np.argsort(-self.weight)[:k]
        return DoAObservation(self.t, self.azimuth[idx], self.elevation[idx],
                              self.weight[idx], self.track_id[idx])


def vectors_to_bearings(vecs: np.ndarray, extrinsics: ArrayExtrinsics
                        ) -> tuple[np.ndarray, np.ndarray]:
    """(N, 3) array-frame unit vectors -> (azimuth, elevation) in the base frame.

    Degenerate vectors (pointing straight up/down, where the ground-plane
    projection vanishes) yield azimuth 0 with elevation +/-pi/2; gate them out
    via ``DoAObservation.filtered`` rather than trusting the azimuth.
    """
    vecs = np.atleast_2d(np.asarray(vecs, dtype=np.float64))
    if vecs.size == 0:
        return np.zeros(0), np.zeros(0)

    base = extrinsics.to_base(vecs)
    norm = np.linalg.norm(base, axis=1, keepdims=True)
    base = base / np.clip(norm, 1e-9, None)

    azimuth = np.arctan2(base[:, 1], base[:, 0])
    if extrinsics.flip_azimuth:
        azimuth = -azimuth
    azimuth = wrap_angle(azimuth + np.radians(extrinsics.azimuth_offset_deg))
    elevation = np.arcsin(np.clip(base[:, 2], -1.0, 1.0))
    return azimuth, elevation


def ssl_to_observation(stamp: float, sources: np.ndarray,
                       extrinsics: ArrayExtrinsics) -> DoAObservation:
    """``OdasSslArrayStamped.sources`` (K, 4) = (x, y, z, E) -> observation."""
    if sources.size == 0:
        z = np.zeros(0)
        return DoAObservation(stamp, z, z, z, np.zeros(0, dtype=np.int64))
    az, el = vectors_to_bearings(sources[:, :3], extrinsics)
    weight = np.clip(sources[:, 3].astype(np.float64), 0.0, 1.0)
    return DoAObservation(stamp, az, el, weight,
                          np.full(len(az), -1, dtype=np.int64))


def sst_to_observation(stamp: float, sources: np.ndarray,
                       extrinsics: ArrayExtrinsics) -> DoAObservation:
    """``OdasSstArrayStamped.sources`` (M, 5) = (id, x, y, z, activity)."""
    if sources.size == 0:
        z = np.zeros(0)
        return DoAObservation(stamp, z, z, z, np.zeros(0, dtype=np.int64))
    az, el = vectors_to_bearings(sources[:, 1:4], extrinsics)
    weight = np.clip(sources[:, 4].astype(np.float64), 0.0, 1.0)
    return DoAObservation(stamp, az, el, weight,
                          sources[:, 0].astype(np.int64))


def merge_observations(obs: list[DoAObservation], t: float | None = None
                       ) -> DoAObservation:
    """Concatenate several observations into one (e.g. SSL + SST at one step)."""
    obs = [o for o in obs if len(o) > 0]
    if not obs:
        z = np.zeros(0)
        return DoAObservation(t or 0.0, z, z, z, np.zeros(0, dtype=np.int64))
    return DoAObservation(
        t if t is not None else obs[0].t,
        np.concatenate([o.azimuth for o in obs]),
        np.concatenate([o.elevation for o in obs]),
        np.concatenate([o.weight for o in obs]),
        np.concatenate([o.track_id for o in obs]),
    )


def dominant_bearing_from_tokens(tokens: np.ndarray, mask: np.ndarray
                                 ) -> tuple[np.ndarray, np.ndarray]:
    """Weighted circular mean bearing of a batch of DoA token frames.

    This recovers the *measured* sound direction from the tensors the model
    actually consumed -- useful for visualising what the policy heard, as
    opposed to the ground-truth source position it was never shown.

    Args:
        tokens: ``(B, K, F)`` with columns (sin az, cos az, sin el, cos el,
            weight, is_tracked) -- i.e. one frame out of the dataset's
            ``(B, T, K, F)`` DoA stack.
        mask: ``(B, K)`` validity, 1 for a real detection.

    Returns:
        ``(bearing, strength)``, each ``(B,)``.  ``strength`` is the resultant
        length in [0, 1]: near 1 the detections agree, near 0 they cancel and
        the bearing is meaningless.
    """
    tokens = np.asarray(tokens, dtype=np.float64)
    mask = np.asarray(mask, dtype=np.float64)
    w = tokens[..., 4] * mask                       # (B, K)
    s = np.sum(w * tokens[..., 0], axis=-1)         # sin components
    c = np.sum(w * tokens[..., 1], axis=-1)         # cos components
    total = np.clip(w.sum(axis=-1), 1e-9, None)
    return np.arctan2(s, c), np.clip(np.hypot(c, s) / total, 0.0, 1.0)


def dominant_bearing(obs: DoAObservation) -> tuple[float, float]:
    """Circular-mean bearing of an observation, plus its resultant length.

    The resultant length in [0, 1] doubles as an agreement score: near 1 the
    detections all point the same way, near 0 they cancel out.
    """
    if len(obs) == 0:
        return 0.0, 0.0
    w = obs.weight
    if w.sum() <= 0:
        w = np.ones_like(w)
    c = np.sum(w * np.cos(obs.azimuth))
    s = np.sum(w * np.sin(obs.azimuth))
    r = np.hypot(c, s) / max(w.sum(), 1e-9)
    return float(np.arctan2(s, c)), float(np.clip(r, 0.0, 1.0))

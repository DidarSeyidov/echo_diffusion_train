"""Recursive Bayesian BEV field over "where is the sound source?".

A single direction-of-arrival measurement carries **no range information**: it
constrains the source to a ray, not a point.  The field below exploits that
directly.  Every detection deposits a fan-shaped log-odds ridge along its
bearing; because the ridge is anchored to the *robot's pose at that instant*,
two detections taken from different positions intersect, and the posterior
collapses from a ridge onto a blob.  Driving is what makes the estimate sharp
-- which is exactly the behaviour we want the policy to learn to exploit.

Representation
--------------
* Robot-centric grid, ``x`` forward and ``y`` left, covering
  ``[-range_m, +range_m]`` on both axes (the source can be behind the robot).
* Row 0 is the far-forward edge and column 0 the far-left edge, so
  ``imshow(field.probability)`` renders a conventional top-down view.
* State is log-odds, so the update is an addition and the prior (p = 0.5) is 0.

Recursion
---------
``predict(delta_pose)`` rigidly warps the accumulated log-odds into the new
body frame (odometry motion model, unknown territory enters at log-odds 0);
``update(obs)`` adds the inverse sensor model of the new detections.  A
``decay`` factor per step pulls the field back toward the prior so a source
that moves or stops does not stay burned in forever.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import affine_transform

from ..utils.geometry import rot2d, wrap_angle
from .odas import DoAObservation


@dataclass
class BEVFieldConfig:
    range_m: float = 6.0          # half-extent of the grid (m)
    resolution: float = 0.15      # metres per cell
    kappa_min: float = 4.0        # von Mises concentration at zero confidence
    kappa_max: float = 40.0       # ... and at full confidence
    p_hit: float = 0.60           # probability assigned on the ray axis
    p_miss: float = 0.48          # ... and far off-axis
    l_max: float = 15.0           # log-odds clamp (numerical guard only)
    #: Evidence half-life in *seconds*.  This is the single knob that trades
    #: triangulation baseline against responsiveness: rays older than a few
    #: half-lives no longer contribute, so the filter only triangulates over
    #: the motion executed within that window.  Too short and every ray comes
    #: from nearly the same place (no parallax, posterior stays a ridge); too
    #: long and a source that moves leaves a stale ghost.
    decay_half_life_s: float = 6.0
    dt_default: float = 0.1       # assumed step when ``step`` is called without dt
    min_range: float = 0.25       # cells nearer than this are not updated (m)
    min_weight: float = 0.05      # ignore detections below this confidence
    #: Drop DoAs above this elevation.  Two reasons, both load-bearing:
    #: azimuth is numerically degenerate for near-vertical vectors, and ODAS
    #: tends to park phantom/reflection tracks near the zenith.  Measured on
    #: session_audio/session_01, where a phantom track sits at 78-90 deg and
    #: the real source at 47-78 deg: at 89 deg the median bearing error is
    #: 1.0 deg but the *worst* case is 107 deg, because the phantom
    #: intermittently takes over; anywhere in 75-80 deg the worst case is
    #: 1.8 deg.  Below ~70 deg the real source starts being rejected too and
    #: the failures come back.
    max_elevation_deg: float = 78.0
    history_len: int = 4          # instantaneous ray maps exposed to the network
    #: Softmax temperature (in log-odds units) for the position readout.
    readout_temperature: float = 0.5

    @classmethod
    def from_config(cls, cfg: dict | None) -> "BEVFieldConfig":
        cfg = cfg or {}
        known = {f: cfg[f] for f in cls.__dataclass_fields__ if f in cfg}
        return cls(**known)

    @property
    def grid_size(self) -> int:
        return int(round(2.0 * self.range_m / self.resolution))

    def decay_for(self, dt: float) -> float:
        """Per-step multiplicative decay for an elapsed time of ``dt`` seconds."""
        if self.decay_half_life_s <= 0:
            return 0.0
        return float(0.5 ** (max(dt, 0.0) / self.decay_half_life_s))


class SoundBEVField:
    """Recursive Bayesian occupancy filter over sound-source position."""

    def __init__(self, config: BEVFieldConfig | dict | None = None):
        if not isinstance(config, BEVFieldConfig):
            config = BEVFieldConfig.from_config(config)
        self.cfg = config

        n = self.cfg.grid_size
        self.shape = (n, n)
        res = self.cfg.resolution
        # Cell-centre coordinates in the robot body frame.  ``c`` is the
        # coordinate of index 0 and is reused by the motion warp below.
        self.c = self.cfg.range_m - 0.5 * res
        idx = np.arange(n, dtype=np.float64)
        self.xs = self.c - idx * res                    # row -> x (forward)
        self.ys = self.c - idx * res                    # col -> y (left)
        gx, gy = np.meshgrid(self.xs, self.ys, indexing="ij")
        self.cell_x, self.cell_y = gx, gy
        self.cell_bearing = np.arctan2(gy, gx)          # (H, W) radians
        self.cell_range = np.hypot(gx, gy)              # (H, W) metres
        self._near_mask = self.cell_range < self.cfg.min_range
        # cos/sin of the cell bearings, so the measurement model can use the
        # angle-difference identity instead of a full-grid cos per detection.
        self._cos_bearing = np.cos(self.cell_bearing)
        self._sin_bearing = np.sin(self.cell_bearing)

        self.reset()

    # ── lifecycle ─────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Clear the posterior back to a uniform prior."""
        self.logodds = np.zeros(self.shape, dtype=np.float32)
        self.last_increment = np.zeros(self.shape, dtype=np.float32)
        #: dead-reckoned pose of the robot in the frame the filter started in;
        #: only ever used for *relative* geometry, so its origin is arbitrary.
        self.pose = np.zeros(3, dtype=np.float64)
        self.history: deque = deque(maxlen=max(self.cfg.history_len, 1))
        self.n_updates = 0

    # ── motion model ──────────────────────────────────────────────────────

    def predict(self, delta_pose: np.ndarray | None) -> None:
        """Warp the posterior into the new body frame.

        ``delta_pose = (dx, dy, dyaw)`` is the new pose expressed in the
        *previous* body frame, i.e. the output of
        :func:`~echodiffusion.utils.geometry.relative_pose`.
        """
        if delta_pose is None:
            return
        dx, dy, dyaw = (float(v) for v in delta_pose)

        # Track the dead-reckoned pose first: history entries are stored in
        # this frame and must stay consistent with the warp.
        c_yaw, s_yaw = np.cos(self.pose[2]), np.sin(self.pose[2])
        self.pose = np.array([
            self.pose[0] + c_yaw * dx - s_yaw * dy,
            self.pose[1] + s_yaw * dx + c_yaw * dy,
            wrap_angle(self.pose[2] + dyaw),
        ])

        if abs(dx) < 1e-9 and abs(dy) < 1e-9 and abs(dyaw) < 1e-9:
            return

        # A point at body-new coords p maps to body-old coords R(dyaw) p + t.
        # In index space (i = (c - x)/res) the same relation holds with the
        # identical rotation, because the axis flip is a 180 deg rotation and
        # commutes with R.
        R = rot2d(dyaw)
        res = self.cfg.resolution
        c_vec = np.array([self.c, self.c])
        offset = (c_vec - R @ c_vec - np.array([dx, dy])) / res

        self.logodds = affine_transform(
            self.logodds, matrix=R, offset=offset,
            order=1, mode="constant", cval=0.0, output=np.float32,
        )

    # ── measurement model ─────────────────────────────────────────────────

    def ray_logodds(self, obs: DoAObservation,
                    observer_rel: np.ndarray | None = None) -> np.ndarray:
        """Inverse sensor model for one observation, as a log-odds increment.

        Args:
            obs: detections, with bearings in the observer's body frame.
            observer_rel: ``(x, y, yaw)`` of the observer in the *current* body
                frame.  ``None`` means the observation was taken from here.

        Each detection contributes a von Mises ridge
        ``exp(kappa (cos(phi - theta) - 1))`` in the cell bearing ``phi``, with
        ``kappa`` scaled by the detection's confidence.  The ridge is uniform
        in range -- that is the whole point, and it is why motion triangulates.
        """
        obs = obs.filtered(self.cfg.min_weight, self.cfg.max_elevation_deg)
        if len(obs) == 0:
            return np.zeros(self.shape, dtype=np.float32)

        if observer_rel is None or not np.any(observer_rel):
            cos_b, sin_b = self._cos_bearing, self._sin_bearing
            near = self._near_mask
        else:
            ox, oy, oyaw = (float(v) for v in observer_rel)
            dx = self.cell_x - ox
            dy = self.cell_y - oy
            r = np.hypot(dx, dy)
            near = r < self.cfg.min_range
            inv_r = 1.0 / np.clip(r, 1e-9, None)
            # Rotating the unit vector by -oyaw is the same as taking the
            # bearing relative to the past observer's heading, minus an arctan.
            c_yaw, s_yaw = np.cos(oyaw), np.sin(oyaw)
            cos_b = (dx * c_yaw + dy * s_yaw) * inv_r
            sin_b = (dy * c_yaw - dx * s_yaw) * inv_r

        cfg = self.cfg
        theta = obs.azimuth[:, None, None]
        w = obs.weight[:, None, None]
        kappa = cfg.kappa_min + (cfg.kappa_max - cfg.kappa_min) * w

        # Broadcast over detections in one pass: (D, H, W).  cos(phi - theta)
        # via the difference identity avoids a full-grid cos per detection.
        cos_delta = cos_b[None] * np.cos(theta) + sin_b[None] * np.sin(theta)
        # ``cos - 1`` keeps the ridge peak at exactly 1.0 with no Bessel term.
        like = np.exp(kappa * (cos_delta - 1.0))
        # Confidence scales how far from the 0.5 prior we are willing to move.
        p_hit = 0.5 + (cfg.p_hit - 0.5) * w
        p = np.clip(cfg.p_miss + (p_hit - cfg.p_miss) * like, 1e-4, 1.0 - 1e-4)

        inc = np.log(p / (1.0 - p)).sum(axis=0).astype(np.float32)
        inc[near] = 0.0
        return inc

    def update(self, obs: DoAObservation, dt: float | None = None) -> None:
        """Fold a new observation into the posterior."""
        inc = self.ray_logodds(obs)
        self.logodds *= self.cfg.decay_for(
            self.cfg.dt_default if dt is None else dt)
        self.logodds += inc
        np.clip(self.logodds, -self.cfg.l_max, self.cfg.l_max, out=self.logodds)

        self.last_increment = inc
        self.history.append((self.pose.copy(), obs))
        self.n_updates += 1

    def step(self, obs: DoAObservation | None,
             delta_pose: np.ndarray | None = None,
             dt: float | None = None) -> None:
        """One filter iteration: motion warp, then measurement update."""
        self.predict(delta_pose)
        if obs is not None:
            self.update(obs, dt=dt)

    # ── readouts ──────────────────────────────────────────────────────────

    @property
    def probability(self) -> np.ndarray:
        """Posterior P(source in cell), shape (H, W), float32 in (0, 1)."""
        return _sigmoid(self.logodds)

    def instantaneous_maps(self) -> np.ndarray:
        """Ray maps of the last ``history_len`` observations, in the current frame.

        Shape ``(history_len, H, W)``, newest first, zero-padded when the
        history is not yet full.  Re-rendering from the stored bearings (rather
        than warping cached images) keeps these crisp -- no resampling blur --
        and it costs one vectorised pass per entry.

        Together with :attr:`probability` this is the "temporal field": the
        fused posterior plus the raw recent evidence that produced it.
        """
        n = max(self.cfg.history_len, 1)
        out = np.zeros((n,) + self.shape, dtype=np.float32)
        for k, (pose_hist, obs) in enumerate(reversed(self.history)):
            if k >= n:
                break
            rel = _relative(self.pose, pose_hist)
            out[k] = _sigmoid(self.ray_logodds(obs, observer_rel=rel))
        return out

    def channels(self) -> np.ndarray:
        """Full network input: ``(1 + history_len, H, W)`` float32 in [0, 1]."""
        return np.concatenate(
            [self.probability[None], self.instantaneous_maps()], axis=0)

    def map_estimate(self) -> tuple[float, float, float]:
        """MAP source position ``(x, y, p)`` in the current body frame."""
        idx = int(np.argmax(self.logodds))
        i, j = np.unravel_index(idx, self.shape)
        return float(self.cell_x[i, j]), float(self.cell_y[i, j]), \
            float(_sigmoid(self.logodds[i, j]))

    def expected_position(self, temperature: float | None = None
                          ) -> tuple[float, float, float]:
        """Softmax-weighted centroid of the posterior, and its spread.

        Returns ``(x, y, spread_m)`` in the current body frame.  Weights are
        ``softmax((L - L_max) / T)`` computed on the **log-odds**, not on the
        probability: once a cell passes ~p = 0.99 the sigmoid compresses every
        remaining difference away, so a centroid taken in probability space
        drifts along the ridge instead of settling on its peak.

        ``spread_m`` is the weighted RMS distance from the centroid -- a direct
        localisation-uncertainty readout.  It stays large while the belief is
        still a bearing-only ridge and shrinks once motion has triangulated it,
        which makes it the natural metric for "did driving help?".
        """
        T = self.cfg.readout_temperature if temperature is None else temperature
        L = self.logodds.astype(np.float64)
        w = np.exp((L - L.max()) / max(T, 1e-6))
        total = w.sum()
        if total <= 1e-12:
            return 0.0, 0.0, float(self.cfg.range_m)
        w = w / total

        cx = float((w * self.cell_x).sum())
        cy = float((w * self.cell_y).sum())
        var = float((w * ((self.cell_x - cx) ** 2 + (self.cell_y - cy) ** 2)).sum())
        return cx, cy, float(np.sqrt(max(var, 0.0)))

    def confidence(self) -> float:
        """Peak posterior probability -- "is there a source at all?"."""
        return float(_sigmoid(np.asarray(self.logodds.max())))

    def entropy(self) -> float:
        """Mean per-cell Bernoulli entropy in nats -- lower = sharper belief."""
        p = np.clip(self.probability, 1e-6, 1 - 1e-6)
        return float(np.mean(-(p * np.log(p) + (1 - p) * np.log(1 - p))))

    def world_to_cell(self, x: float, y: float) -> tuple[int, int] | None:
        """Body-frame metres -> (row, col), or None when outside the grid."""
        i = int(round((self.c - x) / self.cfg.resolution))
        j = int(round((self.c - y) / self.cfg.resolution))
        if 0 <= i < self.shape[0] and 0 <= j < self.shape[1]:
            return i, j
        return None


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return (1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float32)))).astype(np.float32)


def _relative(pose_from: np.ndarray, pose_to: np.ndarray) -> np.ndarray:
    """``pose_to`` expressed in the body frame of ``pose_from`` (local copy of
    :func:`~echodiffusion.utils.geometry.relative_pose`, kept inline to avoid a
    per-cell import in the hot loop)."""
    d = pose_to[:2] - pose_from[:2]
    R = rot2d(pose_from[2])
    local = d @ R
    return np.array([local[0], local[1], wrap_angle(pose_to[2] - pose_from[2])])

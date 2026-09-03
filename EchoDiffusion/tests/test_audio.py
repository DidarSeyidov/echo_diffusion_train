"""Tests for the ODAS front-end and the Bayesian BEV field."""

from __future__ import annotations

import numpy as np
import pytest

from echodiffusion.audio.bev_field import BEVFieldConfig, SoundBEVField
from echodiffusion.audio.odas import (ArrayExtrinsics, DoAObservation,
                                      dominant_bearing, ssl_to_observation,
                                      sst_to_observation, vectors_to_bearings)
from echodiffusion.utils.geometry import (interpolate_poses, relative_pose,
                                          world_to_body, wrap_angle)


# ── geometry ──────────────────────────────────────────────────────────────

def test_world_to_body_roundtrip():
    from echodiffusion.utils.geometry import body_to_world
    pose = np.array([1.5, -2.0, 0.7])
    pts = np.array([[3.0, 1.0], [-1.0, 4.0]])
    assert np.allclose(body_to_world(world_to_body(pts, pose), pose), pts)


def test_relative_pose_composes():
    a = np.array([1.0, 2.0, 0.3])
    b = np.array([2.5, 2.2, -0.4])
    d = relative_pose(a, b)
    # Applying the increment to `a` must land exactly on `b`.
    c, s = np.cos(a[2]), np.sin(a[2])
    recon = np.array([a[0] + c * d[0] - s * d[1],
                      a[1] + s * d[0] + c * d[1],
                      wrap_angle(a[2] + d[2])])
    assert np.allclose(recon, b)


def test_interpolate_poses_handles_yaw_wrap():
    t_ref = np.array([0.0, 1.0])
    poses = np.array([[0.0, 0.0, 3.0], [0.0, 0.0, -3.0]])   # crosses +/-pi
    out = interpolate_poses(np.array([0.5]), t_ref, poses)
    # Unwrapped, the midpoint is pi (== -pi), not 0.
    assert abs(abs(out[0, 2]) - np.pi) < 1e-6


# ── ODAS conversion ───────────────────────────────────────────────────────

def test_vectors_to_bearings_identity():
    ex = ArrayExtrinsics()
    vecs = np.array([[1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0]], dtype=float)
    az, el = vectors_to_bearings(vecs, ex)
    # Compared circularly: wrap_angle is half-open, so +180 comes back as -180.
    assert np.allclose(wrap_angle(az - np.radians([0, 90, 180, -90])), 0.0,
                       atol=1e-9)
    assert np.allclose(el, 0.0, atol=1e-9)


def test_vectors_to_bearings_elevation():
    ex = ArrayExtrinsics()
    az, el = vectors_to_bearings(np.array([[0.0, 0.0, 1.0]]), ex)
    assert np.isclose(np.degrees(el[0]), 90.0)      # straight up
    az, el = vectors_to_bearings(np.array([[1.0, 0.0, 1.0]]), ex)
    assert np.isclose(np.degrees(el[0]), 45.0)      # normalised internally


def test_azimuth_offset_and_flip():
    vec = np.array([[1.0, 0.0, 0.0]])
    az, _ = vectors_to_bearings(vec, ArrayExtrinsics(azimuth_offset_deg=90.0))
    assert np.isclose(np.degrees(az[0]), 90.0)

    vec = np.array([[0.0, 1.0, 0.0]])
    az, _ = vectors_to_bearings(vec, ArrayExtrinsics(flip_azimuth=True))
    assert np.isclose(np.degrees(az[0]), -90.0)


def test_rotation_rpy_yaw():
    """A 90 deg yaw of the array frame shifts every reported bearing by 90 deg."""
    ex = ArrayExtrinsics(rotation_rpy_deg=(0.0, 0.0, 90.0))
    az, _ = vectors_to_bearings(np.array([[1.0, 0.0, 0.0]]), ex)
    assert np.isclose(np.degrees(az[0]), 90.0)


def test_ssl_and_sst_decode_shapes():
    ex = ArrayExtrinsics()
    ssl = np.array([[1.0, 0.0, 0.0, 0.9], [0.0, 1.0, 0.0, 0.2]], dtype=np.float32)
    obs = ssl_to_observation(0.0, ssl, ex)
    assert len(obs) == 2 and np.all(obs.track_id == -1)

    sst = np.array([[7, 1.0, 0.0, 0.0, 0.8]], dtype=np.float32)
    obs = sst_to_observation(0.0, sst, ex)
    assert obs.track_id[0] == 7 and np.isclose(obs.weight[0], 0.8)


def test_observation_filtering():
    obs = DoAObservation(
        0.0,
        azimuth=np.array([0.0, 1.0, 2.0]),
        elevation=np.radians([10.0, 89.9, 20.0]),
        weight=np.array([0.9, 0.9, 0.01]),
        track_id=np.array([-1, -1, -1]),
    )
    kept = obs.filtered(min_weight=0.05, max_elevation_deg=85.0)
    assert len(kept) == 1 and np.isclose(kept.azimuth[0], 0.0)


def test_dominant_bearing_wraps():
    """Bearings straddling +/-pi must average to pi, not to 0."""
    obs = DoAObservation(0.0, np.array([3.10, -3.10]), np.zeros(2),
                         np.ones(2), np.full(2, -1))
    az, r = dominant_bearing(obs)
    assert abs(abs(az) - np.pi) < 0.1 and r > 0.9


# ── BEV field ─────────────────────────────────────────────────────────────

def _obs(bearing: float, weight: float = 0.8) -> DoAObservation:
    return DoAObservation(0.0, np.array([bearing]), np.array([0.0]),
                          np.array([weight]), np.array([-1]))


def test_field_starts_uninformative():
    f = SoundBEVField(BEVFieldConfig())
    assert np.allclose(f.probability, 0.5)
    assert f.entropy() == pytest.approx(np.log(2), rel=1e-3)


def test_single_detection_puts_mass_on_the_bearing():
    f = SoundBEVField(BEVFieldConfig(range_m=6.0, resolution=0.15))
    f.update(_obs(np.radians(45.0)))
    x, y, _ = f.map_estimate()
    assert np.isclose(np.degrees(np.arctan2(y, x)), 45.0, atol=6.0)


def test_motion_triangulates_better_than_standing_still():
    """The core claim: driving sideways sharpens the posterior.

    A stationary observer can never recover range from bearings alone, so its
    posterior stays a ridge; a moving one intersects rays from different
    positions.  Spread is the readout that has to show it.
    """
    src = np.array([3.0, 1.5])
    cfg = BEVFieldConfig(range_m=6.0, resolution=0.15, decay_half_life_s=6.0)
    rng = np.random.default_rng(0)

    spreads = {}
    for label, path in (
        ("static", [np.array([0.0, 0.0, 0.0])] * 25),
        ("lateral", [np.array([0.0, 0.1 * k, 0.0]) for k in range(25)]),
    ):
        f = SoundBEVField(cfg)
        prev = None
        for pose in path:
            rel = world_to_body(src, pose)
            bearing = np.arctan2(rel[1], rel[0]) + rng.normal(0, 0.03)
            delta = None if prev is None else relative_pose(prev, pose)
            f.step(_obs(bearing), delta, dt=0.1)
            prev = pose
        spreads[label] = f.expected_position()[2]

    assert spreads["lateral"] < spreads["static"]


def test_motion_warp_tracks_a_translating_frame():
    """Belief must move with the robot: drive 1 m forward and the peak, which
    was 3 m ahead, has to sit ~2 m ahead in the new body frame."""
    cfg = BEVFieldConfig(range_m=6.0, resolution=0.1, decay_half_life_s=1e6)
    f = SoundBEVField(cfg)

    # Concentrate belief on a single cell rather than a whole ray.
    i, j = f.world_to_cell(3.0, 0.0)
    f.logodds[i, j] = 8.0

    f.predict(np.array([1.0, 0.0, 0.0]))
    x, y, _ = f.map_estimate()
    assert x == pytest.approx(2.0, abs=0.2)
    assert y == pytest.approx(0.0, abs=0.2)


def test_rotation_warp():
    """Rotating the robot +90 deg moves a target that was ahead to its right."""
    cfg = BEVFieldConfig(range_m=6.0, resolution=0.1, decay_half_life_s=1e6)
    f = SoundBEVField(cfg)
    i, j = f.world_to_cell(3.0, 0.0)
    f.logodds[i, j] = 8.0

    f.predict(np.array([0.0, 0.0, np.pi / 2]))
    x, y, _ = f.map_estimate()
    assert x == pytest.approx(0.0, abs=0.25)
    assert y == pytest.approx(-3.0, abs=0.25)


def test_decay_pulls_back_toward_the_prior():
    cfg = BEVFieldConfig(decay_half_life_s=1.0)
    f = SoundBEVField(cfg)
    f.update(_obs(0.0), dt=0.1)
    peak = f.logodds.max()
    for _ in range(50):
        f.predict(None)
        f.logodds *= cfg.decay_for(0.1)
    assert f.logodds.max() < peak * 0.1


def test_empty_observation_is_a_noop():
    f = SoundBEVField(BEVFieldConfig())
    empty = DoAObservation(0.0, np.zeros(0), np.zeros(0), np.zeros(0),
                           np.zeros(0, dtype=np.int64))
    f.update(empty)
    assert np.all(np.isfinite(f.logodds))
    assert f.expected_position()[2] >= 0.0


def test_channels_shape_and_range():
    cfg = BEVFieldConfig(history_len=3, range_m=6.0, resolution=0.15)
    f = SoundBEVField(cfg)
    for _ in range(5):
        f.step(_obs(0.5), np.array([0.05, 0.0, 0.0]), dt=0.1)
    ch = f.channels()
    assert ch.shape == (4, cfg.grid_size, cfg.grid_size)
    assert ch.min() >= 0.0 and ch.max() <= 1.0

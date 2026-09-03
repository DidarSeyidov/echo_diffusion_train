"""Tests for episodes, the CDR/rosbag reader, and the torch dataset."""

from __future__ import annotations

import struct

import numpy as np
import pytest
import torch

from echodiffusion.data.dataset import DataConfig, EchoTrajectoryDataset
from echodiffusion.data.episode import Episode, longest_true_run
from echodiffusion.data.ros_messages import (decode_audio_frame, decode_odas_ssl,
                                             decode_odas_sst)
from echodiffusion.data.synthetic import simulate_episode


# ── CDR decoding ──────────────────────────────────────────────────────────

def _encapsulate(payload: bytes) -> bytes:
    return b"\x00\x01\x00\x00" + payload


def _header(frame_id: str = "odas") -> bytes:
    raw = frame_id.encode() + b"\x00"
    out = struct.pack("<iII", 1784767037, 234161316, len(raw)) + raw
    out += b"\x00" * ((-len(out)) % 4)          # pad to a 4-byte boundary
    return out


def test_decode_odas_ssl():
    sources = [(1.0, 0.0, 0.0, 0.9), (0.0, 1.0, 0.0, 0.1)]
    payload = _header() + struct.pack("<I", len(sources))
    for row in sources:
        payload += struct.pack("<4d", *row)

    msg = decode_odas_ssl(_encapsulate(payload))
    assert msg["frame_id"] == "odas"
    assert msg["sources"].shape == (2, 4)
    assert np.allclose(msg["sources"][0], [1.0, 0.0, 0.0, 0.9])


def test_decode_odas_sst():
    payload = _header() + struct.pack("<I", 1)
    payload += struct.pack("<q", 2887) + struct.pack("<4d", 0.0, 0.0, 1.0, 0.75)

    msg = decode_odas_sst(_encapsulate(payload))
    assert msg["sources"].shape == (1, 5)
    assert msg["sources"][0, 0] == 2887
    assert msg["sources"][0, 4] == pytest.approx(0.75)


def test_decode_audio_frame():
    fmt = b"signed_16\x00"
    payload = _header("") + struct.pack("<I", len(fmt)) + fmt
    payload += b"\x00" * ((-len(payload)) % 4)
    channels, samples = 6, 4
    payload += struct.pack("<III", channels, 16000, samples)
    data = np.arange(channels * samples, dtype="<i2").tobytes()
    payload += struct.pack("<I", len(data)) + data

    msg = decode_audio_frame(_encapsulate(payload))
    assert msg["channel_count"] == 6
    assert msg["sampling_frequency"] == 16000
    # De-interleaved to channel-major and scaled into [-1, 1].
    assert msg["data"].shape == (channels, samples)
    assert abs(msg["data"]).max() <= 1.0

    meta_only = decode_audio_frame(_encapsulate(payload), decode_samples=False)
    assert meta_only["data"] is None


# ── episodes ──────────────────────────────────────────────────────────────

def test_longest_true_run():
    assert longest_true_run(np.array([0, 1, 1, 0, 1, 1, 1, 0], dtype=bool)) == (4, 7)
    assert longest_true_run(np.zeros(5, dtype=bool)) == (0, 0)
    assert longest_true_run(np.ones(3, dtype=bool)) == (0, 3)


def test_episode_roundtrip(tmp_path):
    ep = simulate_episode(seed=1, duration_s=5.0)
    ep.save(tmp_path)
    loaded = Episode.load(tmp_path)

    assert len(loaded) == len(ep)
    assert np.allclose(loaded.pose_vicon, ep.pose_vicon, equal_nan=True)
    assert np.allclose(loaded.ssl, ep.ssl)
    assert set(loaded.meta.pose_sources) == {"odometry", "vicon"}


def test_pose_source_resolution():
    ep = simulate_episode(seed=2, duration_s=5.0)
    assert ep.resolve_pose_source("vicon") == "vicon"
    assert ep.resolve_pose_source("odometry") == "odometry"
    assert ep.resolve_pose_source("auto") == "vicon"     # auto prefers vicon

    # Drop vicon and the same requests must degrade sensibly.
    ep.pose_vicon[:] = np.nan
    assert ep.resolve_pose_source("auto") == "odometry"
    assert ep.resolve_pose_source("vicon", fallback=True) == "odometry"
    assert ep.resolve_pose_source("vicon", fallback=False) is None


def test_valid_range_skips_nan_gap():
    ep = simulate_episode(seed=3, duration_s=10.0)
    ep.pose_vicon[40:45] = np.nan
    lo, hi = ep.valid_range("vicon")
    assert (lo, hi) == (45, len(ep))          # the later run is the longer one


def test_expert_actually_approaches_the_source():
    """The synthetic supervision is only useful if the expert homes in."""
    for seed in range(5):
        ep = simulate_episode(seed=seed, duration_s=30.0)
        d = np.linalg.norm(ep.source_pose - ep.pose_vicon[:, :2], axis=1)
        assert d[-1] < d[0], f"seed {seed}: expert did not close the distance"


# ── dataset ───────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def episode_dirs(tmp_path_factory):
    root = tmp_path_factory.mktemp("episodes")
    dirs = []
    for i in range(3):
        d = root / f"ep_{i}"
        simulate_episode(seed=i, duration_s=15.0, name=f"ep_{i}").save(d)
        dirs.append(d)
    return dirs


def test_dataset_sample_shapes(episode_dirs):
    cfg = DataConfig(horizon=20, warmup=15, past_len=10, doa_frames=4,
                     max_doa_tokens=8, traj_scale=1.0, pose_source="vicon",
                     bev={"range_m": 4.0, "resolution": 0.2, "history_len": 3})
    ds = EchoTrajectoryDataset(episode_dirs, cfg)
    assert len(ds) > 0

    s = ds[len(ds) // 2]
    assert s["bev"].shape == (4, 40, 40)        # 1 posterior + 3 history
    assert s["doa"].shape == (4, 8, 6)
    assert s["doa_mask"].shape == (4, 8)
    assert s["past"].shape == (10, 3)
    assert s["traj"].shape == (20, 2)
    assert s["field_estimate"].shape == (4,)
    assert torch.isfinite(s["bev"]).all()
    assert torch.isfinite(s["traj"]).all()
    assert 0.0 <= s["bev"].min() and s["bev"].max() <= 1.0


def test_dataset_targets_are_body_frame(episode_dirs):
    """The first waypoint must be near the origin -- the robot is at (0, 0)."""
    cfg = DataConfig(horizon=20, warmup=5, traj_scale=1.0, pose_source="vicon")
    ds = EchoTrajectoryDataset(episode_dirs, cfg)
    for idx in (0, len(ds) // 3, len(ds) - 1):
        traj = ds[idx]["traj"].numpy()
        assert np.linalg.norm(traj[0]) < 0.3    # one step at <=0.5 m/s, 0.1 s


def test_auto_traj_scale_is_positive(episode_dirs):
    cfg = DataConfig(horizon=20, warmup=5, traj_scale="auto", pose_source="vicon")
    ds = EchoTrajectoryDataset(episode_dirs, cfg)
    assert ds.traj_scale > 0
    # With the scale fitted at the 99th percentile, targets stay ~[-1, 1].
    assert abs(ds[len(ds) // 2]["traj"]).max() < 3.0


def test_explicit_traj_scale_overrides_auto(episode_dirs):
    cfg = DataConfig(horizon=20, warmup=5, traj_scale="auto", pose_source="vicon")
    ds = EchoTrajectoryDataset(episode_dirs, cfg, traj_scale=2.5)
    assert ds.traj_scale == 2.5


def test_dataset_rejects_episodes_without_poses(episode_dirs, tmp_path):
    ep = simulate_episode(seed=99, duration_s=10.0, name="no_pose")
    ep.pose_vicon[:] = np.nan
    ep.pose_odometry[:] = np.nan
    d = tmp_path / "no_pose"
    ep.save(d)

    cfg = DataConfig(horizon=20, warmup=5, pose_source="auto")
    with pytest.raises(ValueError, match="no window"):
        EchoTrajectoryDataset([d], cfg)

    # ...but it is still usable for inference over audio alone.
    ds = EchoTrajectoryDataset([d], cfg, require_targets=False)
    assert len(ds) > 0
    assert ds[0]["traj_valid"].item() == 0.0


def test_strict_pose_source_is_honoured(tmp_path):
    ep = simulate_episode(seed=5, duration_s=10.0, name="odom_only")
    ep.pose_vicon[:] = np.nan
    d = tmp_path / "odom_only"
    ep.save(d)

    strict = DataConfig(horizon=20, warmup=5, pose_source="vicon",
                        pose_fallback=False)
    with pytest.raises(ValueError, match="no window"):
        EchoTrajectoryDataset([d], strict)

    lenient = DataConfig(horizon=20, warmup=5, pose_source="vicon",
                         pose_fallback=True)
    assert len(EchoTrajectoryDataset([d], lenient)) > 0

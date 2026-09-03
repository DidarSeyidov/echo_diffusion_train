"""Decoders for the ROS 2 message types EchoDiffusion reads out of a bag.

Every decoder takes the raw CDR blob and returns a plain dict / numpy array --
no ROS types leak into the rest of the codebase.  Layouts were verified byte
for byte against ``session_audio/session_01`` (ODAS + audio_utils, Humble).

The pose / image decoders are already wired up even though the current bags do
not carry those topics yet: once they are recorded, ``prepare_dataset.py``
picks them up with no code change.
"""

from __future__ import annotations

import numpy as np

from .cdr import CDRReader

# ── ODAS ──────────────────────────────────────────────────────────────────
# odas_ros_msgs/msg/OdasSsl : float64 x, y, z, E
# odas_ros_msgs/msg/OdasSst : int64 id; float64 x, y, z, activity
ODAS_SSL_FIELDS = ("x", "y", "z", "E")
ODAS_SST_FIELDS = ("id", "x", "y", "z", "activity")


def decode_odas_ssl(blob: bytes) -> dict:
    """``odas_ros_msgs/msg/OdasSslArrayStamped``.

    Returns ``{'stamp', 'frame_id', 'sources': (K, 4) float32}`` where the
    columns are (x, y, z, E): a unit direction-of-arrival vector in the array
    frame plus the potential source's energy in [0, 1].

    SSL reports *potential* sources every ODAS hop -- no track identity and no
    temporal smoothing.  Energy is the usable confidence signal here.
    """
    r = CDRReader(blob)
    stamp, frame_id = r.header()
    n = r.seq_len()
    # All four members are float64 and the struct needs no interior padding,
    # so the whole sequence is one contiguous (n, 4) block.
    sources = r.array("f8", n * 4).reshape(n, 4).astype(np.float32)
    return {"stamp": stamp, "frame_id": frame_id, "sources": sources}


def decode_odas_sst(blob: bytes) -> dict:
    """``odas_ros_msgs/msg/OdasSstArrayStamped``.

    Returns ``{'stamp', 'frame_id', 'sources': (M, 5) float32}`` with columns
    (id, x, y, z, activity).  SST sources are *tracked*: ``id`` is stable while
    a source stays alive, and ``activity`` in [0, 1] says whether the track is
    currently emitting.  A dormant track keeps its last direction with
    ``activity`` near 0, so downstream code must gate on activity.
    """
    r = CDRReader(blob)
    stamp, frame_id = r.header()
    n = r.seq_len()
    out = np.zeros((n, 5), dtype=np.float32)
    for i in range(n):
        out[i, 0] = r.int64()
        out[i, 1] = r.float64()
        out[i, 2] = r.float64()
        out[i, 3] = r.float64()
        out[i, 4] = r.float64()
    return {"stamp": stamp, "frame_id": frame_id, "sources": out}


# ── audio_utils ───────────────────────────────────────────────────────────
_AUDIO_FORMAT_DTYPE = {
    "signed_8": np.int8,
    "unsigned_8": np.uint8,
    "signed_16": np.int16,
    "unsigned_16": np.uint16,
    "signed_24": None,       # packed 24-bit, not supported
    "signed_32": np.int32,
    "unsigned_32": np.uint32,
    "float": np.float32,
    "double": np.float64,
}


def decode_audio_frame(blob: bytes, decode_samples: bool = True) -> dict:
    """``audio_utils_msgs/msg/AudioFrame``.

    Returns ``{'stamp', 'frame_id', 'format', 'channel_count',
    'sampling_frequency', 'frame_sample_count', 'data': (C, N) float32}``.

    Samples are de-interleaved to channel-major and normalised to [-1, 1].
    Pass ``decode_samples=False`` to skip the payload when only the timing
    metadata is needed -- that is the common case, since the ODAS front-end
    already turned the waveform into directions.
    """
    r = CDRReader(blob)
    stamp, frame_id = r.header()
    fmt = r.string()
    channel_count = r.uint32()
    sampling_frequency = r.uint32()
    frame_sample_count = r.uint32()

    out = {
        "stamp": stamp,
        "frame_id": frame_id,
        "format": fmt,
        "channel_count": int(channel_count),
        "sampling_frequency": int(sampling_frequency),
        "frame_sample_count": int(frame_sample_count),
        "data": None,
    }
    if not decode_samples:
        return out

    dtype = _AUDIO_FORMAT_DTYPE.get(fmt)
    if dtype is None:
        raise ValueError(f"unsupported AudioFrame format {fmt!r}")

    raw = r.sequence("u1")
    samples = raw.view(dtype)
    if channel_count > 0:
        samples = samples.reshape(-1, channel_count).T  # (C, N), de-interleaved
    if np.issubdtype(dtype, np.integer):
        samples = samples.astype(np.float32) / float(np.iinfo(dtype).max)
    else:
        samples = samples.astype(np.float32)
    out["data"] = np.ascontiguousarray(samples)
    return out


# ── geometry / nav ────────────────────────────────────────────────────────

def _read_pose(r: CDRReader) -> tuple[np.ndarray, np.ndarray]:
    """``geometry_msgs/Pose`` -> (xyz, quaternion xyzw)."""
    pos = np.array([r.float64(), r.float64(), r.float64()], dtype=np.float64)
    quat = np.array([r.float64(), r.float64(), r.float64(), r.float64()],
                    dtype=np.float64)
    return pos, quat


def decode_odometry(blob: bytes) -> dict:
    """``nav_msgs/msg/Odometry`` -> stamp, position, quaternion, twist."""
    r = CDRReader(blob)
    stamp, frame_id = r.header()
    child_frame_id = r.string()
    pos, quat = _read_pose(r)
    r.array("f8", 36)                       # pose covariance (unused)
    lin = np.array([r.float64(), r.float64(), r.float64()])
    ang = np.array([r.float64(), r.float64(), r.float64()])
    return {
        "stamp": stamp,
        "frame_id": frame_id,
        "child_frame_id": child_frame_id,
        "position": pos,
        "orientation": quat,
        "linear_velocity": lin,
        "angular_velocity": ang,
    }


def decode_pose_stamped(blob: bytes) -> dict:
    """``geometry_msgs/msg/PoseStamped`` -- the usual Vicon/mocap bridge type."""
    r = CDRReader(blob)
    stamp, frame_id = r.header()
    pos, quat = _read_pose(r)
    return {"stamp": stamp, "frame_id": frame_id,
            "position": pos, "orientation": quat}


def decode_transform_stamped(blob: bytes) -> dict:
    """``geometry_msgs/msg/TransformStamped`` -- Vicon bridges often use this."""
    r = CDRReader(blob)
    stamp, frame_id = r.header()
    child_frame_id = r.string()
    pos = np.array([r.float64(), r.float64(), r.float64()])
    quat = np.array([r.float64(), r.float64(), r.float64(), r.float64()])
    return {"stamp": stamp, "frame_id": frame_id, "child_frame_id": child_frame_id,
            "position": pos, "orientation": quat}


def decode_image(blob: bytes) -> dict:
    """``sensor_msgs/msg/Image`` -> HxWxC uint8 (or raw dtype) array."""
    r = CDRReader(blob)
    stamp, frame_id = r.header()
    height = r.uint32()
    width = r.uint32()
    encoding = r.string()
    r.uint8()                               # is_bigendian
    r.uint32()                              # step
    data = r.sequence("u1")

    channels = {"mono8": 1, "8UC1": 1, "bgr8": 3, "rgb8": 3, "8UC3": 3,
                "bgra8": 4, "rgba8": 4}.get(encoding)
    if channels is None:
        img = data                          # leave exotic encodings to the caller
    else:
        img = data.reshape(height, width, channels)
        if encoding.startswith("bgr"):
            img = img[..., ::-1]            # -> RGB
        elif encoding.startswith("bgra"):
            img = img[..., [2, 1, 0, 3]]
    return {"stamp": stamp, "frame_id": frame_id, "height": int(height),
            "width": int(width), "encoding": encoding, "data": img}


def decode_compressed_image(blob: bytes) -> dict:
    """``sensor_msgs/msg/CompressedImage`` -> undecoded JPEG/PNG bytes."""
    r = CDRReader(blob)
    stamp, frame_id = r.header()
    fmt = r.string()
    data = r.sequence("u1")
    return {"stamp": stamp, "frame_id": frame_id, "format": fmt,
            "data": data.tobytes()}


DECODERS = {
    "odas_ros_msgs/msg/OdasSslArrayStamped": decode_odas_ssl,
    "odas_ros_msgs/msg/OdasSstArrayStamped": decode_odas_sst,
    "audio_utils_msgs/msg/AudioFrame": decode_audio_frame,
    "nav_msgs/msg/Odometry": decode_odometry,
    "geometry_msgs/msg/PoseStamped": decode_pose_stamped,
    "geometry_msgs/msg/TransformStamped": decode_transform_stamped,
    "sensor_msgs/msg/Image": decode_image,
    "sensor_msgs/msg/CompressedImage": decode_compressed_image,
}

"""Minimal CDR (Common Data Representation) reader for ROS 2 messages.

rosbag2 stores each message as a CDR-encapsulated blob: a 4-byte encapsulation
header (``00 01 00 00`` for little-endian) followed by the serialised fields.
Alignment padding is computed relative to the *start of the payload*, i.e. from
byte 4 of the blob -- this is the detail that trips up most hand-rolled readers.

Only the handful of types EchoDiffusion consumes are implemented (see
``echodiffusion/data/ros_messages.py``).  That keeps the dataset pipeline free
of any ROS or ``rosbags`` install, which matters because training usually runs
in a plain conda env with no ROS on the path.
"""

from __future__ import annotations

import struct

import numpy as np

# Encapsulation identifiers from the OMG CDR spec (only the two PLAIN_CDR
# variants show up in rosbag2 output).
CDR_LE = 0x0001
CDR_BE = 0x0000


class CDRReader:
    """Sequential reader over a single CDR-encapsulated ROS 2 message."""

    def __init__(self, buf: bytes):
        if len(buf) < 4:
            raise ValueError(f"CDR blob too short: {len(buf)} bytes")
        # Byte 0 is reserved, byte 1 selects endianness.
        eid = buf[1]
        if eid == CDR_LE:
            self.endian = "<"
        elif eid == CDR_BE:
            self.endian = ">"
        else:
            raise ValueError(f"unsupported CDR encapsulation 0x{eid:04x}")
        self.buf = buf
        # Payload starts after the 4-byte encapsulation header; every
        # alignment below is relative to this origin.
        self.origin = 4
        self.pos = 4

    # ── primitives ────────────────────────────────────────────────────────

    def _align(self, size: int) -> None:
        rel = self.pos - self.origin
        pad = (-rel) % size
        self.pos += pad

    def _scalar(self, fmt: str, size: int):
        self._align(size)
        (val,) = struct.unpack_from(self.endian + fmt, self.buf, self.pos)
        self.pos += size
        return val

    def uint8(self) -> int:
        return self._scalar("B", 1)

    def int8(self) -> int:
        return self._scalar("b", 1)

    def bool(self) -> bool:
        return bool(self._scalar("B", 1))

    def int32(self) -> int:
        return self._scalar("i", 4)

    def uint32(self) -> int:
        return self._scalar("I", 4)

    def int64(self) -> int:
        return self._scalar("q", 8)

    def uint64(self) -> int:
        return self._scalar("Q", 8)

    def float32(self) -> float:
        return self._scalar("f", 4)

    def float64(self) -> float:
        return self._scalar("d", 8)

    def string(self) -> str:
        n = self.uint32()
        # Length includes the trailing NUL, which we drop.
        raw = self.buf[self.pos:self.pos + max(n - 1, 0)]
        self.pos += n
        return raw.decode("utf-8", errors="replace")

    def array(self, dtype: str, count: int) -> np.ndarray:
        """Read ``count`` primitives of numpy ``dtype`` as a contiguous block."""
        dt = np.dtype(self.endian + dtype)
        self._align(dt.itemsize)
        out = np.frombuffer(self.buf, dtype=dt, count=count, offset=self.pos)
        self.pos += count * dt.itemsize
        return out

    def sequence(self, dtype: str) -> np.ndarray:
        """Read a length-prefixed sequence of primitives."""
        n = self.uint32()
        return self.array(dtype, n)

    def seq_len(self) -> int:
        """Read just the length prefix of a sequence of complex members."""
        return self.uint32()

    # ── common ROS structs ────────────────────────────────────────────────

    def time(self) -> float:
        """``builtin_interfaces/Time`` -> seconds as float."""
        sec = self.int32()
        nanosec = self.uint32()
        return sec + nanosec * 1e-9

    def header(self) -> tuple[float, str]:
        """``std_msgs/Header`` -> (stamp_seconds, frame_id)."""
        stamp = self.time()
        frame_id = self.string()
        return stamp, frame_id

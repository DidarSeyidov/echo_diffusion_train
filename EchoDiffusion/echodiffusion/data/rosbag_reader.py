"""Read rosbag2 sqlite3 (.db3) sessions without a ROS installation.

A rosbag2 "session" is a directory holding ``metadata.yaml`` plus one or more
``*.db3`` shards.  Each shard has a ``topics`` table (id, name, type) and a
``messages`` table (topic_id, timestamp, blob).  This module joins the two,
decodes blobs via :mod:`echodiffusion.data.ros_messages`, and yields messages
in timestamp order across shards.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Sequence

import yaml

from .ros_messages import DECODERS


@dataclass
class TopicInfo:
    name: str
    msg_type: str
    count: int = 0

    @property
    def decodable(self) -> bool:
        return self.msg_type in DECODERS


@dataclass
class BagMessage:
    topic: str
    msg_type: str
    #: bag receive time (seconds).  Prefer ``msg['stamp']`` when the publisher
    #: filled the header -- receive time includes transport latency.
    t_recv: float
    msg: dict = field(repr=False)


class Rosbag2Reader:
    """Sequential reader over one rosbag2 session directory."""

    def __init__(self, session_dir: str | Path):
        self.session_dir = Path(session_dir)
        if not self.session_dir.is_dir():
            raise FileNotFoundError(f"not a bag session directory: {session_dir}")

        self.shards = sorted(self.session_dir.glob("*.db3"))
        if not self.shards:
            raise FileNotFoundError(f"no .db3 shards under {session_dir}")

        self.metadata = self._read_metadata()
        self.topics = self._read_topics()

    # ── metadata ──────────────────────────────────────────────────────────

    def _read_metadata(self) -> dict:
        meta_path = self.session_dir / "metadata.yaml"
        if not meta_path.exists():
            return {}
        with open(meta_path) as f:
            return yaml.safe_load(f).get("rosbag2_bagfile_information", {})

    def _read_topics(self) -> dict[str, TopicInfo]:
        topics: dict[str, TopicInfo] = {}
        for entry in self.metadata.get("topics_with_message_count", []):
            tm = entry["topic_metadata"]
            topics[tm["name"]] = TopicInfo(tm["name"], tm["type"],
                                           int(entry.get("message_count", 0)))
        if topics:
            return topics
        # No metadata.yaml -- fall back to the per-shard topics table.
        for shard in self.shards:
            with sqlite3.connect(f"file:{shard}?mode=ro", uri=True) as con:
                for name, mtype in con.execute("SELECT name, type FROM topics"):
                    topics.setdefault(name, TopicInfo(name, mtype))
        return topics

    @property
    def duration(self) -> float:
        return self.metadata.get("duration", {}).get("nanoseconds", 0) * 1e-9

    @property
    def start_time(self) -> float:
        return self.metadata.get("starting_time", {}).get(
            "nanoseconds_since_epoch", 0) * 1e-9

    # ── iteration ─────────────────────────────────────────────────────────

    def read(
        self,
        topics: Sequence[str] | None = None,
        decode: bool = True,
        **decoder_kwargs,
    ) -> Iterator[BagMessage]:
        """Yield messages in timestamp order.

        Args:
            topics: topic names to read; ``None`` reads every decodable topic.
            decode: when False, ``BagMessage.msg`` holds ``{'raw': bytes}``.
            decoder_kwargs: forwarded to the type decoder (e.g.
                ``decode_samples=False`` to skip AudioFrame payloads).
        """
        wanted = set(topics) if topics is not None else None

        for shard in self.shards:
            with sqlite3.connect(f"file:{shard}?mode=ro", uri=True) as con:
                id_to_topic = {
                    tid: (name, mtype)
                    for tid, name, mtype in con.execute(
                        "SELECT id, name, type FROM topics")
                }
                keep = {
                    tid for tid, (name, mtype) in id_to_topic.items()
                    if (wanted is None or name in wanted)
                    and (not decode or mtype in DECODERS)
                }
                if not keep:
                    continue

                placeholders = ",".join("?" * len(keep))
                query = (
                    f"SELECT topic_id, timestamp, data FROM messages "
                    f"WHERE topic_id IN ({placeholders}) ORDER BY timestamp"
                )
                for tid, stamp_ns, blob in con.execute(query, tuple(keep)):
                    name, mtype = id_to_topic[tid]
                    if decode:
                        msg = DECODERS[mtype](blob, **decoder_kwargs)
                    else:
                        msg = {"raw": blob}
                    yield BagMessage(name, mtype, stamp_ns * 1e-9, msg)

    def read_topic(self, topic: str, **kwargs) -> list[BagMessage]:
        """Collect one topic into a list (convenience for small topics)."""
        return list(self.read(topics=[topic], **kwargs))

    def summary(self) -> str:
        lines = [
            f"session: {self.session_dir.name}",
            f"duration: {self.duration:.1f} s   "
            f"messages: {self.metadata.get('message_count', '?')}",
            "topics:",
        ]
        for t in sorted(self.topics.values(), key=lambda x: x.name):
            rate = t.count / self.duration if self.duration > 0 else 0.0
            mark = " " if t.decodable else "!"
            lines.append(f"  {mark} {t.name:<24} {t.msg_type:<44} "
                         f"{t.count:>7} msgs  {rate:6.1f} Hz")
        if any(not t.decodable for t in self.topics.values()):
            lines.append("  (! = no decoder; add one in data/ros_messages.py)")
        return "\n".join(lines)


def find_sessions(root: str | Path) -> list[Path]:
    """Find every rosbag2 session directory under ``root`` (recursively).

    A directory counts as a session when it contains at least one ``.db3``
    shard.  ``root`` itself is included when it is a session.
    """
    root = Path(root)
    if not root.exists():
        return []
    sessions = {p.parent for p in root.rglob("*.db3")}
    return sorted(sessions)

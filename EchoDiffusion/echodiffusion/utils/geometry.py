"""SE(2) helpers.

The robot is treated as planar throughout: a pose is ``(x, y, yaw)`` in a fixed
world frame, with the body frame following the ROS convention -- **x forward,
y left, yaw counter-clockwise**.
"""

from __future__ import annotations

import numpy as np


def quat_to_yaw(quat: np.ndarray) -> np.ndarray:
    """Quaternion (x, y, z, w) -> yaw in radians.  Accepts (4,) or (N, 4)."""
    q = np.atleast_2d(np.asarray(quat, dtype=np.float64))
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return yaw if np.ndim(quat) > 1 else yaw[0]


def wrap_angle(a):
    """Wrap angle(s) to ``[-pi, pi)``.

    Note the half-open end: exactly +pi maps to -pi.  They denote the same
    bearing, so compare wrapped angles with a circular difference rather than
    for equality.
    """
    return (np.asarray(a) + np.pi) % (2.0 * np.pi) - np.pi


def rot2d(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def world_to_body(points_w: np.ndarray, pose: np.ndarray) -> np.ndarray:
    """Express world points in the body frame of ``pose = (x, y, yaw)``.

    ``points_w`` is (..., 2); the result has the same shape.
    """
    pose = np.asarray(pose, dtype=np.float64)
    delta = np.asarray(points_w, dtype=np.float64) - pose[:2]
    return delta @ rot2d(pose[2])           # == R(-yaw) @ delta


def body_to_world(points_b: np.ndarray, pose: np.ndarray) -> np.ndarray:
    """Inverse of :func:`world_to_body`."""
    pose = np.asarray(pose, dtype=np.float64)
    return np.asarray(points_b, dtype=np.float64) @ rot2d(pose[2]).T + pose[:2]


def relative_pose(pose_from: np.ndarray, pose_to: np.ndarray) -> np.ndarray:
    """Pose of ``pose_to`` expressed in the body frame of ``pose_from``.

    Returns ``(dx, dy, dyaw)`` -- the odometry increment used to warp the BEV
    field between consecutive observation frames.
    """
    pose_from = np.asarray(pose_from, dtype=np.float64)
    pose_to = np.asarray(pose_to, dtype=np.float64)
    d_xy = world_to_body(pose_to[:2], pose_from)
    return np.array([d_xy[0], d_xy[1], wrap_angle(pose_to[2] - pose_from[2])])


def interpolate_poses(t_query: np.ndarray, t_ref: np.ndarray,
                      poses_ref: np.ndarray) -> np.ndarray:
    """Linearly interpolate (x, y, yaw) poses onto ``t_query`` timestamps.

    Yaw is interpolated through its unwrapped form so a +/-pi crossing does not
    produce a spurious full rotation.  Queries outside ``t_ref`` are clamped to
    the endpoints.
    """
    t_ref = np.asarray(t_ref, dtype=np.float64)
    poses_ref = np.asarray(poses_ref, dtype=np.float64)
    order = np.argsort(t_ref)
    t_ref, poses_ref = t_ref[order], poses_ref[order]

    x = np.interp(t_query, t_ref, poses_ref[:, 0])
    y = np.interp(t_query, t_ref, poses_ref[:, 1])
    yaw = np.interp(t_query, t_ref, np.unwrap(poses_ref[:, 2]))
    return np.stack([x, y, wrap_angle(yaw)], axis=-1)

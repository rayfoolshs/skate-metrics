"""Turns per-frame pose keypoints into the 1-D signals the analysis needs."""

from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import percentile_filter

from .geometry import interpolate_nan, line_angle, unwrap_angles
from .models import (
    FOOT_KEYPOINTS,
    LEFT_HIP,
    LEFT_SHOULDER,
    PoseFrame,
    RIGHT_HIP,
    RIGHT_SHOULDER,
)


def _point_xy(frame: PoseFrame, name: str) -> tuple[float, float] | None:
    point = frame.get(name)
    if point is None:
        return None
    return point.x, point.y


def _pair_angle(frame: PoseFrame, a: str, b: str) -> float:
    pa = _point_xy(frame, a)
    pb = _point_xy(frame, b)
    if pa is None or pb is None:
        return math.nan
    if math.hypot(pb[0] - pa[0], pb[1] - pa[1]) < 4.0:
        return math.nan
    return line_angle(pa, pb)


def _pair_yaw_3d(frame: PoseFrame, a: str, b: str) -> float:
    """Yaw of a body line in the world x-z plane.

    A 2D line is only defined modulo pi: when the shoulders go edge-on the
    projected direction flips, which makes full-revolution counting ambiguous.
    MediaPipe's metric world landmarks do not have that singularity, so they are
    strongly preferred whenever the backend provides them.
    """
    pa = frame.get_world(a)
    pb = frame.get_world(b)
    if pa is None or pb is None:
        return math.nan
    if math.hypot(pb.x - pa.x, pb.z - pa.z) < 0.02:
        return math.nan
    return math.atan2(pb.z - pa.z, pb.x - pa.x)


def lowest_foot_y(frame: PoseFrame) -> float:
    """Image y of the lowest visible foot point (largest y = closest to ice).

    Falls back to ankles if a backend does not provide heel/toe points.
    """
    ys = [
        point.y
        for name in FOOT_KEYPOINTS
        if (point := frame.get(name)) is not None
    ]
    if ys:
        return max(ys)
    ys = [
        point.y
        for name in ("left_ankle", "right_ankle")
        if (point := frame.get(name)) is not None
    ]
    return max(ys) if ys else math.nan


def body_center(frame: PoseFrame) -> tuple[float, float] | None:
    """Approximate centre of mass from the four torso points."""
    points = [
        point
        for name in (LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP)
        if (point := frame.get(name)) is not None
    ]
    if not points:
        return None
    return (
        float(np.mean([p.x for p in points])),
        float(np.mean([p.y for p in points])),
    )


def foot_height_series(
    frames: list[PoseFrame],
    *,
    ground_percentile: float = 90.0,
    baseline_window_s: float = 3.0,
) -> tuple[np.ndarray, float]:
    """Height of the lowest foot above the estimated ice line, in pixels.

    The ice line is a rolling high percentile of the lowest-foot y. A rolling
    baseline absorbs slow camera pans and zooms, and also ignores a lifted foot
    when the planted foot is momentarily undetected. A 3 s window still sees
    plenty of ice time around a 0.5–1 s jump.

    Returns ``(heights, median_baseline_y)``.
    """
    raw = np.array([lowest_foot_y(frame) for frame in frames], dtype=float)
    if np.isfinite(raw).sum() == 0:
        raise ValueError("no foot keypoints found in any frame")
    raw = interpolate_nan(raw)

    times = np.array([frame.timestamp for frame in frames], dtype=float)
    dt = float(np.median(np.diff(times))) if times.size > 1 else 1 / 60
    size = int(round(baseline_window_s / dt)) if dt > 0 else 1
    size = max(3, size | 1)

    if raw.size >= size:
        baseline: np.ndarray | float = percentile_filter(
            raw,
            percentile=ground_percentile,
            size=size,
            mode="nearest",
        )
    else:
        baseline = float(np.percentile(raw, ground_percentile))

    heights = baseline - raw
    median_baseline = float(np.median(baseline))
    return heights, median_baseline


def camera_drift(frames: list[PoseFrame], *, window_s: float = 1.0) -> float:
    """Vertical drift of the ice line over the clip, in pixels.

    A fixed-camera clip has a stable lowest-foot baseline. A large drift means
    the camera panned, zoomed, or the skater moved toward/away from it; the
    height and rotation numbers should not be trusted.
    """
    raw = np.array([lowest_foot_y(frame) for frame in frames], dtype=float)
    if np.isfinite(raw).sum() < 2:
        return 0.0
    raw = interpolate_nan(raw)
    times = np.array([frame.timestamp for frame in frames], dtype=float)
    if times.size < 2:
        return 0.0

    start = times[0]
    end = times[-1]
    if end - start < window_s * 2:
        return 0.0

    levels = []
    t = start
    while t + window_s <= end:
        mask = (times >= t) & (times < t + window_s)
        if mask.any():
            levels.append(float(np.percentile(raw[mask], 90)))
        t += window_s / 2
    if len(levels) < 2:
        return 0.0
    return float(max(levels) - min(levels))


def rotation_signal(frames: list[PoseFrame]) -> tuple[np.ndarray, str]:
    """Unwrapped body rotation angle, preferring metric 3D world landmarks.

    Returns ``(angles, source)``. The zero point is arbitrary; only differences
    matter. ``*-3d`` sources use the x-z yaw of the shoulder or hip line and can
    count full revolutions unambiguously. ``*-2d`` sources fall back to the
    projected line and are only reliable for sub-revolution measurements.
    """
    shoulder_3d = np.array(
        [_pair_yaw_3d(frame, LEFT_SHOULDER, RIGHT_SHOULDER) for frame in frames]
    )
    hip_3d = np.array([_pair_yaw_3d(frame, LEFT_HIP, RIGHT_HIP) for frame in frames])

    shoulder_2d = np.array(
        [_pair_angle(frame, LEFT_SHOULDER, RIGHT_SHOULDER) for frame in frames]
    )
    hip_2d = np.array([_pair_angle(frame, LEFT_HIP, RIGHT_HIP) for frame in frames])

    def coverage(values: np.ndarray) -> float:
        return float(np.isfinite(values).mean()) if values.size else 0.0

    candidates = (
        ("shoulders-3d", shoulder_3d),
        ("hips-3d", hip_3d),
        ("shoulders-2d", shoulder_2d),
        ("hips-2d", hip_2d),
    )
    source, values = max(candidates, key=lambda item: coverage(item[1]))

    if np.isfinite(values).sum() < 2:
        raise ValueError("not enough visible torso keypoints to measure rotation")

    return unwrap_angles(values), source


def body_center_series(frames: list[PoseFrame]) -> np.ndarray:
    """Nx2 array of torso-centre positions with gaps interpolated."""
    points = [body_center(frame) for frame in frames]
    xs = interpolate_nan(np.array([p[0] if p else np.nan for p in points]))
    ys = interpolate_nan(np.array([p[1] if p else np.nan for p in points]))
    return np.column_stack([xs, ys])

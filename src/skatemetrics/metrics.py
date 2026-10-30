"""Jump metrics derived from pose and airborne segmentation."""

from __future__ import annotations

import math

import numpy as np

from .geometry import derivative
from .kinematics import rotation_signal
from .models import JumpMetrics, JumpWindow, PoseFrame
from .segmentation import SegmentationConfig, detect_any_jumps

GRAVITY = 9.80665


def height_from_flight_time(flight_time: float) -> float:
    """Ballistic jump height: h = g * t^2 / 8."""
    return GRAVITY * flight_time**2 / 8.0


def _interp_at(times: np.ndarray, values: np.ndarray, t: float) -> float:
    return float(np.interp(t, times, values))


def analyze_jump(
    frames: list[PoseFrame],
    window: JumpWindow,
    angles: np.ndarray,
    *,
    playback_speed: float = 1.0,
    method: str = "foot",
) -> JumpMetrics:
    if playback_speed <= 0:
        raise ValueError("playback_speed must be positive")

    times = np.array([frame.timestamp for frame in frames], dtype=float)
    # Angular rate in revolutions per second of *video* time. Slow-motion clips
    # are corrected below so coaches get real-world rates.
    rates = derivative(angles, times) / (2.0 * math.pi)

    angle_takeoff = _interp_at(times, angles, window.takeoff_time)
    angle_landing = _interp_at(times, angles, window.landing_time)
    delta = angle_landing - angle_takeoff

    rotations = abs(delta) / (2.0 * math.pi)
    flight_video = window.flight_time
    flight = flight_video * playback_speed
    avg_rate = rotations / flight if flight > 0 else 0.0

    in_air = (times >= window.takeoff_time) & (times <= window.landing_time)
    peak_rates = np.abs(rates[in_air]) / playback_speed
    # Ignore the first/last couple of samples: the derivative is transitioning
    # between the slow entry and the airborne rotation there.
    if peak_rates.size > 6:
        peak_rates = peak_rates[2:-2]
    peak_rate = float(np.max(peak_rates)) if peak_rates.size else 0.0

    profile = [
        (float(t), float(r) / playback_speed)
        for t, r in zip(times[in_air], rates[in_air])
    ]

    return JumpMetrics(
        takeoff_index=window.takeoff_index,
        landing_index=window.landing_index,
        takeoff_time=window.takeoff_time,
        landing_time=window.landing_time,
        flight_time=flight,
        height_m=height_from_flight_time(flight),
        rotations=rotations,
        half_turns=round(rotations * 2.0) / 2.0,
        direction=1 if delta >= 0 else -1,
        avg_rotation_rps=avg_rate,
        peak_rotation_rps=peak_rate,
        playback_speed=playback_speed,
        method=method,
        rotation_profile=profile,
    )


def analyze(
    frames: list[PoseFrame],
    *,
    segmentation: SegmentationConfig | None = None,
    min_rotations: float = 0.4,
    max_height_m: float = 1.1,
    playback_speed: float = 1.0,
) -> list[JumpMetrics]:
    """Detect and measure jumps, dropping physically implausible candidates.

    A figure-skating jump has at least a half rotation, and a 1.1 m jump would
    require a flight time beyond the configured maximum. These filters remove
    most tracking/camera-motion false positives.
    """
    if not frames:
        return []
    angles, _ = rotation_signal(frames)
    windows, method = detect_any_jumps(frames, segmentation)
    jumps = [
        analyze_jump(
            frames,
            window,
            angles,
            playback_speed=playback_speed,
            method=method,
        )
        for window in windows
    ]
    return [
        jump
        for jump in jumps
        if jump.rotations >= min_rotations and jump.height_m <= max_height_m
    ]

"""Detect airborne intervals from the lowest-foot trajectory."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import binary_closing, median_filter

from scipy.signal import savgol_filter

from .geometry import contiguous_runs, smooth, zero_crossing_time
from .kinematics import body_center_series, foot_height_series
from .models import JumpWindow, PoseFrame


@dataclass(frozen=True)
class SegmentationConfig:
    min_flight_s: float = 0.15
    max_flight_s: float = 1.0
    min_gap_s: float = 0.08
    threshold_px: float | None = None
    # Multiplier applied to the auto threshold when deciding if a run is a
    # real jump rather than a foot lift / tracking wobble.
    min_peak_ratio: float = 1.8


def _auto_threshold(heights: np.ndarray) -> float:
    finite = heights[np.isfinite(heights)]
    if finite.size == 0:
        return 6.0
    span = float(np.percentile(finite, 99) - np.percentile(finite, 5))
    return max(4.0, 0.04 * span)


def _parabola_flight(
    times: np.ndarray,
    heights: np.ndarray,
    start: int,
    end: int,
    fallback_takeoff: float,
    fallback_landing: float,
) -> tuple[float, float]:
    """Fit the airborne foot-height arc and solve for the contact times.

    A clean jump is close to ballistic, so the two roots of a quadratic are a
    better contact estimate than a threshold crossing. If the foot tucks (or the
    tracker is noisy) the fit is poor and the threshold estimate is retained.
    """
    idx = np.arange(start, end + 1)
    if idx.size < 6:
        return fallback_takeoff, fallback_landing

    t = times[idx]
    h = heights[idx]
    if np.any(~np.isfinite(h)):
        return fallback_takeoff, fallback_landing

    coeff = np.polyfit(t, h, 2)
    if coeff[0] >= 0:
        return fallback_takeoff, fallback_landing

    fitted = np.polyval(coeff, t)
    ss_res = float(np.sum((h - fitted) ** 2))
    ss_tot = float(np.sum((h - h.mean()) ** 2))
    if ss_tot <= 0 or 1.0 - ss_res / ss_tot < 0.98:
        return fallback_takeoff, fallback_landing

    roots = np.roots(coeff)
    roots = np.sort(roots[np.isreal(roots)].real)
    if roots.size < 2:
        return fallback_takeoff, fallback_landing
    takeoff, landing = float(roots[0]), float(roots[-1])
    if not (
        abs(takeoff - fallback_takeoff) <= 0.10
        and abs(landing - fallback_landing) <= 0.10
        and landing > takeoff
    ):
        return fallback_takeoff, fallback_landing
    return takeoff, landing


def detect_jumps(
    frames: list[PoseFrame],
    config: SegmentationConfig | None = None,
) -> list[JumpWindow]:
    """Find airborne intervals.

    The lowest visible foot is on the ice except during a jump, so its height
    above the ice line is a simple and robust takeoff/landing signal. Boundary
    times are interpolated between frames to avoid 60 fps quantisation.
    """
    config = config or SegmentationConfig()
    if len(frames) < 3:
        return []

    heights, _ = foot_height_series(frames)
    heights = smooth(heights, window=5, polyorder=2)
    times = np.array([frame.timestamp for frame in frames], dtype=float)

    threshold = (
        config.threshold_px
        if config.threshold_px is not None
        else _auto_threshold(heights)
    )

    mask = median_filter((heights > threshold).astype(np.uint8), size=3) > 0
    # Close one- or two-frame dropouts while airborne.
    mask = binary_closing(mask, structure=np.ones(3, dtype=bool))

    runs = contiguous_runs(mask)
    if not runs:
        return []

    # Merge runs separated by less than min_gap_s (e.g. a single bad frame).
    merged: list[list[int]] = []
    for start, end in runs:
        if merged and times[start] - times[merged[-1][1]] <= config.min_gap_s:
            merged[-1][1] = end
        else:
            merged.append([start, end])

    windows: list[JumpWindow] = []
    for start, end in merged:
        flight = times[end] - times[start]
        if flight < config.min_flight_s or flight > config.max_flight_s:
            continue
        peak = float(np.max(heights[start : end + 1]))
        if peak < threshold * config.min_peak_ratio:
            continue

        takeoff_t = zero_crossing_time(times, heights, start, threshold, rising=True)
        landing_t = zero_crossing_time(times, heights, end, threshold, rising=False)
        takeoff_t, landing_t = _parabola_flight(
            times,
            heights,
            start,
            end,
            takeoff_t,
            landing_t,
        )
        if landing_t - takeoff_t < config.min_flight_s:
            continue

        windows.append(
            JumpWindow(
                takeoff_index=start,
                landing_index=end,
                takeoff_time=takeoff_t,
                landing_time=landing_t,
            )
        )
    return windows


def detect_com_jumps(
    frames: list[PoseFrame],
    config: SegmentationConfig | None = None,
) -> list[JumpWindow]:
    """Fallback detector using the body-centre excursion above a slow trend.

    A fixed ice line is unavailable when the camera follows the skater. The
    centre of mass still traces a slow trend (camera motion + skating) with a
    short ballistic excursion at the jump, so a high-pass residual isolates it.
    """
    config = config or SegmentationConfig()
    if len(frames) < 8:
        return []

    centers = body_center_series(frames)
    y = centers[:, 1]
    times = np.array([frame.timestamp for frame in frames], dtype=float)
    dt = float(np.median(np.diff(times))) if times.size > 1 else 1 / 60
    window = max(7, int(round(1.5 / dt)) | 1)
    window = min(window, len(y) if len(y) % 2 else len(y) - 1)
    if window <= 3:
        return []

    trend = savgol_filter(y, window_length=window, polyorder=2, mode="interp")
    residual = trend - y

    positive = residual[residual > 0]
    if positive.size == 0:
        return []
    threshold = max(6.0, 0.12 * float(np.max(positive)))

    runs = contiguous_runs(residual > threshold)
    windows: list[JumpWindow] = []
    for start, end in runs:
        peak = float(np.max(residual[start : end + 1]))
        if peak < threshold * 1.4:
            continue

        # Walk out to the zero crossings that bracket the positive lobe.
        a = start
        while a > 0 and residual[a - 1] > 0:
            a -= 1
        b = end
        while b < len(residual) - 1 and residual[b + 1] > 0:
            b += 1

        takeoff_t = float(times[a])
        landing_t = float(times[b])
        if a > 0:
            frac = residual[a - 1] / (residual[a - 1] - residual[a])
            takeoff_t = float(times[a - 1] + frac * (times[a] - times[a - 1]))
        if b < len(residual) - 1:
            frac = residual[b] / (residual[b] - residual[b + 1])
            landing_t = float(times[b] + frac * (times[b + 1] - times[b]))

        flight = landing_t - takeoff_t
        if flight < max(config.min_flight_s, 0.2) or flight > config.max_flight_s:
            continue
        windows.append(
            JumpWindow(
                takeoff_index=a,
                landing_index=b,
                takeoff_time=takeoff_t,
                landing_time=landing_t,
            )
        )
    return windows


def detect_any_jumps(
    frames: list[PoseFrame],
    config: SegmentationConfig | None = None,
) -> tuple[list[JumpWindow], str]:
    """Use the foot baseline when it works, otherwise the CoM fallback."""
    foot = detect_jumps(frames, config)
    if foot:
        return foot, "foot"
    com = detect_com_jumps(frames, config)
    return com, "com" if com else "none"

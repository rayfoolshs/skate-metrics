"""Small numeric helpers used by the jump analysis pipeline."""

from __future__ import annotations

import math

import numpy as np
from scipy.signal import savgol_filter


def line_angle(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Angle of the directed line a -> b, in radians, in image coordinates."""
    return math.atan2(b[1] - a[1], b[0] - a[0])


def interpolate_nan(values: np.ndarray) -> np.ndarray:
    """Linearly interpolate NaNs. Leading/trailing NaNs are edge-extended."""
    values = np.asarray(values, dtype=float)
    if not np.isnan(values).any():
        return values.copy()
    if np.isnan(values).all():
        raise ValueError("cannot interpolate an all-NaN series")
    idx = np.arange(values.size)
    good = ~np.isnan(values)
    return np.interp(idx, idx[good], values[good])


def unwrap_angles(angles: np.ndarray, max_step: float = 2.5) -> np.ndarray:
    """Unwrap a wrapped angle series and repair impossible single-frame jumps.

    Pose models occasionally swap left/right (or lose a limb) for a frame, which
    shows up as a ~pi discontinuity. ``np.unwrap`` treats that as real rotation.
    We remove any step larger than ``max_step`` by offsetting the remainder of
    the series. At 60 fps even a triple Axel is well under 1 rad/frame, so a
    ~pi jump is instrumentation noise rather than motion.
    """
    angles = interpolate_nan(np.asarray(angles, dtype=float))
    unwrapped = np.unwrap(angles)
    if unwrapped.size < 2:
        return unwrapped

    diffs = np.diff(unwrapped)
    for i in np.flatnonzero(np.abs(diffs) > max_step):
        unwrapped[i + 1 :] -= diffs[i]
    return unwrapped


def smooth(values: np.ndarray, window: int = 9, polyorder: int = 3) -> np.ndarray:
    """Savitzky-Golay smoothing that degrades gracefully on short signals."""
    values = np.asarray(values, dtype=float)
    if values.size < 5:
        return values.copy()
    window = min(window if window % 2 else window + 1, values.size)
    if window <= polyorder:
        return values.copy()
    return savgol_filter(values, window_length=window, polyorder=polyorder)


def derivative(values: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Smoothed time derivative, robust to non-uniform frame timestamps."""
    values = np.asarray(values, dtype=float)
    times = np.asarray(times, dtype=float)
    if values.size < 2:
        return np.zeros_like(values)
    rate = np.gradient(values, times)
    return smooth(rate, window=9, polyorder=3)


def percentiles(values: np.ndarray, q: float) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("no finite values")
    return float(np.percentile(finite, q))


def crossing_time(
    times: np.ndarray,
    heights: np.ndarray,
    index: int,
    threshold: float,
    rising: bool,
) -> float:
    """Sub-frame time at which ``heights`` crosses ``threshold``.

    ``index`` is a frame inside the run. We walk outwards until the crossing is
    bracketed, then linearly interpolate. Falls back to the frame timestamp.
    """
    n = len(heights)
    j = index
    if rising:
        while j > 0 and heights[j - 1] > threshold:
            j -= 1
    else:
        while j < n - 1 and heights[j + 1] > threshold:
            j += 1

    if rising:
        if j == 0:
            return float(times[0])
        a, b = j - 1, j
    else:
        if j >= n - 1:
            return float(times[-1])
        a, b = j, j + 1

    ya, yb = heights[a], heights[b]
    if not np.isfinite(ya) or not np.isfinite(yb) or yb == ya:
        return float(times[j])
    frac = (threshold - ya) / (yb - ya)
    frac = min(max(frac, 0.0), 1.0)
    return float(times[a] + frac * (times[b] - times[a]))


def zero_crossing_time(
    times: np.ndarray,
    heights: np.ndarray,
    index: int,
    threshold: float,
    rising: bool,
) -> float:
    """Extrapolate the threshold crossing back to the ice line (height 0).

    Detecting takeoff at a small positive threshold is necessary for noise
    rejection, but it shortens the measured flight by a few milliseconds. The
    near-linear slope through the crossing recovers the true contact time.
    """
    t_cross = crossing_time(times, heights, index, threshold, rising)

    n = len(heights)
    j = index
    if rising:
        while j > 0 and heights[j - 1] > threshold:
            j -= 1
        a, b = max(j - 1, 0), j
    else:
        while j < n - 1 and heights[j + 1] > threshold:
            j += 1
        a, b = j, min(j + 1, n - 1)

    if a == b or not np.isfinite(heights[a]) or not np.isfinite(heights[b]):
        return t_cross
    slope = (heights[b] - heights[a]) / (times[b] - times[a])
    if not np.isfinite(slope) or abs(slope) < 1e-9:
        return t_cross

    t_zero = t_cross - threshold / slope
    # Do not let extrapolation run away from the observed crossing.
    dt = abs(times[b] - times[a])
    return float(min(max(t_zero, t_cross - 2.0 * dt), t_cross + 2.0 * dt))


def contiguous_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Inclusive (start, end) index pairs where a boolean mask is True."""
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return []
    edges = np.diff(np.concatenate(([0], mask.view(np.int8), [0])))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1) - 1
    return list(zip(starts.tolist(), ends.tolist()))

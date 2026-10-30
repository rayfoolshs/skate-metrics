"""Camera-motion compensation for moving/broadcast footage.

A fixed camera is ideal, but YouTube clips and phone footage often pan, tilt or
zoom. This module estimates a background homography and writes a stabilized
clip before pose estimation, so the ice line stays put and jump segmentation
sees a consistent ground plane.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .overlay import _open_writer
from .video import VideoInfo, probe_video

ORB = cv2.ORB_create(nfeatures=1500, fastThreshold=8)
MATCHER = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)


@dataclass
class StabilizationResult:
    path: Path
    applied: bool
    motion_score: float
    segments: int


def _features(gray: np.ndarray):
    return ORB.detectAndCompute(gray, None)


def _step_homography(
    prev_kp,
    prev_des,
    curr_kp,
    curr_des,
) -> tuple[np.ndarray | None, int]:
    if prev_des is None or curr_des is None or len(prev_kp) < 8 or len(curr_kp) < 8:
        return None, 0
    matches = MATCHER.match(prev_des, curr_des)
    if len(matches) < 12:
        return None, len(matches)
    matches = sorted(matches, key=lambda m: m.distance)[:250]
    src = np.float32([prev_kp[m.queryIdx].pt for m in matches])
    dst = np.float32([curr_kp[m.trainIdx].pt for m in matches])
    # Map the current frame back into the previous frame.
    matrix, mask = cv2.findHomography(dst, src, cv2.RANSAC, 4.0)
    if matrix is None or mask is None:
        return None, 0
    return matrix, int(mask.sum())


def _full_resolution(matrix: np.ndarray, scale: float) -> np.ndarray:
    """Convert a downscaled-coordinate homography to full-resolution pixels."""
    if scale == 1.0:
        return matrix
    down = np.diag([scale, scale, 1.0])
    up = np.diag([1.0 / scale, 1.0 / scale, 1.0])
    return up @ matrix @ down


def motion_score(video_path: str | Path, *, samples: int = 120) -> float:
    """Rough camera-motion magnitude in frame-height units.

    This is intentionally cheap: it samples frames and accumulates the motion of
    the frame centre, plus scale change. It is only used to decide whether the
    full stabilization pass is worth running.
    """
    info = probe_video(video_path)
    cap = cv2.VideoCapture(str(info.path))
    if not cap.isOpened():
        return 0.0
    step = max(1, (info.frame_count or samples) // samples)
    gray_prev = None
    kp_prev = des_prev = None
    cumulative_y = 0.0
    cumulative_x = 0.0
    ys: list[float] = []
    xs: list[float] = []
    scales: list[float] = []
    try:
        index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if index % step == 0:
                small = cv2.resize(frame, None, fx=0.5, fy=0.5)
                gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                kp, des = _features(gray)
                if gray_prev is not None:
                    matrix, inliers = _step_homography(
                        kp_prev, des_prev, kp, des
                    )
                    if matrix is not None and inliers >= 12:
                        centre = np.array(
                            [[[gray.shape[1] / 2, gray.shape[0] / 2]]],
                            dtype=np.float32,
                        )
                        moved = cv2.perspectiveTransform(centre, matrix)[0, 0]
                        cumulative_y += float(moved[1] - centre[0, 0, 1])
                        cumulative_x += float(moved[0] - centre[0, 0, 0])
                        ys.append(cumulative_y)
                        xs.append(cumulative_x)
                        scales.append(float(np.hypot(matrix[0, 0], matrix[1, 0])))
                gray_prev, kp_prev, des_prev = gray, kp, des
            index += 1
    finally:
        cap.release()

    if not ys:
        return 0.0
    vertical = (max(ys) - min(ys)) / max(info.height, 1)
    horizontal = (max(xs) - min(xs)) / max(info.width, 1)
    scale = (max(scales) - min(scales)) if scales else 0.0
    return float(max(vertical, horizontal, scale))


def should_stabilize(video_path: str | Path, *, threshold: float = 0.03) -> bool:
    return motion_score(video_path) > threshold


def stabilize_video(
    video_path: str | Path,
    output_path: str | Path,
    *,
    progress: Callable[[str], None] | None = None,
    proc_scale: float = 0.5,
) -> StabilizationResult:
    """Estimate and remove global camera motion.

    Frames are warped into the coordinate system of an anchor frame. When
    matching confidence collapses (a cut, a title card, a whip pan), the anchor
    resets and the segment counter increments.
    """
    info = probe_video(video_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(info.path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {info.path}")

    writer = _open_writer(
        output_path,
        info.fps if info.fps else 60.0,
        (info.width, info.height),
    )

    anchor_gray = None
    anchor_kp = anchor_des = None
    prev_kp = prev_des = None
    accumulated = np.eye(3, dtype=np.float64)
    segments = 1
    total = info.frame_count or 0
    processed = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            small = cv2.resize(frame, None, fx=proc_scale, fy=proc_scale)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            kp, des = _features(gray)

            matrix = None
            inliers = 0
            if prev_des is not None:
                matrix, inliers = _step_homography(
                    prev_kp, prev_des, kp, des
                )

            if matrix is not None and inliers >= 18:
                accumulated = accumulated @ matrix
            elif anchor_kp is not None:
                # Try a direct match to the anchor before declaring a cut.
                matrix, inliers = _step_homography(
                    anchor_kp, anchor_des, kp, des
                )
                if matrix is not None and inliers >= 18:
                    accumulated = matrix
                else:
                    segments += 1
                    accumulated = np.eye(3, dtype=np.float64)
                    anchor_gray, anchor_kp, anchor_des = gray, kp, des
            else:
                accumulated = np.eye(3, dtype=np.float64)

            if anchor_gray is None:
                anchor_gray, anchor_kp, anchor_des = gray, kp, des

            warped = cv2.warpPerspective(
                frame,
                _full_resolution(accumulated, proc_scale),
                (info.width, info.height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )
            writer.write(warped)
            prev_kp, prev_des = kp, des
            processed += 1
            if progress is not None and processed % 30 == 0:
                progress(f"Stabilizing camera ({processed}/{total or '?'} frames)")
    finally:
        cap.release()
        writer.release()

    return StabilizationResult(
        path=output_path,
        applied=True,
        motion_score=motion_score(video_path),
        segments=segments,
    )

"""Render skeleton, trajectories and metrics onto the source video."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .kinematics import body_center
from .models import SKELETON_CONNECTIONS, JumpMetrics, PoseFrame
from .video import VideoInfo, iter_frames

TEAL = (196, 214, 0)
AMBER = (0, 176, 255)
WHITE = (245, 245, 245)
MUTED = (185, 185, 185)
PANEL = (28, 28, 28)

_SKELETON_JOINTS = {
    name for connection in SKELETON_CONNECTIONS for name in connection
} | {"head", "nose"}


def _put(
    frame: np.ndarray,
    text: str,
    org: tuple[int, int],
    scale: float,
    color: tuple[int, int, int] = WHITE,
    thickness: int = 2,
) -> None:
    cv2.putText(
        frame,
        text,
        (int(org[0]), int(org[1])),
        cv2.FONT_HERSHEY_DUPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def _panel(
    frame: np.ndarray,
    top_left: tuple[int, int],
    bottom_right: tuple[int, int],
    alpha: float = 0.72,
) -> None:
    overlay = frame.copy()
    cv2.rectangle(overlay, top_left, bottom_right, PANEL, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def _draw_skeleton(frame: np.ndarray, pose: PoseFrame, scale: float) -> None:
    for a, b in SKELETON_CONNECTIONS:
        pa, pb = pose.get(a), pose.get(b)
        if pa is None or pb is None:
            continue
        cv2.line(
            frame,
            (int(pa.x), int(pa.y)),
            (int(pb.x), int(pb.y)),
            TEAL,
            max(2, int(2 * scale)),
            cv2.LINE_AA,
        )

    left, right = pose.get("left_shoulder"), pose.get("right_shoulder")
    left_hip, right_hip = pose.get("left_hip"), pose.get("right_hip")
    if left and right and left_hip and right_hip:
        shoulder_mid = ((left.x + right.x) / 2, (left.y + right.y) / 2)
        hip_mid = ((left_hip.x + right_hip.x) / 2, (left_hip.y + right_hip.y) / 2)
        cv2.line(
            frame,
            (int(shoulder_mid[0]), int(shoulder_mid[1])),
            (int(hip_mid[0]), int(hip_mid[1])),
            (120, 140, 145),
            max(1, int(1.5 * scale)),
            cv2.LINE_AA,
        )

        head = pose.get("head") or pose.get("nose")
        if head is not None:
            shoulder_width = abs(right.x - left.x)
            radius = max(int(9 * scale), int(shoulder_width * 0.34))
            cv2.line(
                frame,
                (int(shoulder_mid[0]), int(shoulder_mid[1])),
                (int(head.x), int(head.y + radius * 0.7)),
                TEAL,
                max(2, int(2 * scale)),
                cv2.LINE_AA,
            )
            cv2.circle(
                frame,
                (int(head.x), int(head.y)),
                radius,
                (48, 54, 62),
                -1,
                cv2.LINE_AA,
            )
            cv2.circle(
                frame,
                (int(head.x), int(head.y)),
                radius,
                AMBER,
                max(1, int(1.5 * scale)),
                cv2.LINE_AA,
            )

    for name, point in pose.keypoints.items():
        if name in _SKELETON_JOINTS and point.visible:
            cv2.circle(
                frame,
                (int(point.x), int(point.y)),
                max(2, int(3 * scale)),
                AMBER,
                -1,
                cv2.LINE_AA,
            )


def _draw_rotation_graph(
    frame: np.ndarray,
    metrics: JumpMetrics,
    width: int,
    height: int,
    scale: float,
    now: float,
) -> None:
    if len(metrics.rotation_profile) < 2:
        return
    graph_w = int(300 * scale)
    graph_h = int(110 * scale)
    margin = int(18 * scale)
    x0, y0 = width - graph_w - margin, height - graph_h - margin
    _panel(frame, (x0 - 12, y0 - 30), (x0 + graph_w + 12, y0 + graph_h + 12), 0.7)
    _put(frame, "ROTATION SPEED (rev/s)", (x0, y0 - 10), 0.4 * scale, MUTED, 1)

    times = np.array([t for t, _ in metrics.rotation_profile])
    rates = np.abs(np.array([r for _, r in metrics.rotation_profile]))
    peak = max(float(np.max(rates)), 1e-6)
    span = max(times[-1] - times[0], 1e-6)

    def to_px(t: float, r: float) -> tuple[int, int]:
        x = x0 + int((t - times[0]) / span * graph_w)
        y = y0 + graph_h - int((r / (peak * 1.10)) * graph_h)
        return x, y

    cv2.line(
        frame,
        (x0, y0 + graph_h),
        (x0 + graph_w, y0 + graph_h),
        (90, 90, 90),
        1,
    )
    pts = np.array([to_px(t, r) for t, r in zip(times, rates)], dtype=np.int32)
    cv2.polylines(frame, [pts], False, AMBER, max(2, int(2 * scale)), cv2.LINE_AA)
    if times[0] <= now <= times[-1]:
        cursor = x0 + int((now - times[0]) / span * graph_w)
        cv2.line(frame, (cursor, y0), (cursor, y0 + graph_h), WHITE, 1)


def _draw_metrics_card(
    frame: np.ndarray,
    metrics: JumpMetrics,
    jump_number: int,
    width: int,
    height: int,
    scale: float,
) -> None:
    x, y = int(18 * scale), int(24 * scale)
    w, h = int(320 * scale), int(206 * scale)
    _panel(frame, (x - 10, y - 18), (x + w, y + h), 0.76)
    _put(frame, f"JUMP {jump_number}", (x, y + 14 * scale), 0.55 * scale, AMBER)
    lines = [
        (f"HEIGHT      {metrics.height_m:.2f} m", WHITE),
        (f"ROTATION    {metrics.half_turns:.1f} turns", WHITE),
        (f"AVG RATE    {metrics.avg_rotation_rps:.2f} rev/s", WHITE),
        (f"PEAK RATE   {metrics.peak_rotation_rps:.2f} rev/s", MUTED),
        (f"FLIGHT      {metrics.flight_time:.2f} s", MUTED),
    ]
    for i, (text, color) in enumerate(lines):
        _put(
            frame,
            text,
            (x, y + int((56 + i * 27) * scale)),
            0.5 * scale,
            color,
            1,
        )


def _draw_timeline(
    frame: np.ndarray,
    metrics: list[JumpMetrics],
    duration: float,
    width: int,
    height: int,
    scale: float,
    now: float,
) -> None:
    if not metrics or duration <= 0:
        return
    margin = int(24 * scale)
    y = height - int(14 * scale)
    x0, x1 = margin, width - margin
    cv2.line(frame, (x0, y), (x1, y), (80, 80, 80), max(2, int(2 * scale)))
    for m in metrics:
        a = x0 + int(m.takeoff_time / duration * (x1 - x0))
        b = x0 + int(m.landing_time / duration * (x1 - x0))
        cv2.line(frame, (a, y), (b, y), AMBER, max(4, int(4 * scale)))
    cursor = x0 + int(min(max(now / duration, 0.0), 1.0) * (x1 - x0))
    cv2.circle(frame, (cursor, y), max(4, int(5 * scale)), WHITE, -1, cv2.LINE_AA)


def _open_writer(path: Path, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    """Prefer H.264 so Safari/iOS can play the result in a <video> element."""
    for codec in ("avc1", "H264", "mp4v"):
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*codec),
            fps,
            size,
        )
        if writer.isOpened():
            return writer
        writer.release()
    raise RuntimeError(f"could not open video writer for {path}")


def _skeleton_background(width: int, height: int) -> np.ndarray:
    """Dark studio backdrop for the skeleton-only export."""
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        shade = 18 + int(16 * (1.0 - y / max(height, 1)))
        canvas[y, :, 0] = shade
        canvas[y, :, 1] = shade
        canvas[y, :, 2] = shade
    step = max(40, width // 24)
    for x in range(0, width, step):
        cv2.line(canvas, (x, 0), (x, height), (30, 30, 30), 1)
    for y in range(0, height, step):
        cv2.line(canvas, (0, y), (width, y), (30, 30, 30), 1)
    return canvas


def render_overlay(
    info: VideoInfo,
    poses: list[PoseFrame],
    jumps: list[JumpMetrics],
    output_path: str | Path,
    *,
    start: float | None = None,
    end: float | None = None,
    skeleton_path: str | Path | None = None,
) -> tuple[Path, Path | None]:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    by_index = {pose.index: pose for pose in poses}
    writer = _open_writer(
        output_path,
        info.fps if info.fps else 60.0,
        (info.width, info.height),
    )

    skeleton_writer: cv2.VideoWriter | None = None
    skeleton_output: Path | None = None
    background: np.ndarray | None = None
    if skeleton_path is not None:
        skeleton_output = Path(skeleton_path)
        skeleton_output.parent.mkdir(parents=True, exist_ok=True)
        skeleton_writer = _open_writer(
            skeleton_output,
            info.fps if info.fps else 60.0,
            (info.width, info.height),
        )
        background = _skeleton_background(info.width, info.height)

    scale = max(info.width / 1280.0, 0.6)
    trail: list[tuple[int, int]] = []
    max_trail = int(45 * scale)

    try:
        for index, timestamp, frame in iter_frames(info, start=start, end=end):
            pose = by_index.get(index)
            if pose is not None:
                _draw_skeleton(frame, pose, scale)
                center = body_center(pose)
                if center is not None:
                    trail.append((int(center[0]), int(center[1])))
                    if len(trail) > max_trail:
                        trail.pop(0)
            if len(trail) > 1:
                cv2.polylines(
                    frame,
                    [np.array(trail, dtype=np.int32)],
                    False,
                    (80, 200, 255),
                    max(1, int(1.5 * scale)),
                    cv2.LINE_AA,
                )

            active = [
                (i, m)
                for i, m in enumerate(jumps)
                if timestamp >= m.takeoff_time
            ]
            if active:
                jump_number, current = active[-1]
                _draw_metrics_card(
                    frame, current, jump_number + 1, info.width, info.height, scale
                )
                _draw_rotation_graph(
                    frame, current, info.width, info.height, scale, timestamp
                )

            _draw_timeline(
                frame,
                jumps,
                info.duration,
                info.width,
                info.height,
                scale,
                timestamp,
            )
            writer.write(frame)

            if skeleton_writer is not None and background is not None:
                skeleton = background.copy()
                if pose is not None:
                    _draw_skeleton(skeleton, pose, scale)
                if len(trail) > 1:
                    cv2.polylines(
                        skeleton,
                        [np.array(trail, dtype=np.int32)],
                        False,
                        (80, 200, 255),
                        max(1, int(1.5 * scale)),
                        cv2.LINE_AA,
                    )
                _put(
                    skeleton,
                    "SKELETON VIEW",
                    (info.width - int(190 * scale), int(34 * scale)),
                    0.55 * scale,
                    MUTED,
                    1,
                )
                if active:
                    jump_number, current = active[-1]
                    _draw_metrics_card(
                        skeleton,
                        current,
                        jump_number + 1,
                        info.width,
                        info.height,
                        scale,
                    )
                    _draw_rotation_graph(
                        skeleton, current, info.width, info.height, scale, timestamp
                    )
                _draw_timeline(
                    skeleton,
                    jumps,
                    info.duration,
                    info.width,
                    info.height,
                    scale,
                    timestamp,
                )
                skeleton_writer.write(skeleton)
    finally:
        writer.release()
        if skeleton_writer is not None:
            skeleton_writer.release()
    return output_path, skeleton_output

"""Deterministic synthetic skater used for tests and the ``demo`` command.

It is deliberately not a rendering of a real person. It gives the pipeline a
known answer — 0.60 s flight, 1.5 revolutions — so segmentation, rotation
unwrapping, overlay and report can be exercised without a video or a pose model.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np

from .metrics import GRAVITY

WIDTH, HEIGHT = 1280, 720
GROUND_Y = 620
PPM = 250.0  # pixels per metre
CENTER_X = 640.0

BODY = {
    "head": (0.0, 1.62, 0.0),
    "neck": (0.0, 1.48, 0.0),
    "left_shoulder": (-0.22, 1.42, 0.0),
    "right_shoulder": (0.22, 1.42, 0.0),
    "left_elbow": (-0.30, 1.16, 0.0),
    "right_elbow": (0.30, 1.16, 0.0),
    "left_wrist": (-0.26, 0.92, 0.0),
    "right_wrist": (0.26, 0.92, 0.0),
    "left_hip": (-0.13, 1.00, 0.0),
    "right_hip": (0.13, 1.00, 0.0),
    "left_knee": (-0.12, 0.55, 0.0),
    "right_knee": (0.12, 0.55, 0.0),
    "left_ankle": (-0.12, 0.08, 0.0),
    "right_ankle": (0.12, 0.08, 0.0),
    "left_heel": (-0.12, 0.04, -0.08),
    "right_heel": (0.12, 0.04, -0.08),
    "left_foot_index": (-0.12, 0.04, 0.16),
    "right_foot_index": (0.12, 0.04, 0.16),
}

BONES = (
    ("head", "neck"),
    ("neck", "left_shoulder"),
    ("neck", "right_shoulder"),
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("right_shoulder", "right_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("right_hip", "right_knee"),
    ("left_knee", "left_ankle"),
    ("right_knee", "right_ankle"),
    ("left_ankle", "left_heel"),
    ("right_ankle", "right_heel"),
    ("left_heel", "left_foot_index"),
    ("right_heel", "right_foot_index"),
)


def _jump(t: float, takeoff: float, flight: float) -> float:
    if t <= takeoff or t >= takeoff + flight:
        return 0.0
    u = (t - takeoff) / flight
    return 4.0 * (GRAVITY * flight**2 / 8.0) * u * (1.0 - u)


def _yaw(t: float, takeoff: float, flight: float, turns: float) -> float:
    if t < takeoff:
        # A little pre-rotation on the ice, as in a real Axel entry.
        return -0.25 * 2.0 * math.pi * (t / takeoff)
    u = min((t - takeoff) / flight, 1.0)
    return -0.25 * 2.0 * math.pi + turns * 2.0 * math.pi * u


def _rotate(x: float, z: float, yaw: float) -> tuple[float, float]:
    c, s = math.cos(yaw), math.sin(yaw)
    return x * c + z * s, -x * s + z * c


def _project(
    x: float, y: float, z: float, jump: float
) -> tuple[int, int]:
    sx = CENTER_X + x * PPM
    sy = GROUND_Y - (y + jump) * PPM + z * PPM * 0.05
    return int(round(sx)), int(round(sy))


def _frame_geometry(t: float, takeoff: float, flight: float, turns: float) -> dict:
    jump = _jump(t, takeoff, flight)
    yaw = _yaw(t, takeoff, flight, turns)

    image: dict[str, dict[str, float]] = {}
    world: dict[str, dict[str, float]] = {}
    for name, (x, y, z) in BODY.items():
        rx, rz = _rotate(x, z, yaw)
        sx, sy = _project(rx, y, rz, jump)
        image[name] = {"x": float(sx), "y": float(sy), "confidence": 0.99}
        world[name] = {"x": rx, "y": -y, "z": rz, "confidence": 0.99}
    return {"image": image, "world": world, "jump": jump, "yaw": yaw}


def _draw_ice(frame: np.ndarray) -> None:
    frame[:] = (232, 238, 240)
    # Cold gradient toward the boards.
    for y in range(HEIGHT):
        shade = int(232 - 26 * (y / HEIGHT))
        frame[y, :, 0] = shade - 4
        frame[y, :, 1] = shade
        frame[y, :, 2] = shade
    cv2.line(frame, (0, GROUND_Y), (WIDTH, GROUND_Y), (170, 180, 184), 2)
    for x in range(80, WIDTH, 160):
        cv2.line(frame, (x, GROUND_Y), (x - 40, HEIGHT), (205, 212, 215), 1)
    cv2.putText(
        frame,
        "SYNTHETIC DEMO",
        (WIDTH // 2 - 92, 44),
        cv2.FONT_HERSHEY_DUPLEX,
        0.6,
        (120, 130, 135),
        1,
        cv2.LINE_AA,
    )


def _draw_skater(frame: np.ndarray, geo: dict[str, dict], jump: float) -> None:
    points = geo
    shadow_scale = max(0.35, 1.0 - jump * 1.6)
    cv2.ellipse(
        frame,
        (int(CENTER_X), GROUND_Y + 4),
        (int(58 * shadow_scale), int(11 * shadow_scale)),
        0,
        0,
        360,
        (178, 188, 192),
        -1,
        cv2.LINE_AA,
    )
    for a, b in BONES:
        pa, pb = points[a], points[b]
        cv2.line(
            frame,
            (int(pa["x"]), int(pa["y"])),
            (int(pb["x"]), int(pb["y"])),
            (48, 54, 62),
            5,
            cv2.LINE_AA,
        )
    head = points["head"]
    cv2.circle(
        frame,
        (int(head["x"]), int(head["y"])),
        26,
        (48, 54, 62),
        -1,
        cv2.LINE_AA,
    )
    for name in ("left_ankle", "right_ankle", "left_heel", "right_heel"):
        p = points[name]
        cv2.circle(frame, (int(p["x"]), int(p["y"])), 5, (220, 90, 70), -1, cv2.LINE_AA)


def write_demo(
    output_dir: str | Path,
    *,
    duration: float = 4.0,
    fps: int = 60,
    takeoff: float = 1.4,
    flight: float = 0.60,
    turns: float = 1.5,
) -> tuple[Path, Path]:
    """Write a synthetic video plus matching keypoints. Returns both paths."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / "synthetic.mp4"
    keypoints_path = output_dir / "keypoints.json"

    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (WIDTH, HEIGHT),
    )
    if not writer.isOpened():
        raise RuntimeError(f"could not open video writer for {video_path}")

    frames = []
    frame_count = int(round(duration * fps))
    try:
        for index in range(frame_count):
            t = index / fps
            geo = _frame_geometry(t, takeoff, flight, turns)
            frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
            _draw_ice(frame)
            _draw_skater(frame, geo["image"], geo["jump"])
            cv2.putText(
                frame,
                f"t={t:0.2f}s",
                (WIDTH - 150, 44),
                cv2.FONT_HERSHEY_DUPLEX,
                0.55,
                (110, 120, 125),
                1,
                cv2.LINE_AA,
            )
            writer.write(frame)
            frames.append(
                {
                    "index": index,
                    "timestamp": t,
                    "keypoints": geo["image"],
                    "world_keypoints": geo["world"],
                }
            )
    finally:
        writer.release()

    keypoints_path.write_text(json.dumps({"frames": frames}))
    return video_path, keypoints_path

"""Video metadata and frame iteration with presentation timestamps.

Phone slow-motion and HEVC files are frequently variable-frame-rate. Using
``index / fps`` silently corrupts flight-time measurements, so timestamps come
from ffprobe whenever it is available.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


@dataclass
class VideoInfo:
    path: Path
    width: int
    height: int
    fps: float
    frame_count: int
    duration: float
    timestamps: np.ndarray

    @property
    def has_timestamps(self) -> bool:
        return self.timestamps.size > 0


def _ffprobe_json(path: Path) -> dict:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,duration",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(path),
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def _frame_timestamps(path: Path) -> np.ndarray:
    if shutil.which("ffprobe") is None:
        return np.array([], dtype=float)
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "frame=best_effort_timestamp_time",
        "-of",
        "csv=p=0",
        str(path),
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except (subprocess.CalledProcessError, OSError):
        return np.array([], dtype=float)
    values = []
    for line in out.stdout.splitlines():
        line = line.strip().rstrip(",")
        if not line:
            continue
        try:
            values.append(float(line))
        except ValueError:
            continue
    return np.asarray(values, dtype=float)


def probe_video(path: str | Path) -> VideoInfo:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    raw = _ffprobe_json(path)
    stream = raw["streams"][0]
    rate = stream.get("avg_frame_rate") or stream.get("r_frame_rate") or "60/1"
    if "/" in rate:
        num, den = rate.split("/")
        fps = float(num) / float(den) if float(den) else 60.0
    else:
        fps = float(rate)

    timestamps = _frame_timestamps(path)
    frame_count = int(stream.get("nb_frames") or 0)
    if timestamps.size:
        frame_count = max(frame_count, timestamps.size)
    if frame_count == 0:
        cap = cv2.VideoCapture(str(path))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

    duration = float(stream.get("duration") or raw.get("format", {}).get("duration") or 0)
    if duration <= 0 and frame_count and fps:
        duration = frame_count / fps

    return VideoInfo(
        path=path,
        width=int(stream["width"]),
        height=int(stream["height"]),
        fps=fps,
        frame_count=frame_count,
        duration=duration,
        timestamps=timestamps,
    )


def iter_frames(
    info: VideoInfo,
    *,
    start: float | None = None,
    end: float | None = None,
) -> Iterator[tuple[int, float, np.ndarray]]:
    cap = cv2.VideoCapture(str(info.path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {info.path}")
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)

    index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if index < info.timestamps.size:
                timestamp = float(info.timestamps[index])
            else:
                timestamp = index / info.fps if info.fps else 0.0
            if start is not None and timestamp < start:
                index += 1
                continue
            if end is not None and timestamp > end:
                break
            yield index, timestamp, frame
            index += 1
    finally:
        cap.release()

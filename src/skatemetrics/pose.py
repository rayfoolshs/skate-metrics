"""Pose-estimation backends.

The rest of the pipeline only sees ``list[PoseFrame]``; backends can be swapped
without touching segmentation or metrics. MediaPipe is the default because it
runs well on Apple Silicon without a heavyweight dependency.
"""

from __future__ import annotations

import json
import os
import urllib.request
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .models import Keypoint, Keypoint3D, PoseFrame
from .video import VideoInfo, iter_frames

MODEL_URLS = {
    "lite": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
    "full": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
    "heavy": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task",
}

COCO_KEYPOINTS = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)


class PoseEstimator(ABC):
    @abstractmethod
    def estimate(
        self,
        info: VideoInfo,
        *,
        start: float | None = None,
        end: float | None = None,
    ) -> list[PoseFrame]:
        raise NotImplementedError


class JsonPoseEstimator(PoseEstimator):
    """Load keypoints produced by another tracker or by the synthetic demo."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def estimate(
        self,
        info: VideoInfo,
        *,
        start: float | None = None,
        end: float | None = None,
    ) -> list[PoseFrame]:
        raw = json.loads(self.path.read_text())
        items = raw.get("frames", raw) if isinstance(raw, dict) else raw
        frames: list[PoseFrame] = []
        for item in items:
            timestamp = float(item["timestamp"])
            if start is not None and timestamp < start:
                continue
            if end is not None and timestamp > end:
                continue
            points = {
                name: Keypoint(
                    x=float(point["x"]),
                    y=float(point["y"]),
                    confidence=float(point.get("confidence", 1.0)),
                )
                for name, point in item["keypoints"].items()
            }
            world = {
                name: Keypoint3D(
                    x=float(point["x"]),
                    y=float(point["y"]),
                    z=float(point["z"]),
                    confidence=float(point.get("confidence", 1.0)),
                )
                for name, point in item.get("world_keypoints", {}).items()
            }
            frames.append(
                PoseFrame(
                    index=int(item.get("index", len(frames))),
                    timestamp=timestamp,
                    keypoints=points,
                    world_keypoints=world,
                )
            )
        return frames


class MediaPipePoseEstimator(PoseEstimator):
    """MediaPipe Tasks PoseLandmarker, one person per video."""

    def __init__(
        self,
        *,
        model: str = "lite",
        model_path: str | Path | None = None,
        cache_dir: str | Path | None = None,
    ):
        self.model = model
        self.model_path = Path(model_path) if model_path else None
        self.cache_dir = Path(cache_dir or Path.home() / ".cache" / "skate-metrics")

    def _ensure_model(self) -> Path:
        if self.model_path is not None:
            if not self.model_path.exists():
                raise FileNotFoundError(self.model_path)
            return self.model_path
        if self.model not in MODEL_URLS:
            raise ValueError(f"unknown MediaPipe model: {self.model}")
        path = self.cache_dir / f"pose_landmarker_{self.model}.task"
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(MODEL_URLS[self.model], path)
        return path

    def estimate(
        self,
        info: VideoInfo,
        *,
        start: float | None = None,
        end: float | None = None,
    ) -> list[PoseFrame]:
        os.environ.setdefault("GLOG_minloglevel", "2")
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        import mediapipe as mp
        from mediapipe.tasks.python import vision

        model_path = self._ensure_model()
        options = vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.3,
            min_pose_presence_confidence=0.3,
            min_tracking_confidence=0.3,
        )
        landmarker = vision.PoseLandmarker.create_from_options(options)

        frames: list[PoseFrame] = []
        last_ms = -1
        try:
            for index, timestamp, frame in iter_frames(info, start=start, end=end):
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp_ms = max(int(round(timestamp * 1000)), last_ms + 1)
                last_ms = timestamp_ms
                result = landmarker.detect_for_video(image, timestamp_ms)

                keypoints: dict[str, Keypoint] = {}
                world: dict[str, Keypoint3D] = {}
                if result.pose_landmarks:
                    height, width = frame.shape[:2]
                    for i, landmark in enumerate(result.pose_landmarks[0]):
                        name = vision.PoseLandmark(i).name.lower()
                        keypoints[name] = Keypoint(
                            x=float(landmark.x) * width,
                            y=float(landmark.y) * height,
                            confidence=float(landmark.visibility),
                        )
                    for i, landmark in enumerate(
                        (result.pose_world_landmarks or [[]])[0]
                    ):
                        name = vision.PoseLandmark(i).name.lower()
                        world[name] = Keypoint3D(
                            x=float(landmark.x),
                            y=float(landmark.y),
                            z=float(landmark.z),
                            confidence=float(landmark.visibility),
                        )
                frames.append(
                    PoseFrame(
                        index=index,
                        timestamp=timestamp,
                        keypoints=keypoints,
                        world_keypoints=world,
                    )
                )
        finally:
            landmarker.close()
        return frames


class UltralyticsPoseEstimator(PoseEstimator):
    """YOLO pose backend. Optional: install with ``--extra yolo``."""

    def __init__(self, model: str = "yolo11n-pose.pt"):
        self.model_name = model

    def estimate(
        self,
        info: VideoInfo,
        *,
        start: float | None = None,
        end: float | None = None,
    ) -> list[PoseFrame]:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "ultralytics is not installed; run `uv sync --extra yolo`"
            ) from exc

        model = YOLO(self.model_name)
        frames: list[PoseFrame] = []
        for index, timestamp, frame in iter_frames(info, start=start, end=end):
            result = model.predict(frame, verbose=False, conf=0.25, imgsz=960)[0]
            keypoints: dict[str, Keypoint] = {}
            if result.keypoints is not None and len(result.keypoints) > 0:
                xy = result.keypoints.xy[0].cpu().numpy()
                conf = result.keypoints.conf[0].cpu().numpy()
                for name, (x, y), c in zip(COCO_KEYPOINTS, xy, conf):
                    keypoints[name] = Keypoint(float(x), float(y), float(c))
            frames.append(
                PoseFrame(index=index, timestamp=timestamp, keypoints=keypoints)
            )
        return frames


def get_estimator(backend: str, **kwargs: Any) -> PoseEstimator:
    if backend == "mediapipe":
        return MediaPipePoseEstimator(model=kwargs.get("model", "lite"))
    if backend == "ultralytics":
        return UltralyticsPoseEstimator()
    if backend == "json":
        path = kwargs.get("keypoints_path")
        if path is None:
            raise ValueError("the json backend requires --keypoints")
        return JsonPoseEstimator(path)
    raise ValueError(f"unknown pose backend: {backend}")

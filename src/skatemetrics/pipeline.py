"""End-to-end offline analysis pipeline."""

from __future__ import annotations

import csv
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .geometry import derivative
from .kinematics import camera_drift, rotation_signal
from .metrics import analyze
from .models import JumpMetrics
from .overlay import render_overlay
from .pose import get_estimator
from .report import build_report
from .segmentation import SegmentationConfig
from .stabilize import motion_score, stabilize_video
from .video import VideoInfo, probe_video


@dataclass
class AnalysisResult:
    info: VideoInfo
    jumps: list[JumpMetrics]
    rotation_source: str
    output_dir: Path
    metrics_path: Path
    csv_path: Path
    report_path: Path
    overlay_path: Path | None
    skeleton_path: Path | None
    pose_path: Path | None
    warnings: list[str]
    stabilization: dict | None = None


def _write_metrics(
    path: Path,
    info: VideoInfo,
    jumps: list[JumpMetrics],
    *,
    pose_backend: str,
    rotation_source: str,
    camera_drift_px: float,
    stabilization: dict,
    warnings: list[str],
) -> None:
    payload = {
        "source": str(info.path),
        "fps": round(info.fps, 4),
        "duration": round(info.duration, 4),
        "frame_count": info.frame_count,
        "pose_backend": pose_backend,
        "rotation_source": rotation_source,
        "camera_stability": {
            "drift_px": round(camera_drift_px, 2),
            "drift_fraction": round(camera_drift_px / info.height, 4),
        },
        "stabilization": stabilization,
        "warnings": warnings,
        "jumps": [jump.to_dict() for jump in jumps],
    }
    path.write_text(json.dumps(payload, indent=2))


def _write_pose(path: Path, frames) -> None:
    """Persist pose data so metrics can be re-run without another model pass."""
    payload = {
        "frames": [
            {
                "index": frame.index,
                "timestamp": frame.timestamp,
                "keypoints": {
                    name: {
                        "x": point.x,
                        "y": point.y,
                        "confidence": point.confidence,
                    }
                    for name, point in frame.keypoints.items()
                },
                "world_keypoints": {
                    name: {
                        "x": point.x,
                        "y": point.y,
                        "z": point.z,
                        "confidence": point.confidence,
                    }
                    for name, point in frame.world_keypoints.items()
                },
            }
            for frame in frames
        ]
    }
    path.write_text(json.dumps(payload))


def _write_rotation_csv(
    path: Path,
    frames,
    times: np.ndarray,
    angles: np.ndarray,
) -> None:
    rates = derivative(angles, times) / (2.0 * np.pi)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp", "angle_deg", "rate_rps"])
        for frame, angle, rate in zip(frames, angles, rates):
            writer.writerow(
                [
                    f"{frame.timestamp:.5f}",
                    f"{np.degrees(angle):.3f}",
                    f"{rate:.4f}",
                ]
            )


def analyze_video(
    video_path: str | Path,
    output_dir: str | Path,
    *,
    pose_backend: str = "mediapipe",
    keypoints_path: str | Path | None = None,
    model: str = "lite",
    start: float | None = None,
    end: float | None = None,
    no_overlay: bool = False,
    write_skeleton: bool = False,
    save_pose: bool = False,
    segmentation: SegmentationConfig | None = None,
    playback_speed: float = 1.0,
    stabilize: bool = False,
    progress: Callable[[str], None] | None = None,
) -> AnalysisResult:
    def notify(stage: str) -> None:
        if progress is not None:
            progress(stage)

    if playback_speed <= 0:
        raise ValueError("playback_speed must be positive")
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    notify("Reading video metadata")
    original_name = video_path.name
    info = probe_video(video_path)

    stabilization: dict = {"applied": False}
    stabilization_error: str | None = None
    if stabilize:
        notify("Stabilizing camera")
        try:
            score = motion_score(video_path)
            result = stabilize_video(
                video_path,
                output_dir / "stabilized.mp4",
                progress=notify,
            )
            info = probe_video(result.path)
            stabilization = {
                "applied": True,
                "motion_score": round(score, 4),
                "segments": result.segments,
            }
        except Exception as exc:
            stabilization_error = f"camera stabilization failed: {exc}"

    estimator = get_estimator(
        pose_backend,
        model=model,
        keypoints_path=keypoints_path,
    )
    notify("Detecting pose in every frame")
    frames = estimator.estimate(info, start=start, end=end)
    if not frames:
        raise RuntimeError("no frames were processed")

    notify("Detecting jumps and measuring rotation")
    jumps = analyze(
        frames,
        segmentation=segmentation,
        playback_speed=playback_speed,
    )
    angles, rotation_source = rotation_signal(frames)
    times = np.array([frame.timestamp for frame in frames], dtype=float)

    drift = camera_drift(frames)
    drift_fraction = drift / info.height if info.height else 0.0
    warnings: list[str] = []
    if stabilization_error:
        warnings.append(stabilization_error)
    uses_com = any(jump.method == "com" for jump in jumps)
    if drift_fraction > 0.03:
        message = (
            f"camera moved during the clip (ice line drifted {drift_fraction:.1%} "
            "of frame height); height and rotation may be unreliable. Use a "
            "fixed camera, or select a 1–2 s range around the jump."
        )
        if uses_com:
            message += (
                " A jump was found with the body-centre fallback; timing is "
                "approximate."
            )
        warnings.append(message)
    elif uses_com:
        warnings.append(
            "jump found with the moving-camera fallback (body-centre excursion); "
            "timing is approximate, so validate playback speed and camera stability"
        )
    if abs(playback_speed - 1.0) > 1e-6:
        warnings.append(
            f"playback speed {playback_speed:g} applied; make sure the whole "
            "clip uses the same speed (slow-motion else distorts flight time)"
        )
    if rotation_source.endswith("-2d"):
        warnings.append(
            "2D rotation signal in use; full revolutions can be undercounted "
            "because a projected shoulder line is ambiguous at edge-on poses"
        )
    if not jumps:
        if drift_fraction > 0.03:
            warnings.append(
                "no plausible jumps were detected; this looks like moving-camera "
                "footage, so try selecting a 1–2 s range around the jump"
            )
        else:
            warnings.append("no plausible jumps were detected")

    pose_path = output_dir / "keypoints.json" if save_pose else None
    if pose_path is not None:
        _write_pose(pose_path, frames)

    metrics_path = output_dir / "metrics.json"
    csv_path = output_dir / "rotation.csv"
    _write_metrics(
        metrics_path,
        info,
        jumps,
        pose_backend=pose_backend,
        rotation_source=rotation_source,
        camera_drift_px=drift,
        stabilization=stabilization,
        warnings=warnings,
    )
    _write_rotation_csv(csv_path, frames, times, angles)

    overlay_path = None
    skeleton_path = None
    if not no_overlay:
        notify("Rendering skeleton overlay")
        overlay_path, skeleton_path = render_overlay(
            info,
            frames,
            jumps,
            output_dir / "overlay.mp4",
            start=start,
            end=end,
            skeleton_path=output_dir / "skeleton.mp4" if write_skeleton else None,
        )

    notify("Writing report")
    report_path = build_report(
        output_dir,
        jumps,
        source_name=original_name,
        skeleton_name="skeleton.mp4" if skeleton_path else None,
        fps=info.fps,
        warnings=warnings,
    )

    return AnalysisResult(
        info=info,
        jumps=jumps,
        rotation_source=rotation_source,
        output_dir=output_dir,
        metrics_path=metrics_path,
        csv_path=csv_path,
        report_path=report_path,
        overlay_path=overlay_path,
        skeleton_path=skeleton_path,
        pose_path=pose_path,
        warnings=warnings,
        stabilization=stabilization,
    )

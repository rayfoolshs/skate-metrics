from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np
import pytest

from skatemetrics.geometry import unwrap_angles
from skatemetrics.kinematics import camera_drift
from skatemetrics.metrics import analyze, height_from_flight_time
from skatemetrics.models import Keypoint, Keypoint3D, PoseFrame
from skatemetrics.pipeline import analyze_video
from skatemetrics.report import build_report
from skatemetrics.segmentation import SegmentationConfig, detect_jumps
from skatemetrics.synthetic import _frame_geometry, write_demo
from skatemetrics.validation import match_and_compare


def make_frames(
    *,
    duration: float = 4.0,
    fps: int = 60,
    takeoff: float = 1.4,
    flight: float = 0.60,
    turns: float = 1.5,
) -> list[PoseFrame]:
    frames = []
    for index in range(int(round(duration * fps))):
        timestamp = index / fps
        geo = _frame_geometry(timestamp, takeoff, flight, turns)
        keypoints = {
            name: Keypoint(p["x"], p["y"], p["confidence"])
            for name, p in geo["image"].items()
        }
        world = {
            name: Keypoint3D(p["x"], p["y"], p["z"], p["confidence"])
            for name, p in geo["world"].items()
        }
        frames.append(
            PoseFrame(
                index=index,
                timestamp=timestamp,
                keypoints=keypoints,
                world_keypoints=world,
            )
        )
    return frames


def test_height_from_flight_time():
    assert height_from_flight_time(0.60) == pytest.approx(0.4413, abs=0.001)


def test_unwrap_repairs_pi_instrumentation_jump():
    angles = np.array([0.0, 0.2, 0.4, 0.4 + math.pi, 0.4 + math.pi + 0.2])
    unwrapped = unwrap_angles(angles)
    # The spurious 180-degree step is removed; the remaining motion is smooth.
    assert np.max(np.abs(np.diff(unwrapped))) < 0.3
    assert unwrapped[-1] - unwrapped[0] == pytest.approx(0.6, abs=0.05)


def test_detects_the_synthetic_flight():
    frames = make_frames()
    windows = detect_jumps(frames)
    assert len(windows) == 1
    window = windows[0]
    assert window.takeoff_time == pytest.approx(1.40, abs=0.025)
    assert window.landing_time == pytest.approx(2.00, abs=0.025)
    assert window.flight_time == pytest.approx(0.60, abs=0.03)


def test_synthetic_jump_metrics():
    frames = make_frames()
    jumps = analyze(frames)
    assert len(jumps) == 1
    jump = jumps[0]
    assert jump.height_m == pytest.approx(0.441, abs=0.03)
    assert jump.rotations == pytest.approx(1.5, abs=0.05)
    assert jump.half_turns == pytest.approx(1.5, abs=0.01)
    assert jump.avg_rotation_rps == pytest.approx(2.5, abs=0.15)
    assert 2.2 < jump.peak_rotation_rps < 3.0
    assert len(jump.rotation_profile) > 20


def test_playback_speed_corrects_time_dependent_metrics():
    frames = make_frames(flight=0.60, turns=1.5)
    jump = analyze(frames, playback_speed=0.5)[0]
    assert jump.flight_time == pytest.approx(0.30, abs=0.02)
    assert jump.height_m == pytest.approx(0.110, abs=0.01)
    assert jump.avg_rotation_rps == pytest.approx(5.0, abs=0.3)
    assert jump.rotations == pytest.approx(1.5, abs=0.05)


def test_multiple_rotations_are_counted():
    frames = make_frames(flight=0.65, turns=3.5)
    jumps = analyze(frames, segmentation=SegmentationConfig())
    assert len(jumps) == 1
    assert jumps[0].rotations == pytest.approx(3.5, abs=0.08)
    assert jumps[0].half_turns == pytest.approx(3.5, abs=0.01)


def test_one_foot_on_the_ice_is_not_a_jump():
    frames = make_frames()
    baseline = max(
        p.y
        for p in frames[10].keypoints.values()
        if p.y > 500
    )
    for frame in frames:
        if 1.4 <= frame.timestamp <= 2.0:
            # Keep the left skate planted while the right leg lifts.
            for name in ("left_ankle", "left_heel", "left_foot_index"):
                point = frame.keypoints[name]
                frame.keypoints[name] = Keypoint(point.x, baseline, point.confidence)
    assert detect_jumps(frames) == []


def test_camera_drift_is_detected():
    frames = make_frames()
    for frame in frames:
        shift = 0.6 * frame.timestamp * 60  # ~0.6 px/frame pan
        frame.keypoints = {
            name: Keypoint(p.x, p.y + shift, p.confidence)
            for name, p in frame.keypoints.items()
        }
    assert camera_drift(frames) > 30.0


def test_time_range_analysis(tmp_path):
    video, keypoints = write_demo(tmp_path / "clip", fps=60)
    result = analyze_video(
        video,
        tmp_path / "out",
        pose_backend="json",
        keypoints_path=keypoints,
        start=1.2,
        end=2.2,
        no_overlay=True,
    )
    assert len(result.jumps) == 1
    assert result.jumps[0].takeoff_time >= 1.2
    assert result.jumps[0].landing_time <= 2.2


def test_60fps_matches_240fps_reference(tmp_path):
    low_video, low_keypoints = write_demo(tmp_path / "low", fps=60)
    high_video, high_keypoints = write_demo(tmp_path / "high", fps=240)
    low = analyze_video(
        low_video,
        tmp_path / "low-out",
        pose_backend="json",
        keypoints_path=low_keypoints,
        no_overlay=True,
    )
    high = analyze_video(
        high_video,
        tmp_path / "high-out",
        pose_backend="json",
        keypoints_path=high_keypoints,
        no_overlay=True,
    )
    pairs = match_and_compare(high.jumps, low.jumps)
    assert len(pairs) == 1
    pair = pairs[0]
    assert abs(pair.height_delta_m) < 0.03
    assert abs(pair.flight_delta_s) < 0.02
    assert abs(pair.rotation_delta) < 0.06
    assert abs(pair.avg_rate_delta_rps) < 0.15


def test_report_contains_metrics_and_svg():
    frames = make_frames()
    jump = analyze(frames)[0]
    with tempfile.TemporaryDirectory() as tmp:
        path = build_report(Path(tmp), [jump], source_name="test.mp4")
        text = path.read_text()
        assert "Jump 1" in text
        assert "svg" in text
        assert "overlay.mp4" in text

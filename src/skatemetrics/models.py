"""Core data types shared by the capture, analysis and rendering layers."""

from __future__ import annotations

from dataclasses import dataclass, field

# Names follow MediaPipe / RTMPose body conventions. A backend may provide only
# a subset; the analysis layer is resilient to missing points.
LEFT_SHOULDER = "left_shoulder"
RIGHT_SHOULDER = "right_shoulder"
LEFT_HIP = "left_hip"
RIGHT_HIP = "right_hip"
LEFT_KNEE = "left_knee"
RIGHT_KNEE = "right_knee"
LEFT_ANKLE = "left_ankle"
RIGHT_ANKLE = "right_ankle"
LEFT_HEEL = "left_heel"
RIGHT_HEEL = "right_heel"
LEFT_FOOT_INDEX = "left_foot_index"
RIGHT_FOOT_INDEX = "right_foot_index"

SKELETON_CONNECTIONS: tuple[tuple[str, str], ...] = (
    (LEFT_SHOULDER, RIGHT_SHOULDER),
    (LEFT_SHOULDER, LEFT_HIP),
    (RIGHT_SHOULDER, RIGHT_HIP),
    (LEFT_HIP, RIGHT_HIP),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    (LEFT_HIP, LEFT_KNEE),
    (RIGHT_HIP, RIGHT_KNEE),
    (LEFT_KNEE, LEFT_ANKLE),
    (RIGHT_KNEE, RIGHT_ANKLE),
    (LEFT_ANKLE, LEFT_HEEL),
    (RIGHT_ANKLE, RIGHT_HEEL),
    (LEFT_HEEL, LEFT_FOOT_INDEX),
    (RIGHT_HEEL, RIGHT_FOOT_INDEX),
)

FOOT_KEYPOINTS: tuple[str, ...] = (
    LEFT_ANKLE,
    RIGHT_ANKLE,
    LEFT_HEEL,
    RIGHT_HEEL,
    LEFT_FOOT_INDEX,
    RIGHT_FOOT_INDEX,
)


@dataclass(frozen=True)
class Keypoint:
    """A 2D image-space keypoint in pixels."""

    x: float
    y: float
    confidence: float = 1.0

    @property
    def visible(self) -> bool:
        return self.confidence >= 0.2


@dataclass(frozen=True)
class Keypoint3D:
    """A world-space keypoint, in metres, as produced by MediaPipe."""

    x: float
    y: float
    z: float
    confidence: float = 1.0

    @property
    def visible(self) -> bool:
        return self.confidence >= 0.2


@dataclass
class PoseFrame:
    """Pose for one video frame."""

    index: int
    timestamp: float
    keypoints: dict[str, Keypoint]
    world_keypoints: dict[str, Keypoint3D] = field(default_factory=dict)

    def get(self, name: str) -> Keypoint | None:
        point = self.keypoints.get(name)
        if point is None or not point.visible:
            return None
        return point

    def get_world(self, name: str) -> Keypoint3D | None:
        point = self.world_keypoints.get(name)
        if point is None or not point.visible:
            return None
        return point


@dataclass(frozen=True)
class JumpWindow:
    """A candidate airborne interval, in frame indices and seconds."""

    takeoff_index: int
    landing_index: int
    takeoff_time: float
    landing_time: float

    @property
    def flight_time(self) -> float:
        return self.landing_time - self.takeoff_time


@dataclass
class JumpMetrics:
    """Metrics for a single detected jump."""

    takeoff_index: int
    landing_index: int
    takeoff_time: float
    landing_time: float
    flight_time: float
    height_m: float
    rotations: float
    half_turns: float
    direction: int
    avg_rotation_rps: float
    peak_rotation_rps: float
    playback_speed: float = 1.0
    method: str = "foot"
    # (timestamp, revolutions per second) samples over the airborne window.
    rotation_profile: list[tuple[float, float]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "takeoff_index": self.takeoff_index,
            "landing_index": self.landing_index,
            "takeoff_time": round(self.takeoff_time, 4),
            "landing_time": round(self.landing_time, 4),
            "flight_time": round(self.flight_time, 4),
            "height_m": round(self.height_m, 4),
            "rotations": round(self.rotations, 4),
            "half_turns": round(self.half_turns, 2),
            "direction": "counter-clockwise" if self.direction > 0 else "clockwise",
            "avg_rotation_rps": round(self.avg_rotation_rps, 4),
            "peak_rotation_rps": round(self.peak_rotation_rps, 4),
            "playback_speed": round(self.playback_speed, 4),
            "method": self.method,
            "rotation_profile": [
                [round(t, 4), round(r, 4)] for t, r in self.rotation_profile
            ],
        }

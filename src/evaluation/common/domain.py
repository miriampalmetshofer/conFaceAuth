"""Domain models for evaluation."""
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class SegmentType(Enum):
    GENUINE = "genuine"
    IMPOSTER = "imposter"
    BLACK = "black"


@dataclass
class FrameCounts:
    """Frame state counts."""
    total_frames: int
    unlocked_frames: int
    locked_frames: int


@dataclass
class AuthenticationMetrics:
    """Core authentication performance metrics."""
    true_accept_rate: float
    false_reject_rate: float
    true_reject_rate: float
    false_accept_rate: float
    equal_error_rate: float
    imposter_lockout_time: Optional[float]
    counts: FrameCounts


@dataclass
class VideoMetadata:
    """Metadata extracted from video path."""
    video_path: str
    genuine_user: str
    imposter_user: str
    scenario: Optional[str] = None


@dataclass
class FrameData:
    """Single frame data point."""
    frame: int
    predicted_state: str
    distance: float
    risk_score: float
    face_detected: bool
    source_type: str
    participant: str
    device: str
    video_path: str
    segment_type: SegmentType


@dataclass
class EvaluationData:
    """Complete evaluation dataset."""
    frames: list[FrameData]
    threshold: float
    videos: list[VideoMetadata]


@dataclass
class DeviceMetrics:
    """Metrics aggregated by device."""
    device: str
    metrics: AuthenticationMetrics


@dataclass
class ScenarioMetrics:
    """Metrics aggregated by scenario."""
    scenario: str
    metrics: AuthenticationMetrics


@dataclass
class ScenarioDeviceMetrics:
    """Metrics aggregated by scenario and device combination."""
    scenario: str
    device: str
    metrics: AuthenticationMetrics

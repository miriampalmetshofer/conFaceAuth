"""Dataclasses for evaluation metrics and results."""
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class SegmentBoundaries:
    """Frame boundaries for a single segment."""
    start_frame: int
    end_frame: int


@dataclass
class VideoBoundaries:
    """Frame boundaries for video segments."""
    genuine_segment: SegmentBoundaries
    black_segment: SegmentBoundaries
    imposter_segment: SegmentBoundaries

@dataclass
class BaseSegmentMetrics:
    """Base class for segment authentication metrics."""
    total_frames: int
    unlocked_frames: int
    locked_frames: int
    avg_risk_score: float
    avg_distance: float


@dataclass
class GenuineSegmentMetrics(BaseSegmentMetrics):
    """Authentication metrics for genuine user segment."""
    true_accept_rate: float
    false_reject_rate: float


@dataclass
class ImposterSegmentMetrics(BaseSegmentMetrics):
    """Authentication metrics for imposter segment."""
    true_reject_rate: float
    false_accept_rate: float


@dataclass
class OverallMetrics:
    """Overall authentication performance metrics."""
    total_frames: int
    unlocked_frames: int
    locked_frames: int
    unlock_rate: float
    lock_rate: float
    avg_risk_score: float
    median_risk_score: float
    std_risk_score: float
    avg_distance: float
    median_distance: float
    std_distance: float
    state_changes: int
    avg_frames_between_changes: float


@dataclass
class ErrorRates:
    """False accept and false reject rates."""
    false_accept_rate: float
    false_reject_rate: float
    equal_error_rate: float


@dataclass
class VideoMetrics:
    """Metrics for a single video."""
    video_path: str
    total_frames: int
    unlocked_frames: int
    locked_frames: int
    unlock_rate: float
    avg_risk_score: float
    avg_distance: float
    face_detection_rate: float
    genuine_unlock_rate: Optional[float] = None
    imposter_lock_rate: Optional[float] = None
    genuine_user: Optional[str] = None
    imposter_user: Optional[str] = None


@dataclass
class SegmentAnalysis:
    """Complete segment analysis with genuine and imposter metrics."""
    genuine: GenuineSegmentMetrics
    imposter: ImposterSegmentMetrics

    @property
    def error_rates(self) -> ErrorRates:
        """Calculate error rates from genuine and imposter metrics."""
        return ErrorRates(
            false_accept_rate=self.imposter.false_accept_rate,
            false_reject_rate=self.genuine.false_reject_rate,
            equal_error_rate=(self.imposter.false_accept_rate + self.genuine.false_reject_rate) / 2
        )


@dataclass
class GroupedMetrics:
    """Metrics grouped by a dimension (device, scenario, environment, etc.)."""
    dimension_name: str
    groups: Dict[str, SegmentAnalysis]

    def get_group(self, group_name: str) -> Optional[SegmentAnalysis]:
        """Get metrics for a specific group."""
        return self.groups.get(group_name)

    def list_groups(self) -> list[str]:
        """List all group names."""
        return list(self.groups.keys())

"""Domain models for evaluation."""
from dataclasses import dataclass
from enum import Enum
from typing import Optional, ClassVar


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
class MetricDefinition:
    """Metadata for a single metric field."""
    field_name: str
    display_label: str
    short_label: str
    format_spec: str = ".2f"
    include_in_tables: bool = True
    include_in_plots: bool = True
    plot_color: Optional[str] = None
    plot_order: int = 0


@dataclass
class AuthenticationMetrics:
    """Core authentication performance metrics."""
    true_accept_rate: float
    false_reject_rate: float
    true_reject_rate: float
    false_accept_rate: float
    equal_error_rate: float
    imposter_lockout_time: Optional[float]
    max_lockout_time: Optional[float]
    similarity_difference: Optional[float]
    genuine_kickout_rate: Optional[float]
    genuine_kickout_time: Optional[float]
    counts: FrameCounts

    METRIC_DEFINITIONS: ClassVar[list[MetricDefinition]] = [
        MetricDefinition(
            "true_accept_rate",
            "TAR (%)",
            "TAR",
            plot_color='#2ecc71',
            plot_order=0,
            include_in_tables=False
        ),
        MetricDefinition(
            "false_reject_rate",
            "FRR (%)",
            "FRR",
            plot_color='#e74c3c',
            plot_order=1
        ),
        MetricDefinition(
            "true_reject_rate",
            "TRR (%)",
            "TRR",
            plot_color='#3498db',
            plot_order=2,
            include_in_tables=False
        ),
        MetricDefinition(
            "false_accept_rate",
            "FAR (%)",
            "FAR",
            plot_color='#f39c12',
            plot_order=3,
            include_in_tables=False
        ),
        MetricDefinition(
            "equal_error_rate",
            "EER (%)",
            "EER",
            include_in_plots=False,
            include_in_tables=False
        ),
        MetricDefinition(
            "imposter_lockout_time",
            "Lockout (s)",
            "Lockout",
            format_spec=".1f",
            include_in_plots=False
        ),
        MetricDefinition(
            "max_lockout_time",
            "Max Lockout (s)",
            "MaxLock",
            format_spec=".1f",
            include_in_plots=False
        ),
        MetricDefinition(
            "similarity_difference",
            "Sim Δ",
            "SimΔ",
            format_spec=".3f",
            include_in_plots=False,
            include_in_tables=True
        ),
        MetricDefinition(
            "genuine_kickout_rate",
            "Genuine Kickout (%)",
            "GKO%",
            format_spec=".1f",
            include_in_plots=False,
            include_in_tables=True
        ),
        MetricDefinition(
            "genuine_kickout_time",
            "Genuine Kickout Time (s)",
            "GKOTime",
            format_spec=".1f",
            include_in_plots=False,
            include_in_tables=True
        ),
    ]

    def to_formatted_values(self) -> list[str]:
        """Return formatted metric values for all metrics."""
        values = []
        for defn in self.METRIC_DEFINITIONS:
            value = getattr(self, defn.field_name)
            if value is None:
                values.append("N/A")
            else:
                values.append(f"{value:{defn.format_spec}}")
        return values

    def to_table_values(self) -> list[str]:
        """Return formatted metric values for table display only."""
        values = []
        for defn in self.METRIC_DEFINITIONS:
            if not defn.include_in_tables:
                continue
            value = getattr(self, defn.field_name)
            if value is None:
                values.append("N/A")
            else:
                values.append(f"{value:{defn.format_spec}}")
        return values

    @classmethod
    def get_column_labels(cls) -> list[str]:
        """Return display labels for all metrics."""
        return [defn.display_label for defn in cls.METRIC_DEFINITIONS]

    @classmethod
    def get_table_column_labels(cls) -> list[str]:
        """Return display labels for table display only."""
        return [defn.display_label for defn in cls.METRIC_DEFINITIONS if defn.include_in_tables]

    @classmethod
    def get_plot_metrics(cls) -> list[MetricDefinition]:
        """Return metrics to include in plots, sorted by plot_order."""
        plot_metrics = [m for m in cls.METRIC_DEFINITIONS if m.include_in_plots]
        return sorted(plot_metrics, key=lambda m: m.plot_order)

    def get_plot_values(self) -> list[float]:
        """Return values for plotted metrics in correct order."""
        return [getattr(self, m.field_name) for m in self.get_plot_metrics()]


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
    similarity: float
    trust_score: float
    face_detected: bool
    source_type: str
    participant: str
    device: str
    video_path: str
    segment_type: SegmentType
    scenario: Optional[str] = None


@dataclass
class EvaluationData:
    """Complete evaluation dataset."""
    frames: list[FrameData]
    threshold: float
    videos: list[VideoMetadata]
    skip_frames: int
    fps: int


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

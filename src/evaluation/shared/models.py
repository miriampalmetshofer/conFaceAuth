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
    mean_genuine_trust: Optional[float]
    imposter_lockout_count: Optional[int]
    imposter_lockout_total: Optional[int]
    imposter_lockout_time: Optional[float]
    imposter_lockout_time_p90: Optional[float]
    max_lockout_time: Optional[float]
    similarity_difference: Optional[float]
    genuine_kickout_count: Optional[int]
    genuine_kickout_total: Optional[int]
    genuine_kickout_time: Optional[float]
    genuine_kickout_time_p90: Optional[float]
    counts: FrameCounts

    METRIC_DEFINITIONS: ClassVar[list[MetricDefinition]] = [
        MetricDefinition(
            "true_accept_rate",
            "TAR (%)",
            "TAR",
            plot_color='#2ecc71',
            plot_order=0,
            include_in_tables=False,
            include_in_plots=False
        ),
        MetricDefinition(
            "false_reject_rate",
            "FRR (%)",
            "FRR",
            format_spec=".1f",
            plot_color='#e74c3c',
            plot_order=1,
            include_in_tables=True
        ),
        MetricDefinition(
            "true_reject_rate",
            "TRR (%)",
            "TRR",
            plot_color='#3498db',
            plot_order=2,
            include_in_tables=False,
            include_in_plots=False
        ),
        MetricDefinition(
            "false_accept_rate",
            "FAR (%)",
            "FAR",
            format_spec=".1f",
            plot_color='#f39c12',
            plot_order=3,
            include_in_tables=True
        ),
        MetricDefinition(
            "equal_error_rate",
            "EER (%)",
            "EER",
            include_in_plots=False,
            include_in_tables=False
        ),
        MetricDefinition(
            "mean_genuine_trust",
            "Mean Genuine Trust",
            "MGT",
            format_spec=".2f",
            include_in_plots=True,
            include_in_tables=True,
            plot_color='#27ae60',
            plot_order=4
        ),
        MetricDefinition(
            "imposter_lockout_time",
            "Lockout Mean (s)",
            "ILT",
            format_spec=".0f",
            include_in_plots=False
        ),
        MetricDefinition(
            "imposter_lockout_time_p90",
            "Lockout P90 (s)",
            r"ILT\textsubscript{P90}",
            format_spec=".0f",
            include_in_plots=False,
            include_in_tables=False
        ),
        MetricDefinition(
            "max_lockout_time",
            "Max Lockout (s)",
            r"ILT\textsubscript{max}",
            format_spec=".0f",
            include_in_plots=False
        ),
        MetricDefinition(
            "similarity_difference",
            "Sim Δ",
            "SimΔ",
            format_spec=".3f",
            include_in_plots=False,
            include_in_tables=False
        ),
        MetricDefinition(
            "genuine_kickout_time",
            "Genuine Kickout Mean (s)",
            "GKT",
            format_spec=".0f",
            include_in_plots=False,
            include_in_tables=True
        ),
        MetricDefinition(
            "genuine_kickout_time_p90",
            "Genuine Kickout P90 (s)",
            r"GKT\textsubscript{P90}",
            format_spec=".0f",
            include_in_plots=False,
            include_in_tables=False
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

    def print_console(self, indent: str) -> None:
        """Print metrics to console with custom indentation. Respects include_in_tables flag."""
        for defn in self.METRIC_DEFINITIONS:
            if not defn.include_in_tables:
                continue
            value = getattr(self, defn.field_name)
            if value is None:
                print(f"{indent}{defn.display_label:<30} N/A")
                continue
            suffix = ""
            if defn.field_name == "false_reject_rate" and self.genuine_kickout_count:
                suffix = f"  ({self.genuine_kickout_count}/{self.genuine_kickout_total})"
            elif defn.field_name == "false_accept_rate" and self.imposter_lockout_total is not None:
                not_locked = self.imposter_lockout_total - self.imposter_lockout_count
                if not_locked:
                    suffix = f"  ({not_locked}/{self.imposter_lockout_total})"
            elif defn.field_name == "imposter_lockout_time" and self.imposter_lockout_time_p90 is not None:
                suffix = f"  ({self.imposter_lockout_time_p90:.0f})"
            elif defn.field_name == "genuine_kickout_time" and self.genuine_kickout_time_p90 is not None:
                suffix = f"  ({self.genuine_kickout_time_p90:.0f})"
            print(f"{indent}{defn.display_label:<30} {value:{defn.format_spec}}{suffix}")


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

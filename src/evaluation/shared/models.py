"""Domain models for evaluation."""
from dataclasses import dataclass
from enum import Enum
from typing import Optional, ClassVar


class SegmentType(Enum):
    GENUINE = "genuine"
    IMPOSTER = "imposter"
    BLACK = "black"


@dataclass
class TimeStat:
    """Time statistic bundling mean, P90, and optional max."""
    mean: Optional[float]
    p90: Optional[float]
    max: Optional[float]


@dataclass
class SessionCounts:
    """Raw session counts — supporting data for FRR/FAR fraction display only."""
    genuine_sessions: int
    genuine_lockouts: int       # genuine sessions where user was wrongly locked out
    imposter_sessions: int
    imposter_lockouts: int      # imposter sessions where imposter was successfully locked out


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
    """Core authentication performance metrics.

    Reported metrics (appear in tables / plots / paper):
        false_reject_rate, false_accept_rate, mean_genuine_trust,
        imposter_lockout_time, genuine_lockout_time

    Supporting data (not directly reported, used for display fractions):
        session_counts
    """
    false_reject_rate: float
    false_accept_rate: float
    mean_genuine_trust: Optional[float]
    imposter_lockout_time: TimeStat
    genuine_lockout_time: TimeStat
    session_counts: SessionCounts

    METRIC_DEFINITIONS: ClassVar[list[MetricDefinition]] = [
        MetricDefinition(
            "false_reject_rate",
            "FRR (%)",
            "FRR",
            format_spec=".1f",
            plot_color='#e74c3c',
            plot_order=0,
            include_in_tables=True,
            include_in_plots=True
    ),
        MetricDefinition(
            "false_accept_rate",
            "FAR (%)",
            "FAR",
            format_spec=".1f",
            plot_color='#f39c12',
            plot_order=1,
            include_in_tables=True,
            include_in_plots=True
        ),
        MetricDefinition(
            "mean_genuine_trust",
            "Mean Genuine Trust",
            "GT",
            format_spec=".2f",
            include_in_tables=True,
            include_in_plots=True,
            plot_color='#27ae60',
            plot_order=2
        ),
        MetricDefinition(
            "imposter_lockout_time.mean",
            "ULT (s)",
            "ULT",
            format_spec=".0f",
            include_in_tables=True,
            include_in_plots=False
        ),
        MetricDefinition(
            "imposter_lockout_time.p90",
            "ULT P90 (s)",
            "ULT P90",
            format_spec=".0f",
            include_in_tables=True,
            include_in_plots=False
        ),
        MetricDefinition(
            "genuine_lockout_time.mean",
            "GKT (s)",
            "GKT",
            format_spec=".0f",
            include_in_tables=True,
            include_in_plots=False
        ),
    ]

    def _resolve_field(self, field_path: str) -> Optional[float]:
        """Resolve a dot-path field name (e.g. 'imposter_lockout_time.mean')."""
        obj = self
        for attr in field_path.split('.'):
            obj = getattr(obj, attr, None)
            if obj is None:
                return None
        return obj

    def to_table_values(self) -> list[str]:
        """Return formatted metric values for table display only."""
        values = []
        for defn in self.METRIC_DEFINITIONS:
            if not defn.include_in_tables:
                continue
            value = self._resolve_field(defn.field_name)
            if value is None:
                values.append("N/A")
            else:
                values.append(f"{value:{defn.format_spec}}")
        return values

    @classmethod
    def get_table_column_labels(cls) -> list[str]:
        """Return display labels for table columns."""
        return [defn.display_label for defn in cls.METRIC_DEFINITIONS if defn.include_in_tables]

    @classmethod
    def get_plot_metrics(cls) -> list[MetricDefinition]:
        """Return metrics to include in plots, sorted by plot_order."""
        return sorted(
            [m for m in cls.METRIC_DEFINITIONS if m.include_in_plots],
            key=lambda m: m.plot_order
        )

    def get_plot_values(self) -> list[float]:
        """Return values for plotted metrics in correct order."""
        return [self._resolve_field(m.field_name) for m in self.get_plot_metrics()]

    def print_console(self, indent: str) -> None:
        """Print reported metrics to console. Respects include_in_tables flag."""
        for defn in self.METRIC_DEFINITIONS:
            if not defn.include_in_tables:
                continue
            value = self._resolve_field(defn.field_name)
            if value is None:
                print(f"{indent}{defn.display_label:<30} N/A")
                continue
            suffix = ""
            if defn.field_name == "false_reject_rate" and self.session_counts.genuine_lockouts:
                suffix = f"  ({self.session_counts.genuine_lockouts}/{self.session_counts.genuine_sessions})"
            elif defn.field_name == "false_accept_rate":
                not_locked = self.session_counts.imposter_sessions - self.session_counts.imposter_lockouts
                if not_locked:
                    suffix = f"  ({not_locked}/{self.session_counts.imposter_sessions})"
            elif defn.field_name == "genuine_lockout_time.mean" and self.genuine_lockout_time.p90 is not None:
                suffix = f"  ({self.genuine_lockout_time.p90:.0f})"
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

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
            "Lockout Median (s)",
            "LockMed",
            format_spec=".1f",
            include_in_plots=False
        ),
        MetricDefinition(
            "imposter_lockout_time_p90",
            "Lockout P90 (s)",
            "LockP90",
            format_spec=".1f",
            include_in_plots=False,
            include_in_tables=True
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
            "genuine_kickout_time",
            "Genuine Kickout Median (s)",
            "GKOMed",
            format_spec=".1f",
            include_in_plots=False,
            include_in_tables=True
        ),
        MetricDefinition(
            "genuine_kickout_time_p90",
            "Genuine Kickout P90 (s)",
            "GKOP90",
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

    def print_console(self, indent: str) -> None:
        """Print metrics to console with custom indentation."""
        print(f"{indent}TAR (True Accept Rate):        {self.true_accept_rate:6.2f}%")
        print(f"{indent}FRR (False Reject Rate):       {self.false_reject_rate:6.2f}%")
        print(f"{indent}TRR (True Reject Rate):        {self.true_reject_rate:6.2f}%")
        print(f"{indent}FAR (False Accept Rate):       {self.false_accept_rate:6.2f}%")
        print(f"{indent}EER (Equal Error Rate):        {self.equal_error_rate:6.2f}%")

        if self.imposter_lockout_count is not None and self.imposter_lockout_total is not None:
            print(f"{indent}Imposter Lockouts:             {self.imposter_lockout_count}/{self.imposter_lockout_total}")
        else:
            print(f"{indent}Imposter Lockouts:             N/A")

        lockout_median = f"{self.imposter_lockout_time:6.1f}s" if self.imposter_lockout_time is not None else "N/A"
        print(f"{indent}Imposter Lockout Median:       {lockout_median}")

        lockout_p90 = f"{self.imposter_lockout_time_p90:6.1f}s" if self.imposter_lockout_time_p90 is not None else "N/A"
        print(f"{indent}Imposter Lockout P90:          {lockout_p90}")

        max_lockout = f"{self.max_lockout_time:6.1f}s" if self.max_lockout_time is not None else "N/A"
        print(f"{indent}Max Imposter Lockout Time:     {max_lockout}")

        if self.genuine_kickout_count is not None and self.genuine_kickout_total is not None:
            print(f"{indent}Genuine Kickouts:              {self.genuine_kickout_count}/{self.genuine_kickout_total}")
        else:
            print(f"{indent}Genuine Kickouts:              N/A")

        kickout_median = f"{self.genuine_kickout_time:6.1f}s" if self.genuine_kickout_time is not None else "N/A"
        print(f"{indent}Genuine Kickout Median:        {kickout_median}")

        kickout_p90 = f"{self.genuine_kickout_time_p90:6.1f}s" if self.genuine_kickout_time_p90 is not None else "N/A"
        print(f"{indent}Genuine Kickout P90:           {kickout_p90}")


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

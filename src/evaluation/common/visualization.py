"""Visualization utilities."""
from pathlib import Path
import json
from typing import Optional, Callable
from dataclasses import dataclass

import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from evaluation.common.domain import (
    EvaluationData,
    DeviceMetrics,
    ScenarioDeviceMetrics,
    SegmentType,
    FrameData,
    AuthenticationMetrics
)


@dataclass
class TableSpec:
    """Specification for building a metrics table."""
    title: str
    row_label_fn: Callable
    row_label_header: str


class MetricsTableBuilder:
    """Builds matplotlib tables from metrics data."""

    @staticmethod
    def build_table_data(
        items: list,
        spec: TableSpec
    ) -> tuple[list[str], list[list[str]]]:
        """Build table columns and rows from metrics items."""
        columns = [spec.row_label_header] + AuthenticationMetrics.get_column_labels()

        rows = []
        for item in items:
            metrics = item.metrics if hasattr(item, 'metrics') else item
            row = [spec.row_label_fn(item)] + metrics.to_formatted_values()
            rows.append(row)

        return columns, rows

    @staticmethod
    def render_table(
        ax: plt.Axes,
        columns: list[str],
        rows: list[list[str]],
        title: str,
        bbox: Optional[list] = None,
        fontsize: int = 11,
        scale_height: float = 3.0
    ) -> None:
        """Render table on matplotlib axes with standard styling."""
        ax.axis('off')

        table = ax.table(
            cellText=rows,
            colLabels=columns,
            cellLoc='center',
            loc='center',
            bbox=bbox or [0, 0, 1, 1]
        )

        table.auto_set_font_size(False)
        table.set_fontsize(fontsize)
        table.scale(1, scale_height)

        # Header styling
        for i in range(len(columns)):
            table[(0, i)].set_facecolor('#3498db')
            table[(0, i)].set_text_props(weight='bold', color='white')

        # Alternating row colors
        for i in range(len(rows)):
            for j in range(len(columns)):
                if i % 2 == 0:
                    table[(i + 1, j)].set_facecolor('#ecf0f1')

    @staticmethod
    def render_table_with_title(
        ax: plt.Axes,
        columns: list[str],
        rows: list[list[str]],
        title: str,
        bbox: Optional[list] = None,
        fontsize: int = 10,
        scale_height: float = 2.5
    ) -> None:
        """Render table with title text above it (for combined figures)."""
        ax.axis('off')
        ax.text(0.5, 0.95, title, ha='center', fontsize=14, weight='bold', transform=ax.transAxes)

        MetricsTableBuilder.render_table(ax, columns, rows, title, bbox, fontsize, scale_height)


def _load_segment_config(config_path: Optional[Path]) -> Optional[tuple[dict, int]]:
    """Load segment boundaries from config file.

    Returns:
        Tuple of (segments in seconds, fps) or None if config not available
    """
    if not config_path:
        return None

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        config = json.load(f)

    imposter_config = config['imposter_creation']
    fps = imposter_config.get('fps', 30)
    genuine_seconds = imposter_config.get('genuine_user_seconds', 0)
    black_seconds = imposter_config.get('black_screen_seconds', 0)
    imposter_seconds = imposter_config.get('impostor_seconds', 0)

    genuine_end = genuine_seconds
    black_end = genuine_end + black_seconds
    imposter_end = black_end + imposter_seconds

    return {
        'genuine': (0, genuine_end),
        'black': (genuine_end, black_end),
        'imposter': (black_end, imposter_end)
    }, fps


def _add_segment_backgrounds(fig: go.Figure, segments: Optional[dict]) -> None:
    """Add colored background regions and boundary lines for video segments."""
    if not segments:
        return

    colors = {
        'genuine': 'rgba(46, 204, 113, 0.1)',  # Green
        'black': 'rgba(149, 165, 166, 0.1)',     # Gray
        'imposter': 'rgba(231, 76, 60, 0.1)'     # Red
    }

    line_colors = {
        'genuine': 'rgba(46, 204, 113, 0.6)',
        'black': 'rgba(149, 165, 166, 0.6)',
        'imposter': 'rgba(231, 76, 60, 0.6)'
    }

    # Add background rectangles
    for segment_name, (start, end) in segments.items():
        if start >= end:
            continue
        fig.add_vrect(
            x0=start, x1=end,
            fillcolor=colors.get(segment_name, 'rgba(0, 0, 0, 0.05)'),
            layer="below",
            line_width=0
        )

    # Add vertical lines at segment boundaries
    boundaries = []
    for segment_name, (start, end) in segments.items():
        if start not in boundaries and start > 0:
            boundaries.append((start, segment_name))
        if end not in boundaries:
            boundaries.append((end, segment_name))

    for position, segment_name in boundaries:
        fig.add_vline(
            x=position,
            line_color=line_colors.get(segment_name, 'rgba(0, 0, 0, 0.3)'),
            line_width=2,
            line_dash="dash",
            layer="below"
        )


def _create_show_hide_buttons(num_traces: int) -> list[dict]:
    """Create Show All / Hide All buttons for plotly figures."""
    return [
        dict(
            label="Show All",
            method="update",
            args=[{"visible": [True] * num_traces}]
        ),
        dict(
            label="Hide All",
            method="update",
            args=[{"visible": ['legendonly'] * num_traces}]
        )
    ]


def _add_video_traces(fig: go.Figure, video_frames_dict: dict[str, list[FrameData]], fps: int) -> None:
    """Add video traces to a plotly figure.

    Args:
        fig: Plotly figure to add traces to
        video_frames_dict: Dictionary mapping video paths to frame data
        fps: Frames per second for converting frames to seconds (required)
    """
    for video_path, frames in video_frames_dict.items():
        frames_sorted = sorted(frames, key=lambda f: f.frame)

        # Prepare custom data for hover: similarity and face_detected
        customdata = [
            [
                f"{f.similarity:.4f}" if f.similarity is not None else "N/A",
                "Yes" if f.face_detected else "No"
            ]
            for f in frames_sorted
        ]

        fig.add_trace(go.Scatter(
            x=[f.frame / fps for f in frames_sorted],
            y=[f.trust_score for f in frames_sorted],
            mode='lines',
            name=Path(video_path).name,
            customdata=customdata,
            hovertemplate=(
                '<b>%{fullData.name}</b><br>'
                'Time: %{x:.2f}s<br>'
                'Trust: %{y:.4f}<br>'
                'Similarity: %{customdata[0]}<br>'
                'Face: %{customdata[1]}'
                '<extra></extra>'
            ),
            visible=True
        ))


def create_trust_timeline_all_videos(data: EvaluationData, study_name: str, config_path: Optional[Path] = None) -> go.Figure:
    """Create interactive trust score timeline showing all videos.

    Args:
        data: Evaluation data containing frames and videos
        study_name: Name of the study for plot title
        config_path: Path to config file (required for FPS and segment boundaries)

    Raises:
        ValueError: If config_path is not provided
    """
    if not config_path:
        raise ValueError("config_path is required to determine FPS for time axis")

    fig = go.Figure()

    # Load segment config and FPS
    config_result = _load_segment_config(config_path)
    if not config_result:
        raise ValueError(f"Failed to load config from {config_path}")

    segments, fps = config_result
    _add_segment_backgrounds(fig, segments)

    videos = {}
    for frame in data.frames:
        if frame.video_path not in videos:
            videos[frame.video_path] = []
        videos[frame.video_path].append(frame)

    _add_video_traces(fig, videos, fps)

    fig.add_hline(
        y=data.threshold,
        line_dash="dash",
        line_color="black"
    )

    fig.update_layout(
        title=f"{study_name} - Trust Score Timeline (All Videos)<br><sub>Device: All | Threshold: {data.threshold}</sub>",
        xaxis_title="Time (seconds)",
        yaxis_title="Trust Score",
        hovermode='closest',
        height=600,
        margin=dict(t=100),
        template='plotly_white',
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                buttons=_create_show_hide_buttons(len(videos)),
                pad={"r": 10, "t": 10},
                showactive=False,
                x=0.0,
                xanchor="left",
                y=1.08,
                yanchor="top"
            )
        ]
    )

    return fig


def create_trust_timeline_by_device(data: EvaluationData, devices: list[str], study_name: str, config_path: Optional[Path] = None) -> list[go.Figure]:
    if not config_path:
        raise ValueError("config_path is required to determine FPS for time axis")

    figures = []

    config_result = _load_segment_config(config_path)
    if not config_result:
        raise ValueError(f"Failed to load config from {config_path}")

    segments, fps = config_result

    for device in devices:
        device_frames = [f for f in data.frames if f.device == device]

        # Skip devices with no data
        if not device_frames:
            raise ValueError(f"No frames found for device: {device}")

        fig = go.Figure()

        # Add segment backgrounds
        _add_segment_backgrounds(fig, segments)

        videos = {}
        for frame in device_frames:
            if frame.video_path not in videos:
                videos[frame.video_path] = []
            videos[frame.video_path].append(frame)

        _add_video_traces(fig, videos, fps)

        fig.add_hline(
            y=data.threshold,
            line_dash="dash",
            line_color="black"
        )

        fig.update_layout(
            title=f"{study_name} - {device.upper()}<br><sub>Threshold: {data.threshold}</sub>",
            xaxis_title="Time (seconds)",
            yaxis_title="Trust Score",
            hovermode='closest',
            height=600,
            margin=dict(t=100),
            template='plotly_white',
            updatemenus=[
                dict(
                    type="buttons",
                    direction="left",
                    buttons=_create_show_hide_buttons(len(videos)),
                    pad={"r": 10, "t": 10},
                    showactive=False,
                    x=0.0,
                    xanchor="left",
                    y=1.08,
                    yanchor="top"
                )
            ]
        )

        figures.append(fig)

    return figures


def plot_trust_distribution(ax: plt.Axes, data: EvaluationData) -> None:
    """Plot trust score distribution histogram."""
    genuine_frames = [f for f in data.frames if f.segment_type == SegmentType.GENUINE]
    imposter_frames = [f for f in data.frames if f.segment_type == SegmentType.IMPOSTER]

    genuine_trust = [f.trust_score for f in genuine_frames]
    imposter_trust = [f.trust_score for f in imposter_frames]

    ax.hist([genuine_trust, imposter_trust], bins=30, label=['Genuine', 'Imposter'],
            color=['green', 'red'], alpha=0.6, edgecolor='black')
    ax.axvline(data.threshold, color='black', linestyle='--', linewidth=2, label='Threshold')
    ax.set_xlabel('Trust Score')
    ax.set_ylabel('Frequency')
    ax.set_title('Trust Score Distribution')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)


def plot_error_rates_by_device(ax: plt.Axes, device_metrics: list[DeviceMetrics]) -> None:
    """Plot authentication rates bar chart by device."""
    devices = [dm.device for dm in device_metrics]
    plot_metrics = AuthenticationMetrics.get_plot_metrics()

    # Extract all values at once using metadata
    metric_values = [
        [getattr(dm.metrics, metric_def.field_name) for dm in device_metrics]
        for metric_def in plot_metrics
    ]

    # Plot bars using metadata
    x = np.arange(len(devices))
    width = 0.2
    num_metrics = len(plot_metrics)
    offset_start = -(num_metrics - 1) / 2  # Center the bars

    for i, (metric_def, values) in enumerate(zip(plot_metrics, metric_values)):
        offset = offset_start + i
        ax.bar(
            x + offset * width,
            values,
            width,
            label=metric_def.short_label,
            color=metric_def.plot_color,
            alpha=0.8
        )

    ax.set_ylabel('Rate (%)')
    ax.set_xlabel('Device')
    ax.set_title('Authentication Rates by Device')
    ax.set_xticks(x)
    ax.set_xticklabels([d.upper() for d in devices])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 105])


def plot_state_pie_chart(ax: plt.Axes, data: EvaluationData, segment_type: SegmentType, title: str) -> None:
    """Plot pie chart for locked/unlocked proportions."""
    frames = [f for f in data.frames if f.segment_type == segment_type]

    unlocked = sum(1 for f in frames if f.predicted_state == 'Unlocked')
    locked = sum(1 for f in frames if f.predicted_state == 'Locked')

    colors = ['#2ecc71', '#e74c3c']
    ax.pie([unlocked, locked], labels=['Unlocked', 'Locked'], autopct='%1.1f%%',
           colors=colors, startangle=90)
    ax.set_title(title)


def create_summary_visualization(data: EvaluationData, device_metrics: list[DeviceMetrics], title: str) -> plt.Figure:
    """Create comprehensive PNG summary visualization."""
    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, 0])
    plot_trust_distribution(ax1, data)

    ax2 = fig.add_subplot(gs[0, 1:])
    plot_error_rates_by_device(ax2, device_metrics)

    ax3 = fig.add_subplot(gs[1, 0])
    plot_state_pie_chart(ax3, data, SegmentType.GENUINE, "Genuine Segments\n(Should be Unlocked)")

    ax4 = fig.add_subplot(gs[1, 1])
    plot_state_pie_chart(ax4, data, SegmentType.IMPOSTER, "Imposter Segments\n(Should be Locked)")

    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis('off')

    fig.suptitle(title, fontsize=20, y=0.995)

    return fig


def create_device_metrics_table(device_metrics: list[DeviceMetrics]) -> plt.Figure:
    """Create overall device metrics table."""
    fig, ax = plt.subplots(figsize=(16, 4))

    spec = TableSpec(
        title='Overall Device Metrics',
        row_label_fn=lambda dm: dm.device.upper(),
        row_label_header='Device'
    )

    columns, rows = MetricsTableBuilder.build_table_data(device_metrics, spec)
    MetricsTableBuilder.render_table(ax, columns, rows, spec.title)

    fig.suptitle(spec.title, fontsize=16, y=0.95)

    return fig


def create_device_scenario_metrics_table(
    scenario_device_metrics: list[ScenarioDeviceMetrics],
    device: str
) -> plt.Figure:
    """Create scenario breakdown table for a specific device."""
    fig, ax = plt.subplots(figsize=(16, 6))

    filtered_metrics = [sdm for sdm in scenario_device_metrics if sdm.device == device]

    spec = TableSpec(
        title=f'{device.upper()} - Scenario Breakdown',
        row_label_fn=lambda sdm: sdm.scenario.upper(),
        row_label_header='Scenario'
    )

    columns, rows = MetricsTableBuilder.build_table_data(filtered_metrics, spec)
    MetricsTableBuilder.render_table(ax, columns, rows, spec.title)

    fig.suptitle(spec.title, fontsize=16, y=0.96)

    return fig


def _add_distance_columns(
    device_metrics: list[DeviceMetrics],
    columns: list[str],
    rows: list[list[str]],
    frames: list[FrameData]
) -> None:
    """Add average similarity columns to device metrics table."""
    columns.extend(['Avg Genuine Sim', 'Avg Imposter Sim'])

    for i, dm in enumerate(device_metrics):
        device_frames = [f for f in frames if f.device == dm.device]
        genuine_similarities = [f.similarity for f in device_frames if f.segment_type == SegmentType.GENUINE]
        imposter_similarities = [f.similarity for f in device_frames if f.segment_type == SegmentType.IMPOSTER]

        avg_genuine = np.mean(genuine_similarities) if genuine_similarities else 0
        avg_imposter = np.mean(imposter_similarities) if imposter_similarities else 0

        rows[i].extend([f'{avg_genuine:.4f}', f'{avg_imposter:.4f}'])


def _render_device_overview_table(
    ax: plt.Axes,
    device_metrics: list[DeviceMetrics],
    frames: list[FrameData]
) -> None:
    """Render overall device metrics table with distance columns."""
    spec = TableSpec(
        title='Overall Device Metrics',
        row_label_fn=lambda dm: dm.device.upper(),
        row_label_header='Device'
    )
    columns, rows = MetricsTableBuilder.build_table_data(device_metrics, spec)
    _add_distance_columns(device_metrics, columns, rows, frames)
    MetricsTableBuilder.render_table_with_title(
        ax, columns, rows, spec.title,
        bbox=[0, 0, 1, 0.8], fontsize=10, scale_height=2.5
    )


def _render_scenario_breakdown_table(
    ax: plt.Axes,
    scenario_device_metrics: list[ScenarioDeviceMetrics],
    device: str
) -> None:
    """Render scenario breakdown table for a specific device."""
    device_scenario_metrics = [sdm for sdm in scenario_device_metrics if sdm.device == device]

    if not device_scenario_metrics:
        return

    spec = TableSpec(
        title=f'{device.upper()} - Scenario Breakdown',
        row_label_fn=lambda sdm: sdm.scenario.upper(),
        row_label_header='Scenario'
    )
    columns, rows = MetricsTableBuilder.build_table_data(device_scenario_metrics, spec)
    MetricsTableBuilder.render_table_with_title(
        ax, columns, rows, spec.title,
        bbox=[0, 0, 1, 0.8], fontsize=10, scale_height=2.2
    )


def create_combined_metrics_tables(
    device_metrics: list[DeviceMetrics],
    scenario_device_metrics: list[ScenarioDeviceMetrics],
    frames: list[FrameData]
) -> plt.Figure:
    """Create combined figure with device overview and per-device scenario breakdowns."""
    devices_with_scenarios = sorted(set(sdm.device for sdm in scenario_device_metrics))
    num_tables = 1 + len(devices_with_scenarios)

    fig = plt.figure(figsize=(16, 4 + len(devices_with_scenarios) * 5))
    gs = fig.add_gridspec(num_tables, 1, hspace=0.4)

    _render_device_overview_table(fig.add_subplot(gs[0]), device_metrics, frames)

    for i, device in enumerate(devices_with_scenarios, start=1):
        _render_scenario_breakdown_table(fig.add_subplot(gs[i]), scenario_device_metrics, device)

    return fig


def save_html(fig: go.Figure | list[go.Figure], output_path: Path, filename: str) -> Path:
    """Save plotly figure(s) as HTML, stacking multiple figures vertically."""
    output_path.mkdir(parents=True, exist_ok=True)
    filepath = output_path / filename

    if isinstance(fig, list):
        html_content = '<html><head><script src="https://cdn.plot.ly/plotly-latest.min.js"></script></head><body>\n'

        for i, single_fig in enumerate(fig):
            html_content += f'<div id="plot_{i}"></div>\n'
            html_content += f'<script>Plotly.newPlot("plot_{i}", {single_fig.to_json()});</script>\n'

        html_content += '</body></html>'

        filepath.write_text(html_content)
    else:
        fig.write_html(str(filepath))

    return filepath


def save_png(fig: plt.Figure, output_path: Path, filename: str) -> Path:
    """Save matplotlib figure as PNG."""
    output_path.mkdir(parents=True, exist_ok=True)
    filepath = output_path / filename
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return filepath

"""Visualization utilities."""
from pathlib import Path
import json
from typing import Optional, Callable
from dataclasses import dataclass

import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from evaluation.shared.models import (
    EvaluationData,
    DeviceMetrics,
    ScenarioMetrics,
    ScenarioDeviceMetrics,
    SegmentType,
    FrameData,
    AuthenticationMetrics
)
from evaluation.shared.font_config import (
    apply_plotly_font,
    plotly_font_style_tag,
    plotly_html,
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
        columns = [spec.row_label_header] + AuthenticationMetrics.get_table_column_labels()

        rows = []
        for item in items:
            metrics = item.metrics if hasattr(item, 'metrics') else item
            row = [spec.row_label_fn(item)] + metrics.to_table_values()
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


def _load_segment_config(config_path: Path) -> Optional[tuple[dict, int]]:
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
    fps = imposter_config.get('fps')
    genuine_seconds = imposter_config.get('genuine_user_seconds')
    black_seconds = imposter_config.get('black_screen_seconds')
    imposter_seconds = imposter_config.get('impostor_seconds')

    genuine_end = genuine_seconds
    black_end = genuine_end + black_seconds
    imposter_end = black_end + imposter_seconds

    return {
        'genuine': (0, genuine_end),
        'black': (genuine_end, black_end),
        'imposter': (black_end, imposter_end)
    }, fps


def _get_segment_header_annotations(segments: dict) -> list[dict]:
    """Create paper figure segment headers between legend and plot area."""
    if not segments:
        return []

    return [
        dict(
            x=(segments['genuine'][0] + segments['genuine'][1]) / 2,
            y=1.005,
            xref='x',
            yref='paper',
            text='<b>Genuine User</b>',
            showarrow=False,
            xanchor='center',
            yanchor='bottom',
            font=dict(size=20, color='black')
        ),
        dict(
            x=(segments['imposter'][0] + segments['imposter'][1]) / 2,
            y=1.005,
            xref='x',
            yref='paper',
            text='<b>Unauthorized User</b>',
            showarrow=False,
            xanchor='center',
            yanchor='bottom',
            font=dict(size=20, color='black')
        )
    ]


def _add_segment_backgrounds(fig: go.Figure, segments: dict) -> None:
    """Add colored background regions and boundary lines for video segments."""
    if not segments:
        return

    colors = {
        'genuine': 'rgba(46, 204, 113, 0.1)',  # Green
        'black': 'rgba(149, 165, 166, 0.1)',  # Gray
        'imposter': 'rgba(231, 76, 60, 0.1)'  # Red
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


def _add_metrics_visualization(fig: go.Figure, metrics: AuthenticationMetrics,
                               threshold: float, segments: dict) -> None:
    """Add ULT markers on the threshold line for mean and P90 unauthorized-user lockout times.

    Args:
        fig: Plotly figure to add visualizations to
        metrics: Metrics containing unauthorized-user lockout timing
        threshold: Trust score threshold value for positioning markers
        segments: Segment boundaries to calculate ULT absolute position
    """
    if not segments or 'imposter' not in segments:
        return

    imposter_start = segments['imposter'][0]

    if metrics.imposter_lockout_time.mean is not None:
        ult_abs = imposter_start + metrics.imposter_lockout_time.mean
        fig.add_trace(go.Scatter(
            x=[ult_abs],
            y=[threshold],
            mode='markers',
            name='Mean ULT',
            marker=dict(size=15, color='rgba(142, 68, 173, 0.9)', symbol='diamond',
                        line=dict(width=2, color='white')),
            showlegend=True,
            hovertemplate=f'Mean ULT: {metrics.imposter_lockout_time.mean:.1f}s<extra></extra>'
        ))

    if metrics.imposter_lockout_time.p90 is not None:
        p90_abs = imposter_start + metrics.imposter_lockout_time.p90
        fig.add_trace(go.Scatter(
            x=[p90_abs],
            y=[threshold],
            mode='markers',
            name='P90 ULT',
            marker=dict(size=15, color='rgba(231, 76, 60, 0.9)', symbol='diamond',
                        line=dict(width=2, color='white')),
            showlegend=True,
            hovertemplate=f'P90 ULT: {metrics.imposter_lockout_time.p90:.1f}s<extra></extra>'
        ))


def _add_rate_legend_entries(fig: go.Figure, metrics: AuthenticationMetrics) -> None:
    """Add FRR and FAR as text-only entries in the metrics legend."""
    for name, value in [
        ("FRR", metrics.false_reject_rate),
        ("FAR", metrics.false_accept_rate),
    ]:
        fig.add_trace(go.Scatter(
            x=[None],
            y=[None],
            mode='markers',
            name=f'{name}: {value:.1f}%',
            marker=dict(size=0, color='rgba(0, 0, 0, 0)'),
            showlegend=True,
            hoverinfo='skip'
        ))


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


def _group_frames_by_video_path(frames: list[FrameData]) -> dict[str, list[FrameData]]:
    """Group frames by video path."""
    videos = {}
    for frame in frames:
        if frame.video_path not in videos:
            videos[frame.video_path] = []
        videos[frame.video_path].append(frame)
    return videos


def _create_trust_timeline_figure(
        frames: list[FrameData],
        title: str,
        threshold: float,
        segments: dict,
        fps: int
) -> go.Figure:
    """Create a single trust timeline figure from frames.

    Args:
        frames: Frame data to visualize
        title: Figure title
        threshold: Trust threshold value
        segments: Segment boundaries for background coloring
        fps: Frames per second for time axis conversion

    Returns:
        Configured Plotly figure
    """
    fig = go.Figure()

    _add_segment_backgrounds(fig, segments)

    videos = _group_frames_by_video_path(frames)
    _add_video_traces(fig, videos, fps)

    fig.add_hline(
        y=threshold,
        line_dash="dash",
        line_color="black"
    )

    fig.update_layout(
        title=title,
        xaxis_title="Time (seconds)",
        yaxis_title="Trust Score",
        yaxis_range=[0, 1],
        hovermode='closest',
        height=600,
        margin=dict(t=100),
        template='plotly_white',
        font=dict(size=18),
        xaxis=dict(title_font=dict(size=22), tickfont=dict(size=18)),
        yaxis=dict(title_font=dict(size=22), tickfont=dict(size=18)),
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


def create_trust_timeline_all_videos(data: EvaluationData, study_name: str, config_path: Path,
                                     metrics: AuthenticationMetrics) -> go.Figure:
    """Create interactive trust score timeline showing all videos with mean trust and ILT markers.

    Args:
        data: Evaluation data containing frames and videos
        study_name: Name of the study for plot title
        config_path: Path to config file (required for FPS and segment boundaries)
        metrics: Overall authentication metrics for ILT marker positions

    Raises:
        ValueError: If config_path is not provided
    """
    if not config_path:
        raise ValueError("config_path is required to determine FPS for time axis")

    config_result = _load_segment_config(config_path)
    if not config_result:
        raise ValueError(f"Failed to load config from {config_path}")

    segments, fps = config_result

    title = f"{study_name} - Trust Score Timeline (All Videos)<br><sub>Device: All | Threshold: {data.threshold}</sub>"

    fig = _create_trust_timeline_figure(data.frames, title, data.threshold, segments, fps)
    fig.update_layout(title=None)

    # Hide individual video legend entries, remove show/hide buttons, and remove only
    # the threshold hline shape (keep vrect segment backgrounds)
    fig.update_traces(showlegend=False)
    fig.layout.updatemenus = ()
    fig.layout.shapes = [s for s in fig.layout.shapes if not (s.type == 'line' and s.xref == 'paper')]

    # Compute and add mean trust trace across all videos
    videos = _group_frames_by_video_path(data.frames)
    frames_by_time: dict[int, list[float]] = {}
    for frames in videos.values():
        for f in frames:
            if f.frame not in frames_by_time:
                frames_by_time[f.frame] = []
            frames_by_time[f.frame].append(f.trust_score)
    sorted_frames = sorted(frames_by_time.keys())
    mean_times = [frame / fps for frame in sorted_frames]
    mean_trust = [float(np.mean(frames_by_time[frame])) for frame in sorted_frames]

    # Add threshold as a named trace so it appears in the legend
    x_end = mean_times[-1] if mean_times else segments.get('imposter', (0, 0))[1]
    if metrics.mean_genuine_trust is not None:
        genuine_start, genuine_end = segments.get('genuine', (0, x_end))
        fig.add_trace(go.Scatter(
            x=[genuine_start, genuine_end],
            y=[metrics.mean_genuine_trust, metrics.mean_genuine_trust],
            mode='lines',
            name='Mean Genuine Trust',
            line=dict(color='rgba(39, 174, 96, 0.95)', width=2, dash='dot'),
            showlegend=True,
            hovertemplate=f'Mean Genuine Trust: {metrics.mean_genuine_trust:.4f}<extra></extra>'
        ))

    fig.add_trace(go.Scatter(
        x=[0, x_end],
        y=[data.threshold, data.threshold],
        mode='lines',
        name='Threshold',
        line=dict(color='black', width=1, dash='dash'),
        showlegend=True,
        hovertemplate=f'Threshold: {data.threshold}<extra></extra>'
    ))

    # Add invisible traces for segment labels in the legend (next to Threshold)
    if segments:
        segment_labels = {
            'black': 'Black Frames'
        }
        segment_colors = {
            'black': 'rgba(149, 165, 166, 0.3)'
        }

        for segment_name in ['black']:
            if segment_name in segments:
                fig.add_trace(go.Scatter(
                    x=[None],
                    y=[None],
                    mode='markers',
                    name=segment_labels[segment_name],
                    marker=dict(size=10, color=segment_colors[segment_name], symbol='square'),
                    showlegend=True,
                    hoverinfo='skip'
                ))

    _add_metrics_visualization(fig, metrics, data.threshold, segments)
    _add_rate_legend_entries(fig, metrics)

    fig.update_layout(
        annotations=_get_segment_header_annotations(segments),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.085,
            xanchor='right',
            x=1.0,
            font=dict(size=22, family='Latin Modern Roman, serif', color='black')
        ),
        margin=dict(l=80, r=20, t=78, b=60),
        font=dict(size=22, family='Latin Modern Roman, serif', color='black'),
        xaxis=dict(title_font=dict(size=22, family='Latin Modern Roman, serif', color='black'), tickfont=dict(size=18, family='Latin Modern Roman, serif', color='black')),
        yaxis=dict(title_font=dict(size=22, family='Latin Modern Roman, serif', color='black'), tickfont=dict(size=18, family='Latin Modern Roman, serif', color='black'))
    )

    return fig


def _create_timelines_by_attribute(
        data: EvaluationData,
        attribute_values: list[str],
        attribute_name: str,
        filter_fn: Callable[[FrameData, str], bool],
        study_name: str,
        config_path: Path
) -> list[go.Figure]:
    """Generic function to create timelines grouped by a specific attribute.

    Args:
        data: Evaluation data containing frames and videos
        attribute_values: List of attribute values to create timelines for
        attribute_name: Name of the attribute (for error messages)
        filter_fn: Function to filter frames by attribute value
        study_name: Name of the study for plot title
        config_path: Path to config file (required for FPS and segment boundaries)

    Returns:
        List of Plotly figures, one per attribute value

    Raises:
        ValueError: If config_path is not provided or attribute value has no data
    """
    if not config_path:
        raise ValueError("config_path is required to determine FPS for time axis")

    config_result = _load_segment_config(config_path)
    if not config_result:
        raise ValueError(f"Failed to load config from {config_path}")

    segments, fps = config_result
    figures = []

    for value in attribute_values:
        filtered_frames = [f for f in data.frames if filter_fn(f, value)]

        if not filtered_frames:
            raise ValueError(f"No frames found for {attribute_name}: {value}")

        title = f"{study_name} - {value.upper()}<br><sub>Threshold: {data.threshold}</sub>"
        fig = _create_trust_timeline_figure(filtered_frames, title, data.threshold, segments, fps)
        figures.append(fig)

    return figures


def create_trust_timeline_by_device(data: EvaluationData, devices: list[str], study_name: str,
                                    config_path: Optional[Path] = None) -> list[go.Figure]:
    """Create interactive trust score timeline for each device.

    Args:
        data: Evaluation data containing frames and videos
        devices: List of device names to create timelines for
        study_name: Name of the study for plot title
        config_path: Path to config file (required for FPS and segment boundaries)

    Returns:
        List of Plotly figures, one per device

    Raises:
        ValueError: If config_path is not provided or device has no data
    """
    return _create_timelines_by_attribute(
        data, devices, "device",
        lambda f, device: f.device == device,
        study_name, config_path
    )


def create_trust_timeline_by_scenario(data: EvaluationData, scenarios: list[str], study_name: str,
                                       config_path: Optional[Path] = None) -> list[go.Figure]:
    """Create interactive trust score timeline for each scenario (aggregated across all devices).

    Args:
        data: Evaluation data containing frames and videos
        scenarios: List of scenario names to create timelines for
        study_name: Name of the study for plot title
        config_path: Path to config file (required for FPS and segment boundaries)

    Returns:
        List of Plotly figures, one per scenario

    Raises:
        ValueError: If config_path is not provided or scenario has no data
    """
    return _create_timelines_by_attribute(
        data, scenarios, "scenario",
        lambda f, scenario: f.scenario == scenario,
        study_name, config_path
    )


def create_aggregated_trust_timeline_by_scenario(data: EvaluationData, scenarios: list[str], study_name: str,
                                                  config_path: Optional[Path] = None) -> go.Figure:
    """Create single timeline with one aggregated line per scenario.

    Args:
        data: Evaluation data containing frames and videos
        scenarios: List of scenario names to aggregate
        study_name: Name of the study for plot title
        config_path: Path to config file (required for FPS and segment boundaries)

    Returns:
        Single Plotly figure with one line per scenario

    Raises:
        ValueError: If config_path is not provided or scenarios have no data
    """
    if not config_path:
        raise ValueError("config_path is required to determine FPS for time axis")

    config_result = _load_segment_config(config_path)
    if not config_result:
        raise ValueError(f"Failed to load config from {config_path}")

    segments, fps = config_result
    fig = go.Figure()

    _add_segment_backgrounds(fig, segments)

    scenario_colors = {
        'easy': '#2ecc71',
        'angle': '#3498db',
        'lighting': '#f39c12'
    }

    for scenario in scenarios:
        scenario_frames = [f for f in data.frames if f.scenario == scenario]

        if not scenario_frames:
            raise ValueError(f"No frames found for scenario: {scenario}")

        frames_by_time = {}
        for frame in scenario_frames:
            time_in_seconds = frame.frame / fps
            if time_in_seconds not in frames_by_time:
                frames_by_time[time_in_seconds] = []
            frames_by_time[time_in_seconds].append(frame.trust_score)

        times = sorted(frames_by_time.keys())
        avg_trust_scores = [np.mean(frames_by_time[t]) for t in times]

        fig.add_trace(go.Scatter(
            x=times,
            y=avg_trust_scores,
            mode='lines',
            name=scenario.upper(),
            line=dict(
                color=scenario_colors.get(scenario, '#95a5a6'),
                width=3
            ),
            hovertemplate=(
                f'<b>{scenario.upper()}</b><br>'
                'Time: %{x:.2f}s<br>'
                'Avg Trust: %{y:.4f}'
                '<extra></extra>'
            )
        ))

    fig.add_hline(
        y=data.threshold,
        line_dash="dash",
        line_color="black",
        annotation_text=f"Threshold ({data.threshold})",
        annotation_position="right"
    )

    fig.update_layout(
        title=f"{study_name} - Aggregated Trust Score by Scenario<br><sub>Averaged across all devices and videos | Threshold: {data.threshold}</sub>",
        xaxis_title="Time (seconds)",
        yaxis_title="Trust Score",
        yaxis_range=[0, 1],
        hovermode='closest',
        height=600,
        margin=dict(t=100),
        template='plotly_white',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )

    return fig


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

    # Extract all values at once using metadata (dot-path aware)
    metric_values = [
        [dm.metrics._resolve_field(metric_def.field_name) for dm in device_metrics]
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


def create_device_metrics_table(device_metrics: list[DeviceMetrics], frames: list[FrameData]) -> plt.Figure:
    """Create overall device metrics table with similarity columns."""
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


def create_scenario_metrics_table(scenario_metrics: list[ScenarioMetrics]) -> plt.Figure:
    """Create aggregated scenario metrics table (across all devices)."""
    fig, ax = plt.subplots(figsize=(16, 4))

    spec = TableSpec(
        title='Scenario Metrics (Aggregated across Devices)',
        row_label_fn=lambda sm: sm.scenario.upper(),
        row_label_header='Scenario'
    )

    columns, rows = MetricsTableBuilder.build_table_data(scenario_metrics, spec)
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

def _add_scenario_distance_columns(
        scenario_device_metrics: list[ScenarioDeviceMetrics],
        columns: list[str],
        rows: list[list[str]],
        frames: list[FrameData],
        device: str,
        video_metadata: list
) -> None:
    """Add average similarity columns to scenario breakdown table."""
    pass


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
    MetricsTableBuilder.render_table_with_title(
        ax, columns, rows, spec.title,
        bbox=[0, 0, 1, 0.8], fontsize=10, scale_height=2.5
    )


def _render_scenario_breakdown_table(
        ax: plt.Axes,
        scenario_device_metrics: list[ScenarioDeviceMetrics],
        device: str,
        frames: list[FrameData],
        video_metadata: list
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
    _add_scenario_distance_columns(device_scenario_metrics, columns, rows, frames, device, video_metadata)
    MetricsTableBuilder.render_table_with_title(
        ax, columns, rows, spec.title,
        bbox=[0, 0, 1, 0.8], fontsize=10, scale_height=2.2
    )


def create_combined_metrics_tables(
        device_metrics: list[DeviceMetrics],
        scenario_device_metrics: list[ScenarioDeviceMetrics],
        frames: list[FrameData],
        video_metadata: list
) -> plt.Figure:
    """Create combined figure with device overview and per-device scenario breakdowns."""
    devices_with_scenarios = sorted(set(sdm.device for sdm in scenario_device_metrics))
    num_tables = 1 + len(devices_with_scenarios)

    fig = plt.figure(figsize=(16, 4 + len(devices_with_scenarios) * 5))
    gs = fig.add_gridspec(num_tables, 1, hspace=0.4)

    _render_device_overview_table(fig.add_subplot(gs[0]), device_metrics, frames)

    for i, device in enumerate(devices_with_scenarios, start=1):
        _render_scenario_breakdown_table(fig.add_subplot(gs[i]), scenario_device_metrics, device, frames,
                                         video_metadata)

    return fig


def save_html(fig: go.Figure | list[go.Figure], output_path: Path, filename: str) -> Path:
    """Save plotly figure(s) as HTML, stacking multiple figures vertically."""
    output_path.mkdir(parents=True, exist_ok=True)
    filepath = output_path / filename

    if isinstance(fig, list):
        html_content = (
            '<html><head><script src="https://cdn.plot.ly/plotly-latest.min.js"></script>'
            f'{plotly_font_style_tag()}</head><body>\n'
        )

        for i, single_fig in enumerate(fig):
            apply_plotly_font(single_fig)
            html_content += f'<div id="plot_{i}"></div>\n'
            html_content += f'<script>Plotly.newPlot("plot_{i}", {single_fig.to_json()});</script>\n'

        html_content += '</body></html>'

        filepath.write_text(html_content)
    else:
        apply_plotly_font(fig)
        filepath.write_text(plotly_html(fig))

    return filepath


def save_png(fig: plt.Figure, output_path: Path, filename: str) -> Path:
    """Save matplotlib figure as PNG."""
    output_path.mkdir(parents=True, exist_ok=True)
    filepath = output_path / filename
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return filepath


def save_plotly_png(fig: go.Figure, output_path: Path, filename: str, width: int = 1200, height: int = 600) -> Path:
    """Save plotly figure as PNG using kaleido.

    Args:
        fig: Plotly figure to save
        output_path: Directory to save the file
        filename: Name of the output file
        width: Width of the output image in pixels
        height: Height of the output image in pixels

    Returns:
        Path to the saved file
    """
    output_path.mkdir(parents=True, exist_ok=True)
    filepath = output_path / filename
    apply_plotly_font(fig)

    try:
        import plotly.io as pio
        pio.kaleido.scope.mathjax = None
        fig.write_image(
            str(filepath),
            width=width,
            height=height,
            scale=2,
            engine="kaleido"
        )
    except Exception as e:
        print(f"Warning: Could not export PNG with Latin Modern Roman font. Error: {e}")
        print("The font may not be available to Kaleido. Attempting fallback...")
        fig.write_image(str(filepath), width=width, height=height, scale=2)

    return filepath

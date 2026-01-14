"""Visualization utilities."""
from pathlib import Path
import json
from typing import Optional

import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from evaluation.common.domain import (
    EvaluationData,
    DeviceMetrics,
    ScenarioDeviceMetrics,
    SegmentType,
    FrameData
)


def _load_segment_config(config_path: Optional[Path]) -> Optional[dict]:
    """Load segment boundaries from config file."""
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

    genuine_end = genuine_seconds * fps
    black_end = genuine_end + (black_seconds * fps)
    imposter_end = black_end + (imposter_seconds * fps)

    return {
        'genuine': (0, genuine_end),
        'black': (genuine_end, black_end),
        'imposter': (black_end, imposter_end)
    }


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


def _add_video_traces(fig: go.Figure, video_frames_dict: dict[str, list[FrameData]]) -> None:
    """Add video traces to a plotly figure."""
    for video_path, frames in video_frames_dict.items():
        frames_sorted = sorted(frames, key=lambda f: f.frame)

        fig.add_trace(go.Scatter(
            x=[f.frame for f in frames_sorted],
            y=[f.risk_score for f in frames_sorted],
            mode='lines',
            name=Path(video_path).name,
            hovertemplate='<b>%{fullData.name}</b><br>Frame: %{x}<br>Risk: %{y:.4f}<extra></extra>',
            visible=True
        ))


def create_risk_timeline_all_videos(data: EvaluationData, study_name: str, config_path: Optional[Path] = None) -> go.Figure:
    """Create interactive risk score timeline showing all videos."""
    fig = go.Figure()

    # Add segment backgrounds if config available
    segments = _load_segment_config(config_path)
    _add_segment_backgrounds(fig, segments)

    videos = {}
    for frame in data.frames:
        if frame.video_path not in videos:
            videos[frame.video_path] = []
        videos[frame.video_path].append(frame)

    _add_video_traces(fig, videos)

    fig.add_hline(
        y=data.threshold,
        line_dash="dash",
        line_color="black"
    )

    fig.update_layout(
        title=f"{study_name} - Risk Score Timeline (All Videos)<br><sub>Device: All | Threshold: {data.threshold}</sub>",
        xaxis_title="Frame",
        yaxis_title="Risk Score",
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


def create_risk_timeline_by_device(data: EvaluationData, devices: list[str], study_name: str, config_path: Optional[Path] = None) -> list[go.Figure]:
    """Create independent risk score timelines for each device."""
    figures = []
    segments = _load_segment_config(config_path)

    for device in devices:
        device_frames = [f for f in data.frames if f.device == device]

        # Skip devices with no data
        if not device_frames:
            continue

        fig = go.Figure()

        # Add segment backgrounds
        _add_segment_backgrounds(fig, segments)

        videos = {}
        for frame in device_frames:
            if frame.video_path not in videos:
                videos[frame.video_path] = []
            videos[frame.video_path].append(frame)

        _add_video_traces(fig, videos)

        fig.add_hline(
            y=data.threshold,
            line_dash="dash",
            line_color="black"
        )

        fig.update_layout(
            title=f"{study_name} - {device.upper()}<br><sub>Threshold: {data.threshold}</sub>",
            xaxis_title="Frame",
            yaxis_title="Risk Score",
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


def plot_risk_distribution(ax: plt.Axes, data: EvaluationData) -> None:
    """Plot risk score distribution histogram."""
    genuine_frames = [f for f in data.frames if f.segment_type == SegmentType.GENUINE]
    imposter_frames = [f for f in data.frames if f.segment_type == SegmentType.IMPOSTER]

    genuine_risk = [f.risk_score for f in genuine_frames]
    imposter_risk = [f.risk_score for f in imposter_frames]

    ax.hist([genuine_risk, imposter_risk], bins=30, label=['Genuine', 'Imposter'],
            color=['green', 'red'], alpha=0.6, edgecolor='black')
    ax.axvline(data.threshold, color='black', linestyle='--', linewidth=2, label='Threshold')
    ax.set_xlabel('Risk Score')
    ax.set_ylabel('Frequency')
    ax.set_title('Risk Score Distribution')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)


def plot_error_rates_by_device(ax: plt.Axes, device_metrics: list[DeviceMetrics]) -> None:
    """Plot TAR/FRR/TRR/FAR bar chart by device."""
    devices = [dm.device for dm in device_metrics]
    tar_vals = [dm.metrics.true_accept_rate for dm in device_metrics]
    frr_vals = [dm.metrics.false_reject_rate for dm in device_metrics]
    trr_vals = [dm.metrics.true_reject_rate for dm in device_metrics]
    far_vals = [dm.metrics.false_accept_rate for dm in device_metrics]

    x = np.arange(len(devices))
    width = 0.2

    ax.bar(x - 1.5*width, tar_vals, width, label='TAR', color='#2ecc71', alpha=0.8)
    ax.bar(x - 0.5*width, frr_vals, width, label='FRR', color='#e74c3c', alpha=0.8)
    ax.bar(x + 0.5*width, trr_vals, width, label='TRR', color='#3498db', alpha=0.8)
    ax.bar(x + 1.5*width, far_vals, width, label='FAR', color='#f39c12', alpha=0.8)

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
    plot_risk_distribution(ax1, data)

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
    ax.axis('off')

    columns = ['Device', 'TAR (%)', 'FRR (%)', 'TRR (%)', 'FAR (%)', 'EER (%)', 'Avg Risk', 'Lockout (frames)']
    rows = []

    for dm in device_metrics:
        m = dm.metrics
        lockout = f"{m.imposter_lockout_time:.1f}" if m.imposter_lockout_time else "N/A"
        rows.append([
            dm.device.upper(),
            f"{m.true_accept_rate:.2f}",
            f"{m.false_reject_rate:.2f}",
            f"{m.true_reject_rate:.2f}",
            f"{m.false_accept_rate:.2f}",
            f"{m.equal_error_rate:.2f}",
            f"{m.avg_risk_score:.4f}",
            lockout
        ])

    table = ax.table(
        cellText=rows,
        colLabels=columns,
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1]
    )

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 3)

    for i in range(len(columns)):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(weight='bold', color='white')

    for i in range(len(rows)):
        for j in range(len(columns)):
            if i % 2 == 0:
                table[(i + 1, j)].set_facecolor('#ecf0f1')

    fig.suptitle('Overall Device Metrics', fontsize=16, y=0.95)

    return fig


def create_device_scenario_metrics_table(
    scenario_device_metrics: list[ScenarioDeviceMetrics],
    device: str
) -> plt.Figure:
    """Create scenario breakdown table for a specific device."""
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.axis('off')

    columns = ['Scenario', 'TAR (%)', 'FRR (%)', 'TRR (%)', 'FAR (%)', 'EER (%)', 'Avg Risk', 'Lockout (frames)']
    rows = []

    filtered_metrics = [sdm for sdm in scenario_device_metrics if sdm.device == device]

    for sdm in filtered_metrics:
        m = sdm.metrics
        lockout = f"{m.imposter_lockout_time:.1f}" if m.imposter_lockout_time else "N/A"
        rows.append([
            sdm.scenario.upper(),
            f"{m.true_accept_rate:.2f}",
            f"{m.false_reject_rate:.2f}",
            f"{m.true_reject_rate:.2f}",
            f"{m.false_accept_rate:.2f}",
            f"{m.equal_error_rate:.2f}",
            f"{m.avg_risk_score:.4f}",
            lockout
        ])

    table = ax.table(
        cellText=rows,
        colLabels=columns,
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1]
    )

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)

    for i in range(len(columns)):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(weight='bold', color='white')

    for i in range(len(rows)):
        for j in range(len(columns)):
            if i % 2 == 0:
                table[(i + 1, j)].set_facecolor('#ecf0f1')

    fig.suptitle(f'{device.upper()} - Scenario Breakdown', fontsize=16, y=0.96)

    return fig


def create_combined_metrics_tables(
    device_metrics: list[DeviceMetrics],
    scenario_device_metrics: list[ScenarioDeviceMetrics]
) -> plt.Figure:
    """Create combined figure with all 3 metric tables."""
    fig = plt.figure(figsize=(16, 14))
    gs = fig.add_gridspec(3, 1, hspace=0.4)

    columns = ['TAR (%)', 'FRR (%)', 'TRR (%)', 'FAR (%)', 'EER (%)', 'Avg Risk', 'Lockout (frames)']

    ax1 = fig.add_subplot(gs[0])
    ax1.axis('off')
    ax1.text(0.5, 0.95, 'Overall Device Metrics', ha='center', fontsize=14, weight='bold', transform=ax1.transAxes)

    device_columns = ['Device'] + columns
    device_rows = []
    for dm in device_metrics:
        m = dm.metrics
        lockout = f"{m.imposter_lockout_time:.1f}" if m.imposter_lockout_time else "N/A"
        device_rows.append([
            dm.device.upper(),
            f"{m.true_accept_rate:.2f}",
            f"{m.false_reject_rate:.2f}",
            f"{m.true_reject_rate:.2f}",
            f"{m.false_accept_rate:.2f}",
            f"{m.equal_error_rate:.2f}",
            f"{m.avg_risk_score:.4f}",
            lockout
        ])

    table1 = ax1.table(
        cellText=device_rows,
        colLabels=device_columns,
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 0.8]
    )
    table1.auto_set_font_size(False)
    table1.set_fontsize(10)
    table1.scale(1, 2.5)

    for i in range(len(device_columns)):
        table1[(0, i)].set_facecolor('#3498db')
        table1[(0, i)].set_text_props(weight='bold', color='white')
    for i in range(len(device_rows)):
        for j in range(len(device_columns)):
            if i % 2 == 0:
                table1[(i + 1, j)].set_facecolor('#ecf0f1')

    ax2 = fig.add_subplot(gs[1])
    ax2.axis('off')
    ax2.text(0.5, 0.95, 'MOBILE - Scenario Breakdown', ha='center', fontsize=14, weight='bold', transform=ax2.transAxes)

    scenario_columns = ['Scenario'] + columns
    mobile_rows = []
    mobile_metrics = [sdm for sdm in scenario_device_metrics if sdm.device == 'mobile']
    for sdm in mobile_metrics:
        m = sdm.metrics
        lockout = f"{m.imposter_lockout_time:.1f}" if m.imposter_lockout_time else "N/A"
        mobile_rows.append([
            sdm.scenario.upper(),
            f"{m.true_accept_rate:.2f}",
            f"{m.false_reject_rate:.2f}",
            f"{m.true_reject_rate:.2f}",
            f"{m.false_accept_rate:.2f}",
            f"{m.equal_error_rate:.2f}",
            f"{m.avg_risk_score:.4f}",
            lockout
        ])

    table2 = ax2.table(
        cellText=mobile_rows,
        colLabels=scenario_columns,
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 0.8]
    )
    table2.auto_set_font_size(False)
    table2.set_fontsize(10)
    table2.scale(1, 2.2)

    for i in range(len(scenario_columns)):
        table2[(0, i)].set_facecolor('#3498db')
        table2[(0, i)].set_text_props(weight='bold', color='white')
    for i in range(len(mobile_rows)):
        for j in range(len(scenario_columns)):
            if i % 2 == 0:
                table2[(i + 1, j)].set_facecolor('#ecf0f1')

    ax3 = fig.add_subplot(gs[2])
    ax3.axis('off')
    ax3.text(0.5, 0.95, 'DESKTOP - Scenario Breakdown', ha='center', fontsize=14, weight='bold', transform=ax3.transAxes)

    desktop_rows = []
    desktop_metrics = [sdm for sdm in scenario_device_metrics if sdm.device == 'desktop']
    for sdm in desktop_metrics:
        m = sdm.metrics
        lockout = f"{m.imposter_lockout_time:.1f}" if m.imposter_lockout_time else "N/A"
        desktop_rows.append([
            sdm.scenario.upper(),
            f"{m.true_accept_rate:.2f}",
            f"{m.false_reject_rate:.2f}",
            f"{m.true_reject_rate:.2f}",
            f"{m.false_accept_rate:.2f}",
            f"{m.equal_error_rate:.2f}",
            f"{m.avg_risk_score:.4f}",
            lockout
        ])

    table3 = ax3.table(
        cellText=desktop_rows,
        colLabels=scenario_columns,
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 0.8]
    )
    table3.auto_set_font_size(False)
    table3.set_fontsize(10)
    table3.scale(1, 2.2)

    for i in range(len(scenario_columns)):
        table3[(0, i)].set_facecolor('#3498db')
        table3[(0, i)].set_text_props(weight='bold', color='white')
    for i in range(len(desktop_rows)):
        for j in range(len(scenario_columns)):
            if i % 2 == 0:
                table3[(i + 1, j)].set_facecolor('#ecf0f1')

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

"""Create the thesis figure comparing operating modes as stacked timelines."""
from dataclasses import dataclass
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add src to path to enable imports when this script is run directly.
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from evaluation.shared.data_loader import load_evaluation_data
from evaluation.shared.metrics import calculate_metrics
from evaluation.shared.models import EvaluationData, FrameData
from evaluation.shared.visualization import save_html, save_plotly_png


@dataclass(frozen=True)
class RunSpec:
    label: str
    results_folder: str
    color: str


PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
THESIS_ROOT = PROJECT_ROOT.parent / "master-thesis"

RUNS = [
    RunSpec("Strict", "data/in_the_wild/_results_archive/MODES/strict", "#d62728"),
    RunSpec("Medium", "data/in_the_wild/_results_archive/MODES/medium", "#1f77b4"),
    RunSpec("Loose", "data/in_the_wild/_results_archive/MODES/loose", "#2ca02c"),
]

USE_COMMON_VIDEO_SUBSET = True
DEFAULT_OUTPUT_PATH = THESIS_ROOT / "images/modes"
DEFAULT_HTML_OUTPUT_PATH = PROJECT_ROOT / "src/evaluation/in_the_wild/output"
DEFAULT_PNG_FILENAME = "mode_trust_timeline.png"
DEFAULT_HTML_FILENAME = "mode_trust_timeline.html"

FIGURE_WIDTH = 1200
FIGURE_HEIGHT = 900
PLOTLY_EXPORT_SCALE = 2

LINE_OPACITY = 0.28
LINE_WIDTH = 1.1
MEAN_LINE_WIDTH = 3
REFERENCE_MEAN_LINE_WIDTH = 2.2
REFERENCE_MEAN_OPACITY = 0.32
THRESHOLD_LINE_WIDTH = 1.5
METRIC_MARKER_SIZE = 12

FONT_FAMILY = "Latin Modern Roman, serif"
FONT_COLOR = "black"
FONT_SIZE = 20
TICK_FONT_SIZE = 16
LEGEND_FONT_SIZE = 20
SEGMENT_LABEL_FONT_SIZE = 20
ROW_LABEL_FONT_SIZE = 18

LEGEND_Y = 1.04
SEGMENT_LABEL_Y = 0.985
LEFT_MARGIN = 88
RIGHT_MARGIN = 24
TOP_MARGIN = 92
BOTTOM_MARGIN = 72
VERTICAL_SPACING = 0.08

SEGMENT_COLORS = {
    "genuine": "rgba(46, 204, 113, 0.10)",
    "black": "rgba(149, 165, 166, 0.10)",
    "imposter": "rgba(231, 76, 60, 0.10)",
}
SEGMENT_LINE_COLORS = {
    "genuine": "rgba(46, 204, 113, 0.60)",
    "black": "rgba(149, 165, 166, 0.60)",
    "imposter": "rgba(231, 76, 60, 0.60)",
}


def _load_segment_config(config_path: Path) -> tuple[dict[str, tuple[float, float]], int]:
    import json

    with open(config_path) as f:
        config = json.load(f)

    imposter_config = config["imposter_creation"]
    fps = int(imposter_config["fps"])
    genuine_end = float(imposter_config["genuine_user_seconds"])
    black_end = genuine_end + float(imposter_config["black_screen_seconds"])
    imposter_end = black_end + float(imposter_config["impostor_seconds"])

    return {
        "genuine": (0.0, genuine_end),
        "black": (genuine_end, black_end),
        "imposter": (black_end, imposter_end),
    }, fps


def _filter_to_video_paths(data: EvaluationData, video_paths: set[str]) -> EvaluationData:
    frames = [frame for frame in data.frames if frame.video_path in video_paths]
    videos = [video for video in data.videos if video.video_path in video_paths]
    return EvaluationData(
        frames=frames,
        threshold=data.threshold,
        videos=videos,
        skip_frames=data.skip_frames,
        fps=data.fps,
    )


def _group_frames_by_video_path(frames: list[FrameData]) -> dict[str, list[FrameData]]:
    videos: dict[str, list[FrameData]] = {}
    for frame in frames:
        videos.setdefault(frame.video_path, []).append(frame)
    return videos


def _mean_trust_trace(frames: list[FrameData], fps: int) -> tuple[list[float], list[float]]:
    frames_by_frame: dict[int, list[float]] = {}
    for frame in frames:
        frames_by_frame.setdefault(frame.frame, []).append(frame.trust_score)

    sorted_frames = sorted(frames_by_frame)
    times = [frame / fps for frame in sorted_frames]
    mean_trust = [float(np.mean(frames_by_frame[frame])) for frame in sorted_frames]
    return times, mean_trust


def _add_segment_backgrounds(fig: go.Figure, segments: dict[str, tuple[float, float]], row: int) -> None:
    for segment_name, (start, end) in segments.items():
        if start >= end:
            continue
        fig.add_vrect(
            x0=start,
            x1=end,
            fillcolor=SEGMENT_COLORS.get(segment_name, "rgba(0, 0, 0, 0.05)"),
            layer="below",
            line_width=0,
            row=row,
            col=1,
        )

    boundaries: list[tuple[float, str]] = []
    for segment_name, (start, end) in segments.items():
        if start > 0 and start not in [position for position, _ in boundaries]:
            boundaries.append((start, segment_name))
        if end not in [position for position, _ in boundaries]:
            boundaries.append((end, segment_name))

    for position, segment_name in boundaries:
        fig.add_vline(
            x=position,
            line_color=SEGMENT_LINE_COLORS.get(segment_name, "rgba(0, 0, 0, 0.30)"),
            line_width=1.5,
            line_dash="dash",
            layer="below",
            row=row,
            col=1,
        )


def _add_video_traces(fig: go.Figure, data: EvaluationData, row: int) -> None:
    for video_path, frames in _group_frames_by_video_path(data.frames).items():
        frames_sorted = sorted(frames, key=lambda frame: frame.frame)
        fig.add_trace(
            go.Scatter(
                x=[frame.frame / data.fps for frame in frames_sorted],
                y=[frame.trust_score for frame in frames_sorted],
                mode="lines",
                name=Path(video_path).name,
                line=dict(width=LINE_WIDTH),
                opacity=LINE_OPACITY,
                showlegend=False,
                hovertemplate=(
                    "<b>%{fullData.name}</b><br>"
                    "Time: %{x:.2f}s<br>"
                    "Trust: %{y:.4f}<extra></extra>"
                ),
            ),
            row=row,
            col=1,
        )


def _add_mean_and_markers(
        fig: go.Figure,
        run: RunSpec,
        data: EvaluationData,
        all_mean_traces: dict[str, tuple[list[float], list[float], str]],
        segments: dict[str, tuple[float, float]],
        row: int,
        show_shared_legend: bool,
) -> None:
    metrics = calculate_metrics(data.frames, data.fps)
    current_mean_times, _ = all_mean_traces[run.label][:2]
    x_end = current_mean_times[-1] if current_mean_times else segments["imposter"][1]

    for label, (mean_times, mean_trust, color) in all_mean_traces.items():
        is_current_run = label == run.label
        fig.add_trace(
            go.Scatter(
                x=mean_times,
                y=mean_trust,
                mode="lines",
                name=f"{label} Mean",
                line=dict(
                    color=color,
                    width=MEAN_LINE_WIDTH if is_current_run else REFERENCE_MEAN_LINE_WIDTH,
                ),
                opacity=1.0 if is_current_run else REFERENCE_MEAN_OPACITY,
                showlegend=show_shared_legend,
                hovertemplate=f"{label} Mean: " + "%{y:.4f}<br>Time: %{x:.2f}s<extra></extra>",
            ),
            row=row,
            col=1,
        )
    fig.add_trace(
        go.Scatter(
            x=[0, x_end],
            y=[data.threshold, data.threshold],
            mode="lines",
            name="Threshold",
            line=dict(color="black", width=THRESHOLD_LINE_WIDTH, dash="dash"),
            showlegend=show_shared_legend,
            hovertemplate=f"Threshold: {data.threshold}<extra></extra>",
        ),
        row=row,
        col=1,
    )

    if show_shared_legend and "black" in segments:
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                name="Black Frames",
                marker=dict(size=10, color="rgba(149, 165, 166, 0.30)", symbol="square"),
                showlegend=True,
                hoverinfo="skip",
            ),
            row=row,
            col=1,
        )

    imposter_start = segments["imposter"][0]
    if metrics.imposter_lockout_time.mean is not None:
        fig.add_trace(
            go.Scatter(
                x=[imposter_start + metrics.imposter_lockout_time.mean],
                y=[data.threshold],
                mode="markers",
                name="Mean ULT",
                marker=dict(
                    size=METRIC_MARKER_SIZE,
                    color="rgba(142, 68, 173, 0.90)",
                    symbol="diamond",
                    line=dict(width=2, color="white"),
                ),
                showlegend=show_shared_legend,
                hovertemplate=f"Mean ULT: {metrics.imposter_lockout_time.mean:.1f}s<extra></extra>",
            ),
            row=row,
            col=1,
        )

    if metrics.imposter_lockout_time.p90 is not None:
        fig.add_trace(
            go.Scatter(
                x=[imposter_start + metrics.imposter_lockout_time.p90],
                y=[data.threshold],
                mode="markers",
                name="P90 ULT",
                marker=dict(
                    size=METRIC_MARKER_SIZE,
                    color="rgba(231, 76, 60, 0.90)",
                    symbol="diamond",
                    line=dict(width=2, color="white"),
                ),
                showlegend=show_shared_legend,
                hovertemplate=f"P90 ULT: {metrics.imposter_lockout_time.p90:.1f}s<extra></extra>",
            ),
            row=row,
            col=1,
        )


def _add_shared_annotations(
        fig: go.Figure,
        runs: list[RunSpec],
        segments: dict[str, tuple[float, float]],
) -> None:
    annotations = list(fig.layout.annotations) if fig.layout.annotations else []
    annotations.extend([
        dict(
            x=(segments["genuine"][0] + segments["genuine"][1]) / 2,
            y=SEGMENT_LABEL_Y,
            xref="x",
            yref="paper",
            text="<b>Genuine User</b>",
            showarrow=False,
            xanchor="center",
            yanchor="bottom",
            font=dict(size=SEGMENT_LABEL_FONT_SIZE, family=FONT_FAMILY, color=FONT_COLOR),
        ),
        dict(
            x=(segments["imposter"][0] + segments["imposter"][1]) / 2,
            y=SEGMENT_LABEL_Y,
            xref="x",
            yref="paper",
            text="<b>Unauthorized User</b>",
            showarrow=False,
            xanchor="center",
            yanchor="bottom",
            font=dict(size=SEGMENT_LABEL_FONT_SIZE, family=FONT_FAMILY, color=FONT_COLOR),
        ),
        dict(
            x=0.5,
            y=0.0,
            xref="paper",
            yref="paper",
            yshift=-54,
            text="Time (seconds)",
            showarrow=False,
            xanchor="center",
            yanchor="top",
            font=dict(size=FONT_SIZE, family=FONT_FAMILY, color=FONT_COLOR),
        ),
        dict(
            x=0.0,
            y=0.5,
            xref="paper",
            yref="paper",
            xshift=-62,
            text="Trust Score",
            textangle=-90,
            showarrow=False,
            xanchor="center",
            yanchor="middle",
            font=dict(size=FONT_SIZE, family=FONT_FAMILY, color=FONT_COLOR),
        ),
    ])

    for index, run in enumerate(runs, start=1):
        yaxis_name = "yaxis" if index == 1 else f"yaxis{index}"
        y_domain = getattr(fig.layout, yaxis_name).domain
        annotations.append(
            dict(
                x=0.012,
                y=y_domain[1] - 0.015,
                xref="paper",
                yref="paper",
                text=f"<b>{run.label}</b>",
                showarrow=False,
                xanchor="left",
                yanchor="top",
                font=dict(size=ROW_LABEL_FONT_SIZE, family=FONT_FAMILY, color=FONT_COLOR),
                bgcolor="rgba(255, 255, 255, 0.75)",
                borderpad=2,
            )
        )

    fig.update_layout(annotations=annotations)


def _create_mode_comparison_figure(run_data: list[tuple[RunSpec, EvaluationData]]) -> go.Figure:
    first_config_path = PROJECT_ROOT / run_data[0][0].results_folder / "config.json"
    segments, _ = _load_segment_config(first_config_path)
    all_mean_traces = {
        run.label: (*_mean_trust_trace(data.frames, data.fps), run.color)
        for run, data in run_data
    }

    fig = make_subplots(
        rows=len(run_data),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=VERTICAL_SPACING,
    )

    for row, (run, data) in enumerate(run_data, start=1):
        _add_segment_backgrounds(fig, segments, row)
        _add_video_traces(fig, data, row)
        _add_mean_and_markers(
            fig,
            run,
            data,
            all_mean_traces,
            segments,
            row,
            show_shared_legend=(row == 1),
        )

        fig.update_yaxes(
            range=[0, 1],
            tickfont=dict(size=TICK_FONT_SIZE, family=FONT_FAMILY, color=FONT_COLOR),
            showgrid=True,
            gridcolor="rgba(232, 238, 245, 1.0)",
            row=row,
            col=1,
        )
        fig.update_xaxes(
            tickfont=dict(size=TICK_FONT_SIZE, family=FONT_FAMILY, color=FONT_COLOR),
            showgrid=True,
            gridcolor="rgba(232, 238, 245, 1.0)",
            row=row,
            col=1,
        )

    fig.update_layout(
        template="plotly_white",
        height=FIGURE_HEIGHT,
        margin=dict(l=LEFT_MARGIN, r=RIGHT_MARGIN, t=TOP_MARGIN, b=BOTTOM_MARGIN),
        hovermode="closest",
        font=dict(size=FONT_SIZE, family=FONT_FAMILY, color=FONT_COLOR),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=LEGEND_Y,
            xanchor="right",
            x=1.0,
            font=dict(size=LEGEND_FONT_SIZE, family=FONT_FAMILY, color=FONT_COLOR),
        ),
    )
    fig.update_xaxes(title_text=None)
    fig.update_yaxes(title_text=None)
    _add_shared_annotations(fig, [run for run, _ in run_data], segments)

    return fig


def _load_runs() -> list[tuple[RunSpec, EvaluationData]]:
    loaded = [
        (run, load_evaluation_data(PROJECT_ROOT / run.results_folder / "results.csv", parse_scenario=False))
        for run in RUNS
    ]

    if not USE_COMMON_VIDEO_SUBSET:
        return loaded

    common_video_paths = set.intersection(
        *(set(video.video_path for video in data.videos) for _, data in loaded)
    )
    if not common_video_paths:
        raise ValueError("No common composed videos found across configured runs.")

    return [
        (run, _filter_to_video_paths(data, common_video_paths))
        for run, data in loaded
    ]


def main() -> None:
    print("Creating operating mode comparison timeline")
    for run in RUNS:
        print(f"{run.label}: {PROJECT_ROOT / run.results_folder / 'results.csv'}")
    print(f"PNG output: {DEFAULT_OUTPUT_PATH / DEFAULT_PNG_FILENAME}")

    run_data = _load_runs()
    fig = _create_mode_comparison_figure(run_data)

    html_output = save_html(fig, DEFAULT_HTML_OUTPUT_PATH, DEFAULT_HTML_FILENAME)
    png_output = save_plotly_png(
        fig,
        DEFAULT_OUTPUT_PATH,
        DEFAULT_PNG_FILENAME,
        width=FIGURE_WIDTH,
        height=FIGURE_HEIGHT,
    )

    print(f"Saved operating mode comparison HTML to: {html_output}")
    print(f"Saved operating mode comparison PNG to: {png_output}")
    print(f"Export scale configured as {PLOTLY_EXPORT_SCALE} through shared save_plotly_png.")


if __name__ == "__main__":
    main()

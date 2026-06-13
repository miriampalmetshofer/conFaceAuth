"""Create the thesis figure comparing operating modes as stacked timelines."""
from dataclasses import dataclass
import json
import sys
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add src to path to enable imports when this script is run directly.
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from evaluation.shared.data_loader import load_evaluation_data
from evaluation.shared.metrics import calculate_metrics
from evaluation.shared.models import AuthenticationMetrics, EvaluationData, SegmentType
from evaluation.shared.reporting import _fmt, _latex_header_cell, _latex_metric_header_cell
from evaluation.shared.visualization import (
    create_trust_timeline_all_videos,
    save_html,
    save_plotly_png,
)


@dataclass(frozen=True)
class RunSpec:
    label: str
    results_folder: str
    mean_line_dash: str


@dataclass(frozen=True)
class RunResult:
    spec: RunSpec
    data: EvaluationData
    metrics: AuthenticationMetrics


PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
THESIS_ROOT = PROJECT_ROOT.parent / "master-thesis"

RUNS = [
    RunSpec("Strict", "data/in_the_wild/_results_archive/V10", "solid"),
    RunSpec("Medium", "data/in_the_wild/_results_archive/V11", "dashdot"),
    RunSpec("Loose", "data/in_the_wild/_results_archive/V12", "dot"),
]

USE_COMMON_VIDEO_SUBSET = True
GENUINE_LOCKOUT_VIDEO_NAMES = [
    "alfred_2026-01-06_09-09-31",
]
DEFAULT_OUTPUT_PATH = THESIS_ROOT / "images/modes"
DEFAULT_HTML_OUTPUT_PATH = PROJECT_ROOT / "src/evaluation/in_the_wild/output"
DEFAULT_PNG_FILENAME = "mode_trust_timeline.png"
DEFAULT_HTML_FILENAME = "mode_trust_timeline.html"

FIGURE_WIDTH = 1200
FIGURE_HEIGHT = 1420
VERTICAL_SPACING = 0.06

MEAN_TRACE_COLOR = "black"
STANDARD_TRACE_COLORS = [
    "#636EFA",
    "#EF553B",
    "#00CC96",
    "#AB63FA",
    "#FFA15A",
    "#19D3F3",
    "#FF6692",
    "#B6E880",
    "#FF97FF",
    "#FECB52",
]
THRESHOLD_TRACE_COLOR = "rgba(0, 0, 0, 0.55)"
THRESHOLD_TRACE_WIDTH = 1.4
REFERENCE_MEAN_OPACITY = 0.8
REFERENCE_MEAN_LINE_WIDTH = 2.0
ACTIVE_MEAN_LINE_WIDTH = 5.2
ROW_LABEL_X = 0.012
ROW_LABEL_Y_OFFSET = 0.015
SEGMENT_LABEL_Y_OFFSET = 0.012
BOTTOM_MARGIN = 92


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


def _calculate_run_results(run_data: list[tuple[RunSpec, EvaluationData]]) -> list[RunResult]:
    return [
        RunResult(run, data, calculate_metrics(data.frames, data.fps))
        for run, data in run_data
    ]


def _create_standard_figures(run_results: list[RunResult]) -> list[tuple[RunResult, go.Figure]]:
    standard_figures = []
    for run_result in run_results:
        config_path = PROJECT_ROOT / run_result.spec.results_folder / "config.json"
        standard_figures.append((
            run_result,
            create_trust_timeline_all_videos(
                run_result.data,
                "In-the-Wild Study",
                config_path,
                run_result.metrics,
            ),
        ))
    return standard_figures


def _mean_traces_by_run(
        standard_figures: list[tuple[RunResult, go.Figure]]
) -> dict[str, tuple[RunSpec, go.Scatter]]:
    mean_traces = {}
    for run_result, standard_fig in standard_figures:
        mean_trace = next(trace for trace in standard_fig.data if trace.name == "Mean Trust")
        mean_traces[run_result.spec.label] = (run_result.spec, mean_trace)
    return mean_traces


def _add_standard_shapes(fig: go.Figure, standard_fig: go.Figure, row: int) -> None:
    for shape in standard_fig.layout.shapes or []:
        shape_data = shape.to_plotly_json()
        line = shape_data.get("line", {})

        if shape_data.get("type") == "rect":
            fig.add_vrect(
                x0=shape_data["x0"],
                x1=shape_data["x1"],
                fillcolor=shape_data.get("fillcolor"),
                layer=shape_data.get("layer", "below"),
                line_width=line.get("width", 0),
                exclude_empty_subplots=False,
                row=row,
                col=1,
            )
        elif shape_data.get("type") == "line" and shape_data.get("x0") == shape_data.get("x1"):
            fig.add_vline(
                x=shape_data["x0"],
                line_color=line.get("color"),
                line_width=line.get("width", 2),
                line_dash=line.get("dash"),
                layer=shape_data.get("layer", "below"),
                exclude_empty_subplots=False,
                row=row,
                col=1,
            )


def _copy_standard_non_mean_traces(fig: go.Figure, standard_fig: go.Figure, row: int, show_shared_legend: bool) -> None:
    individual_trace_index = 0
    for trace in standard_fig.data:
        if trace.name == "Mean Trust" or _is_event_marker_trace(trace):
            continue

        copied_trace = go.Figure(data=[trace]).data[0]
        if _is_individual_video_trace(copied_trace):
            line = copied_trace.line.to_plotly_json() if copied_trace.line else {}
            line["color"] = STANDARD_TRACE_COLORS[individual_trace_index % len(STANDARD_TRACE_COLORS)]
            copied_trace.line = line
            individual_trace_index += 1
        elif copied_trace.name == "Threshold":
            copied_trace.line = dict(
                color=THRESHOLD_TRACE_COLOR,
                width=THRESHOLD_TRACE_WIDTH,
                dash="dash",
            )
        copied_trace.showlegend = bool(show_shared_legend and trace.showlegend)
        fig.add_trace(copied_trace, row=row, col=1)


def _add_event_marker_traces(fig: go.Figure, standard_fig: go.Figure, row: int, show_shared_legend: bool) -> None:
    for trace in standard_fig.data:
        if not _is_event_marker_trace(trace):
            continue

        copied_trace = go.Figure(data=[trace]).data[0]
        copied_trace.showlegend = bool(show_shared_legend and trace.showlegend)
        copied_trace.cliponaxis = False
        fig.add_trace(copied_trace, row=row, col=1)


def _is_event_marker_trace(trace: go.Scatter) -> bool:
    return trace.name in {"Mean ULT", "P90 ULT"}


def _is_individual_video_trace(trace: go.Scatter) -> bool:
    return (
        trace.mode == "lines"
        and trace.name not in {"Threshold", "Mean ULT", "P90 ULT"}
        and not trace.showlegend
    )


def _add_mean_legend(fig: go.Figure, mean_traces: dict[str, tuple[RunSpec, go.Scatter]]) -> None:
    for label, (run, _) in mean_traces.items():
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="lines",
                name=f"{label} GT",
                line=dict(
                    color=MEAN_TRACE_COLOR,
                    width=ACTIVE_MEAN_LINE_WIDTH,
                    dash=run.mean_line_dash,
                ),
                showlegend=True,
                hoverinfo="skip",
            ),
            row=1,
            col=1,
        )


def _add_mode_mean_traces(
        fig: go.Figure,
        active_run: RunSpec,
        mean_traces: dict[str, tuple[RunSpec, go.Scatter]],
        row: int,
) -> None:
    for label, (run, mean_trace) in mean_traces.items():
        active = label == active_run.label
        fig.add_trace(
            go.Scatter(
                x=mean_trace.x,
                y=mean_trace.y,
                mode="lines",
                name=f"{label} GT",
                line=dict(
                    color=MEAN_TRACE_COLOR,
                    width=ACTIVE_MEAN_LINE_WIDTH if active else REFERENCE_MEAN_LINE_WIDTH,
                    dash=run.mean_line_dash,
                ),
                opacity=1.0 if active else REFERENCE_MEAN_OPACITY,
                showlegend=False,
                hovertemplate=f"{label} GT: " + "%{y:.4f}<br>Time: %{x:.2f}s<extra></extra>",
            ),
            row=row,
            col=1,
        )


def _standard_axis_style(axis) -> dict:
    return dict(
        title_text=None,
        tickfont=axis.tickfont,
        title_font=axis.title.font,
        showgrid=True,
        gridcolor="rgba(232, 238, 245, 1.0)",
    )


def _add_shared_annotations(
        fig: go.Figure,
        standard_figures: list[tuple[RunResult, go.Figure]],
) -> None:
    first_standard = standard_figures[0][1]
    segment_ranges = _segment_ranges_from_standard_figure(first_standard)
    annotations = []

    top_y = fig.layout.yaxis.domain[1] + SEGMENT_LABEL_Y_OFFSET
    annotations.extend([
        _segment_header_annotation(
            text="<b>Genuine User</b>",
            x=_paper_x_from_data_center(fig, *segment_ranges["genuine"]),
            y=top_y,
            font=first_standard.layout.font.to_plotly_json(),
        ),
        _segment_header_annotation(
            text="<b>Unauthorized User</b>",
            x=_paper_x_from_data_center(fig, *segment_ranges["imposter"]),
            y=top_y,
            font=first_standard.layout.font.to_plotly_json(),
        ),
    ])

    for index, (run_result, _) in enumerate(standard_figures, start=1):
        yaxis_name = "yaxis" if index == 1 else f"yaxis{index}"
        y_domain = getattr(fig.layout, yaxis_name).domain
        annotations.append(
            dict(
                x=ROW_LABEL_X,
                y=y_domain[1] - ROW_LABEL_Y_OFFSET,
                xref="paper",
                yref="paper",
                text=f"<b>{run_result.spec.label}</b>",
                showarrow=False,
                xanchor="left",
                yanchor="top",
                font=first_standard.layout.font.to_plotly_json(),
                bgcolor="rgba(255, 255, 255, 0.75)",
                borderpad=2,
            )
        )

    annotations.extend([
        dict(
            x=0.5,
            y=0.0,
            xref="paper",
            yref="paper",
            yshift=-42,
            text="Time (seconds)",
            showarrow=False,
            xanchor="center",
            yanchor="top",
            font=first_standard.layout.xaxis.title.font.to_plotly_json(),
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
            font=first_standard.layout.yaxis.title.font.to_plotly_json(),
        ),
    ])
    fig.update_layout(annotations=annotations)


def _segment_ranges_from_standard_figure(fig: go.Figure) -> dict[str, tuple[float, float]]:
    rects = [
        shape
        for shape in fig.layout.shapes or []
        if shape.type == "rect"
    ]
    if len(rects) >= 3:
        return {
            "genuine": (rects[0].x0, rects[0].x1),
            "imposter": (rects[2].x0, rects[2].x1),
        }
    return {
        "genuine": (0, 300),
        "imposter": (303, 600),
    }


def _paper_x_from_data_center(fig: go.Figure, start: float, end: float) -> float:
    x_domain = fig.layout.xaxis.domain
    x_range = fig.layout.xaxis.range or [0, 600]
    visible_start = max(start, x_range[0])
    visible_end = min(end, x_range[1])
    center = (visible_start + visible_end) / 2
    return x_domain[0] + ((center - x_range[0]) / (x_range[1] - x_range[0])) * (x_domain[1] - x_domain[0])


def _segment_header_annotation(text: str, x: float, y: float, font: dict) -> dict:
    return dict(
        x=x,
        y=y,
        xref="paper",
        yref="paper",
        text=text,
        showarrow=False,
        xanchor="center",
        yanchor="bottom",
        font=font,
    )


def _create_mode_comparison_figure(run_results: list[RunResult]) -> go.Figure:
    standard_figures = _create_standard_figures(run_results)
    mean_traces = _mean_traces_by_run(standard_figures)
    first_standard = standard_figures[0][1]

    fig = make_subplots(
        rows=len(standard_figures),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=VERTICAL_SPACING,
    )

    _add_mean_legend(fig, mean_traces)

    for row, (run_result, standard_fig) in enumerate(standard_figures, start=1):
        _add_standard_shapes(fig, standard_fig, row)
        _copy_standard_non_mean_traces(fig, standard_fig, row, show_shared_legend=(row == 1))
        _add_mode_mean_traces(fig, run_result.spec, mean_traces, row)
        _add_event_marker_traces(fig, standard_fig, row, show_shared_legend=(row == 1))

        fig.update_yaxes(range=[0, 1], **_standard_axis_style(first_standard.layout.yaxis), row=row, col=1)
        fig.update_xaxes(**_standard_axis_style(first_standard.layout.xaxis), row=row, col=1)

    fig.update_layout(
        template=first_standard.layout.template,
        height=FIGURE_HEIGHT,
        margin=dict(
            l=first_standard.layout.margin.l,
            r=first_standard.layout.margin.r,
            t=first_standard.layout.margin.t,
            b=BOTTOM_MARGIN,
        ),
        hovermode=first_standard.layout.hovermode,
        font=first_standard.layout.font,
        legend=first_standard.layout.legend,
    )
    fig.update_layout(legend=dict(y=1.04))
    _add_shared_annotations(fig, standard_figures)
    return fig


def _mode_parameter_values(run: RunSpec) -> tuple[str, str, str]:
    config_path = PROJECT_ROOT / run.results_folder / "config.json"
    with open(config_path) as file:
        config = json.load(file)

    temporal_decay = config["authentication"]["temporal_decay"]
    return (
        f"{temporal_decay['threshold']:g}",
        f"{temporal_decay['k_weight']:g}",
        f"{temporal_decay['k_decay']:g}",
    )


def _mode_table_metric_defs():
    return [definition for definition in AuthenticationMetrics.METRIC_DEFINITIONS if definition.include_in_tables]


def _mode_latex_table_lines(run_results: list[RunResult]) -> list[str]:
    table_defs = _mode_table_metric_defs()
    col_spec = "@{\\extracolsep{\\fill}}lcc" + "c" * len(table_defs)
    headers = [_latex_header_cell("Mode"), _latex_header_cell("$k_w$"), _latex_header_cell("$k_d$")] + [
        _latex_metric_header_cell(definition) for definition in table_defs
    ]

    lines = [
        "% ── LaTeX table (tabular only) ──────────────────────────────",
        f"\\begin{{tabular*}}{{\\textwidth}}{{{col_spec}}}",
        "\\toprule",
        " & ".join(headers) + r" \\",
        "\\midrule",
    ]

    for run_result in run_results:
        _, k_weight, k_decay = _mode_parameter_values(run_result.spec)
        metric_cells = [_fmt(run_result.metrics, definition, latex=True) for definition in table_defs]
        lines.append(" & ".join([run_result.spec.label, k_weight, k_decay] + metric_cells) + r" \\")

    lines.extend([
        "\\bottomrule",
        "\\end{tabular*}",
        "% ─────────────────────────────────────────────────────────────",
    ])
    return lines


def _mode_plain_table_lines(run_results: list[RunResult]) -> list[str]:
    table_defs = _mode_table_metric_defs()
    headers = ["Mode", "k_w", "k_d"] + [definition.short_label for definition in table_defs]
    rows = []
    for run_result in run_results:
        _, k_weight, k_decay = _mode_parameter_values(run_result.spec)
        rows.append(
            [run_result.spec.label, k_weight, k_decay]
            + [_fmt(run_result.metrics, definition, latex=False) for definition in table_defs]
        )

    widths = [
        max(len(str(row[index])) for row in [headers] + rows)
        for index in range(len(headers))
    ]
    lines = [
        "Mode comparison",
        "",
        "  ".join(str(cell).ljust(widths[index]) for index, cell in enumerate(headers)),
        "  ".join("-" * width for width in widths),
    ]
    lines.extend(
        "  ".join(str(cell).ljust(widths[index]) for index, cell in enumerate(row))
        for row in rows
    )
    return lines


def _normalize_video_name(video_name: str) -> str:
    return Path(video_name).stem


def _genuine_lockout_times_for_source(data: EvaluationData, source_name: str) -> list[float]:
    """Return genuine-user lockout times for one source video across pairings."""
    normalized_source = _normalize_video_name(source_name)
    frames_by_video: dict[str, list] = {}
    for frame in data.frames:
        if frame.segment_type != SegmentType.GENUINE:
            continue
        if _normalize_video_name(frame.source_type) != normalized_source:
            continue
        frames_by_video.setdefault(frame.video_path, []).append(frame)

    lockout_times = []
    for frames in frames_by_video.values():
        sorted_frames = sorted(frames, key=lambda frame: frame.frame)
        for current, following in zip(sorted_frames, sorted_frames[1:]):
            if current.predicted_state == "Unlocked" and following.predicted_state == "Locked":
                lockout_times.append(following.frame / data.fps)
                break
    return lockout_times


def _format_lockout_times(lockout_times: list[float]) -> str:
    if not lockout_times:
        return "--"
    unique_times = sorted(set(round(time, 1) for time in lockout_times))
    return ", ".join(f"{time:g} s" for time in unique_times)


def _specific_genuine_lockout_table_lines(run_results: list[RunResult]) -> list[str]:
    headers = ["Video", "Mode", "Genuine lockout time", "Pairings"]
    rows = []
    for video_name in GENUINE_LOCKOUT_VIDEO_NAMES:
        for run_result in run_results:
            lockout_times = _genuine_lockout_times_for_source(run_result.data, video_name)
            rows.append([
                _normalize_video_name(video_name),
                run_result.spec.label,
                _format_lockout_times(lockout_times),
                str(len(lockout_times)),
            ])

    widths = [
        max(len(str(row[index])) for row in [headers] + rows)
        for index in range(len(headers))
    ]
    lines = [
        "Configured genuine-user lockout times",
        "Times are first lockouts during the genuine-user segment for the configured source video.",
        "",
        "  ".join(str(cell).ljust(widths[index]) for index, cell in enumerate(headers)),
        "  ".join("-" * width for width in widths),
    ]
    lines.extend(
        "  ".join(str(cell).ljust(widths[index]) for index, cell in enumerate(row))
        for row in rows
    )
    return lines


def _print_mode_table(run_results: list[RunResult]) -> None:
    print("\n".join(_mode_plain_table_lines(run_results)))
    print()
    print("\n".join(_specific_genuine_lockout_table_lines(run_results)))
    print()
    print("\n".join(_mode_latex_table_lines(run_results)))
    print()


def main() -> None:
    print("Creating operating mode comparison timeline")
    for run in RUNS:
        print(f"{run.label}: {PROJECT_ROOT / run.results_folder / 'results.csv'}")
    print(f"PNG output: {DEFAULT_OUTPUT_PATH / DEFAULT_PNG_FILENAME}")

    run_results = _calculate_run_results(_load_runs())
    _print_mode_table(run_results)
    fig = _create_mode_comparison_figure(run_results)

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


if __name__ == "__main__":
    main()

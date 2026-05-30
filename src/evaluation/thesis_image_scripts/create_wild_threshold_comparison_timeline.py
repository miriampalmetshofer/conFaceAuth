"""Create the in-the-wild trust timeline with two annotated thresholds."""
import json
import sys
from pathlib import Path

# Add src to path to enable imports when this script is run directly.
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from evaluation.shared.data_loader import load_evaluation_data
from evaluation.shared.metrics import calculate_metrics
from evaluation.shared.visualization import (
    create_trust_timeline_all_videos,
    save_plotly_png,
)


PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
THESIS_ROOT = PROJECT_ROOT.parent / "master-thesis"
PRIMARY_RESULTS_FOLDER = "data/in_the_wild/_results_archive/V10"
SECONDARY_RESULTS_FOLDER = "data/in_the_wild/_results_archive/V09"
DEFAULT_PNG_OUTPUT_PATH = THESIS_ROOT / "images/trust_scores"
DEFAULT_PNG_FILENAME = "wild_threshold_comparison.png"
PRIMARY_THRESHOLD = 0.5
SECONDARY_THRESHOLD = 0.4
SECONDARY_THRESHOLD_COLOR = "#0072B2"
SECONDARY_THRESHOLD_LINE_WIDTH = 2
THRESHOLD_LABEL_X = 0.975
THRESHOLD_LABEL_YSHIFT = 22


def _format_threshold(threshold: float) -> str:
    return f"{threshold:g}"


def _customize_thresholds(fig, primary_threshold: float, secondary_threshold: float) -> None:
    """Add the comparison threshold without changing the shared figure builder."""
    primary_label = f"Threshold {_format_threshold(primary_threshold)}"
    secondary_label = f"Threshold {_format_threshold(secondary_threshold)}"
    x_end = _get_timeline_end(fig)

    for trace in fig.data:
        if trace.name == "Threshold":
            trace.name = primary_label
            trace.showlegend = False
            trace.hovertemplate = f"{primary_label}<extra></extra>"
            break

    fig.add_scatter(
        x=[0, x_end],
        y=[secondary_threshold, secondary_threshold],
        mode="lines",
        name=secondary_label,
        line=dict(color=SECONDARY_THRESHOLD_COLOR, width=SECONDARY_THRESHOLD_LINE_WIDTH, dash="dash"),
        showlegend=False,
        hovertemplate=f"{secondary_label}<extra></extra>",
    )

    annotations = list(fig.layout.annotations) if fig.layout.annotations else []
    annotations.extend([
        _threshold_annotation(primary_threshold, primary_label, "black"),
        _threshold_annotation(secondary_threshold, secondary_label, SECONDARY_THRESHOLD_COLOR),
    ])
    fig.update_layout(annotations=annotations)


def _add_secondary_metric_markers(fig, metrics, threshold: float, imposter_start: float) -> None:
    suffix = f" ({_format_threshold(threshold)})"
    if metrics.imposter_lockout_time.mean is not None:
        fig.add_scatter(
            x=[imposter_start + metrics.imposter_lockout_time.mean],
            y=[threshold],
            mode="markers",
            name="Mean ULT",
            marker=dict(
                size=15,
                color="rgba(142, 68, 173, 0.9)",
                symbol="diamond",
                line=dict(width=2, color="white"),
            ),
            showlegend=False,
            hovertemplate=f"Mean ULT{suffix}: {metrics.imposter_lockout_time.mean:.1f}s<extra></extra>",
        )

    if metrics.imposter_lockout_time.p90 is not None:
        fig.add_scatter(
            x=[imposter_start + metrics.imposter_lockout_time.p90],
            y=[threshold],
            mode="markers",
            name="P90 ULT",
            marker=dict(
                size=15,
                color="rgba(231, 76, 60, 0.9)",
                symbol="diamond",
                line=dict(width=2, color="white"),
            ),
            showlegend=False,
            hovertemplate=f"P90 ULT{suffix}: {metrics.imposter_lockout_time.p90:.1f}s<extra></extra>",
        )


def _load_imposter_start(config_path: Path) -> float:
    with open(config_path) as f:
        config = json.load(f)
    imposter_config = config["imposter_creation"]
    return imposter_config["genuine_user_seconds"] + imposter_config["black_screen_seconds"]


def _get_timeline_end(fig) -> float:
    x_values = [
        value
        for trace in fig.data
        if trace.x is not None
        for value in trace.x
        if isinstance(value, (int, float))
    ]
    return max(x_values) if x_values else 0


def _threshold_annotation(threshold: float, label: str, color: str) -> dict:
    return dict(
        x=THRESHOLD_LABEL_X,
        y=threshold,
        xref="paper",
        yref="y",
        text=label,
        showarrow=False,
        xanchor="right",
        yanchor="bottom",
        yshift=THRESHOLD_LABEL_YSHIFT,
        font=dict(size=18, family="Latin Modern Roman, serif", color=color),
        bgcolor="rgba(255, 255, 255, 1.0)",
        borderpad=2,
    )


def _move_metric_markers_to_front(fig) -> None:
    metric_marker_names = {"Mean ULT", "P90 ULT"}
    metric_markers = [trace for trace in fig.data if trace.name in metric_marker_names]
    other_traces = [trace for trace in fig.data if trace.name not in metric_marker_names]
    fig.data = tuple(other_traces + metric_markers)


def main() -> None:
    results_path = PROJECT_ROOT / PRIMARY_RESULTS_FOLDER / "results.csv"
    config_path = PROJECT_ROOT / PRIMARY_RESULTS_FOLDER / "config.json"
    secondary_results_path = PROJECT_ROOT / SECONDARY_RESULTS_FOLDER / "results.csv"
    secondary_config_path = PROJECT_ROOT / SECONDARY_RESULTS_FOLDER / "config.json"

    print("Creating in-the-wild threshold comparison timeline")
    print(f"Results: {results_path}")
    print(f"Secondary results: {secondary_results_path}")
    print(f"PNG output: {DEFAULT_PNG_OUTPUT_PATH / DEFAULT_PNG_FILENAME}")

    data = load_evaluation_data(results_path, parse_scenario=False)
    if data.threshold != PRIMARY_THRESHOLD:
        raise ValueError(
            f"Expected primary threshold {PRIMARY_THRESHOLD}, but loaded {data.threshold} from {results_path}."
        )
    secondary_data = load_evaluation_data(secondary_results_path, parse_scenario=False)
    if secondary_data.threshold != SECONDARY_THRESHOLD:
        raise ValueError(
            f"Expected secondary threshold {SECONDARY_THRESHOLD}, but loaded {secondary_data.threshold} "
            f"from {secondary_results_path}."
        )

    metrics = calculate_metrics(data.frames, data.fps)
    secondary_metrics = calculate_metrics(secondary_data.frames, secondary_data.fps)
    imposter_start = _load_imposter_start(secondary_config_path)
    fig = create_trust_timeline_all_videos(
        data,
        "In-the-Wild Study",
        config_path,
        metrics,
    )
    _customize_thresholds(fig, PRIMARY_THRESHOLD, SECONDARY_THRESHOLD)
    _add_secondary_metric_markers(fig, secondary_metrics, SECONDARY_THRESHOLD, imposter_start)
    _move_metric_markers_to_front(fig)

    png_output = save_plotly_png(fig, DEFAULT_PNG_OUTPUT_PATH, DEFAULT_PNG_FILENAME, width=1200, height=560)

    print(f"Saved threshold comparison PNG to: {png_output}")


if __name__ == "__main__":
    main()

"""Create an in-the-wild trust timeline for selected participants."""
import argparse
import sys
from pathlib import Path

# Add src to path to enable imports when this script is run directly.
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from evaluation.shared.data_loader import load_evaluation_data
from evaluation.shared.metrics import calculate_metrics
from evaluation.shared.models import EvaluationData
from evaluation.shared.visualization import (
    create_trust_timeline_all_videos,
    save_html,
    save_plotly_png,
)


PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
DEFAULT_RESULTS_FOLDER = "data/in_the_wild/_results_archive/V04"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "src/evaluation/in_the_wild/output"
DEFAULT_PARTICIPANTS = ["miriam"]
DEFAULT_EXCLUDED_COMPARISONS = []
DEFAULT_EXCLUDED_GENUINE_DATES = ["2025-12-22", "2025-12-27"]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a trust score timeline for selected in-the-wild participants."
    )
    parser.add_argument(
        "--participants",
        nargs="+",
        default=DEFAULT_PARTICIPANTS,
        help="Participant names to include. Defaults to miriam.",
    )
    parser.add_argument(
        "--results-folder",
        default=DEFAULT_RESULTS_FOLDER,
        help="Results folder relative to conFaceAuth. Defaults to the V04 in-the-wild archive.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output HTML filename. Defaults to trust_timeline_participants_<names>.html.",
    )
    parser.add_argument(
        "--exclude-comparisons",
        nargs="*",
        default=DEFAULT_EXCLUDED_COMPARISONS,
        help="Compared users to exclude from the timeline. Defaults to none.",
    )
    parser.add_argument(
        "--exclude-genuine-dates",
        nargs="*",
        default=DEFAULT_EXCLUDED_GENUINE_DATES,
        help="Recording dates to exclude anywhere in the composed video path. Defaults to 2025-12-22 and 2025-12-27.",
    )
    return parser.parse_args()


def _filter_data(
        data: EvaluationData,
        participants: list[str],
        excluded_comparisons: list[str],
        excluded_genuine_dates: list[str],
) -> EvaluationData:
    selected = {participant.lower() for participant in participants}
    excluded = {comparison.lower() for comparison in excluded_comparisons}
    excluded_dates = {_normalize_date(date) for date in excluded_genuine_dates}

    filtered_frames = [
        frame for frame in data.frames
        if (
            frame.participant.lower() in selected
            and not any(f"_vs_{comparison}_" in frame.video_path.lower() for comparison in excluded)
            and not _contains_excluded_date(frame.video_path, excluded_dates)
        )
    ]

    if not filtered_frames:
        available = sorted({frame.participant for frame in data.frames})
        raise ValueError(
            "No frames found for selected participants "
            f"{participants}. Available participants: {', '.join(available)}"
        )

    video_paths = {frame.video_path for frame in filtered_frames}
    filtered_videos = [
        video for video in data.videos
        if video.video_path in video_paths
    ]

    return EvaluationData(
        frames=filtered_frames,
        threshold=data.threshold,
        videos=filtered_videos,
        skip_frames=data.skip_frames,
        fps=data.fps,
    )


def _normalize_date(date: str) -> str:
    parts = date.split("-")
    if len(parts) == 3 and len(parts[0]) == 2:
        return f"20{date}"
    return date


def _contains_excluded_date(video_path: str, excluded_dates: set[str]) -> bool:
    path = video_path.lower()
    return any(date in path for date in excluded_dates)


def _customize_metric_markers(fig, metrics, threshold: float) -> None:
    """Adjust metric markers for the participant-specific thesis figure."""
    fig.data = tuple(trace for trace in fig.data if trace.name != "P90 ULT")
    fig.layout.shapes = [
        shape for shape in fig.layout.shapes
        if not (
            shape.type == "line"
            and shape.y0 == threshold
            and shape.y1 == threshold
        )
    ]

    if metrics.genuine_lockout_time.mean is None:
        _move_metric_markers_to_front(fig)
        return

    fig.add_scatter(
        x=[metrics.genuine_lockout_time.mean],
        y=[threshold],
        mode="markers",
        name="Mean GKT",
        marker=dict(
            size=15,
            color="rgba(39, 174, 96, 0.9)",
            symbol="circle",
            line=dict(width=2, color="white"),
        ),
        showlegend=True,
        hovertemplate=f"Mean GKT: {metrics.genuine_lockout_time.mean:.1f}s<extra></extra>",
    )
    _move_metric_markers_to_front(fig)


def _move_metric_markers_to_front(fig) -> None:
    marker_names = {"Mean ULT", "Mean GKT"}
    threshold_group = {"Threshold", "Black Frames"}

    metric_markers = [trace for trace in fig.data if trace.name in marker_names]
    threshold_and_black = [trace for trace in fig.data if trace.name in threshold_group]
    other_traces = [trace for trace in fig.data if trace.name not in marker_names and trace.name not in threshold_group]

    fig.data = tuple(other_traces + threshold_and_black + metric_markers)


def _default_output_filename(participants: list[str]) -> str:
    return f"trust_timeline_participant.html"


def main() -> None:
    args = _parse_args()

    results_path = PROJECT_ROOT / args.results_folder / "results.csv"
    config_path = PROJECT_ROOT / args.results_folder / "config.json"
    output_filename = args.output or _default_output_filename(args.participants)

    print("Creating participant trust timeline")
    print(f"Results: {results_path}")
    print(f"Participants: {', '.join(args.participants)}")
    if args.exclude_comparisons:
        print(f"Excluded comparisons: {', '.join(args.exclude_comparisons)}")
    if args.exclude_genuine_dates:
        print(f"Excluded dates: {', '.join(args.exclude_genuine_dates)}")
    print(f"Output: {DEFAULT_OUTPUT_PATH / output_filename}")

    data = load_evaluation_data(results_path, parse_scenario=False)
    participant_data = _filter_data(
        data,
        args.participants,
        args.exclude_comparisons,
        args.exclude_genuine_dates,
    )
    metrics = calculate_metrics(participant_data.frames, participant_data.fps)

    fig = create_trust_timeline_all_videos(
        participant_data,
        "In-the-Wild Study",
        config_path,
        metrics,
    )
    _customize_metric_markers(fig, metrics, participant_data.threshold)

    html_output = save_html(fig, DEFAULT_OUTPUT_PATH, output_filename)
    png_filename = output_filename.replace('.html', '.png')
    png_output = save_plotly_png(fig, DEFAULT_OUTPUT_PATH, png_filename, width=1200, height=560)

    print(f"Saved participant trust timeline HTML to: {html_output}")
    print(f"Saved participant trust timeline PNG to: {png_output}")


if __name__ == "__main__":
    main()

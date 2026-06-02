"""Console reporting utilities."""
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from evaluation.shared.models import DeviceMetrics, ScenarioMetrics, ScenarioDeviceMetrics, AuthenticationMetrics, FrameData, SegmentType, MetricDefinition, EvaluationData


def print_section(title: str) -> None:
    """Print formatted section header."""
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}\n")


def print_dataset_summary(frames: list[FrameData], total_videos: int) -> None:
    """Print summary of dataset including unique genuine videos and total sessions."""
    unique_genuine_videos = set()
    genuine_videos_by_participant: dict[str, set[str]] = {}
    for frame in frames:
        if frame.segment_type == SegmentType.GENUINE:
            unique_genuine_videos.add(frame.source_type)
            genuine_videos_by_participant.setdefault(frame.participant, set()).add(frame.source_type)

    print(f"Total user videos: {len(unique_genuine_videos)}")
    videos_per_participant = [len(videos) for videos in genuine_videos_by_participant.values()]
    if videos_per_participant:
        mean_videos = sum(videos_per_participant) / len(videos_per_participant)
        print(
            "Videos per participant: "
            f"min {min(videos_per_participant)}, "
            f"max {max(videos_per_participant)}, "
            f"mean {mean_videos:.1f}"
        )
    print(f"Total generated evaluation sessions: {total_videos}")

    print()


def print_metrics_by_device(device_metrics: list[DeviceMetrics]) -> None:
    """Print metrics table grouped by device."""
    print_section("METRICS BY DEVICE")

    for dm in device_metrics:
        print(f"{dm.device.upper()}")
        dm.metrics.print_console(indent="  ")
        print()


def print_metrics_by_scenario(scenario_metrics: list[ScenarioMetrics]) -> None:
    """Print metrics table grouped by scenario."""
    print_section("METRICS BY SCENARIO")

    for sm in scenario_metrics:
        print(f"{sm.scenario.upper()}")
        sm.metrics.print_console(indent="  ")
        print()


def _latex_escape(text: str) -> str:
    """Escape LaTeX special characters in a cell value."""
    return re.sub(r"(?<!\\)%", r"\\%", text)


def _fmt(metrics: AuthenticationMetrics, defn: MetricDefinition, latex: bool = False) -> str:
    value = metrics._resolve_field(defn.field_name)
    if value is None:
        return "--"
    cell = f"{value:{defn.format_spec}}"
    if defn.field_name in {"false_reject_rate", "false_accept_rate"}:
        cell += r"\%" if latex else "%"
    if defn.field_name in {"imposter_lockout_time.mean", "imposter_lockout_time.p90"}:
        cell += r"\,s" if latex else " s"

    if defn.field_name == "false_reject_rate" and metrics.session_counts.genuine_lockouts:
        frac = f"{metrics.session_counts.genuine_lockouts}/{metrics.session_counts.genuine_sessions}"
        if latex:
            cell += r" {{\scriptsize (" + frac + r")}}"
        else:
            cell += f"  ({frac})"

    elif defn.field_name == "false_accept_rate":
        not_locked = metrics.session_counts.imposter_sessions - metrics.session_counts.imposter_lockouts
        if not_locked:
            frac = f"{not_locked}/{metrics.session_counts.imposter_sessions}"
            if latex:
                cell += r" {{\scriptsize (" + frac + r")}}"
            else:
                cell += f"  ({frac})"

    elif defn.field_name == "genuine_lockout_time.mean" and metrics.genuine_lockout_time.p90 is not None:
        p90 = f"{metrics.genuine_lockout_time.p90:.0f}"
        if latex:
            cell += r" {{\scriptsize (" + p90 + r")}}"
        else:
            cell += f"  ({p90})"

    if latex:
        cell = _latex_escape(cell)
    return cell


def print_latex_table(
    scenario_device_metrics: list[ScenarioDeviceMetrics],
    device_metrics: list[DeviceMetrics],
    scenarios: list[str],
    devices: list[str],
) -> None:
    """Print a LaTeX tabular* divided by device, with one row per scenario plus Overall."""
    table_defs = [d for d in AuthenticationMetrics.METRIC_DEFINITIONS if d.include_in_tables]
    col_spec = "@{\\extracolsep{\\fill}}ll" + "c" * len(table_defs)

    sdm_lookup = {(sdm.scenario, sdm.device): sdm.metrics for sdm in scenario_device_metrics}
    dm_lookup = {dm.device: dm.metrics for dm in device_metrics}

    print("\n% ── LaTeX table (tabular only) ──────────────────────────────")
    print(f"\\begin{{tabular*}}{{\\textwidth}}{{{col_spec}}}")
    print("\\toprule")

    headers = ["Platform", "Scenario"] + [d.short_label for d in table_defs]
    print(" & ".join(f"\\textbf{{{h}}}" for h in headers) + " \\\\")
    print("\\midrule")

    for i, device in enumerate(devices):
        if i > 0:
            print("\\midrule")

        for j, scenario in enumerate(scenarios):
            device_cell = device.capitalize() if j == 0 else ""
            scenario_cell = scenario.capitalize()

            metrics = sdm_lookup.get((scenario, device))

            if metrics:
                cells = [_fmt(metrics, d, latex=True) for d in table_defs]
            else:
                cells = ["--"] * len(table_defs)

            row = [device_cell, scenario_cell] + cells
            print(" & ".join(row) + " \\\\")

    print("\\bottomrule")
    print("\\end{tabular*}")
    print("% ─────────────────────────────────────────────────────────────\n")


def print_latex_table_devices(
    device_metrics: list[DeviceMetrics],
    devices: list[str],
) -> None:
    """Print a LaTeX tabular* with one row per device (no scenario breakdown)."""
    table_defs = [d for d in AuthenticationMetrics.METRIC_DEFINITIONS if d.include_in_tables]
    col_spec = "@{\\extracolsep{\\fill}}l" + "c" * len(table_defs)

    dm_lookup = {dm.device: dm.metrics for dm in device_metrics}

    print("\n% ── LaTeX table (tabular only) ──────────────────────────────")
    print(f"\\begin{{tabular*}}{{\\textwidth}}{{{col_spec}}}")
    print("\\toprule")

    headers = ["Platform"] + [d.short_label for d in table_defs]
    print(" & ".join(f"\\textbf{{{h}}}" for h in headers) + " \\\\")
    print("\\midrule")

    for device in devices:
        metrics = dm_lookup.get(device)
        if metrics:
            cells = [_fmt(metrics, d, latex=True) for d in table_defs]
        else:
            cells = ["--"] * len(table_defs)
        row = [device.capitalize()] + cells
        print(" & ".join(row) + " \\\\")

    print("\\bottomrule")
    print("\\end{tabular*}")
    print("% ─────────────────────────────────────────────────────────────\n")


def print_latex_table_study(metrics: AuthenticationMetrics, study_label: str) -> None:
    """Print a one-row LaTeX tabular* using thesis study-result columns."""
    table_defs = [d for d in AuthenticationMetrics.METRIC_DEFINITIONS if d.include_in_tables]
    col_spec = "@{\\extracolsep{\\fill}}l" + "c" * len(table_defs)

    print("\n% ── LaTeX table (tabular only) ──────────────────────────────")
    print(f"\\begin{{tabular*}}{{\\textwidth}}{{{col_spec}}}")
    print("\\toprule")

    headers = ["Study"] + [d.short_label for d in table_defs]
    print(" & ".join(f"\\textbf{{{h}}}" for h in headers) + " \\\\")
    print("\\midrule")

    cells = [_fmt(metrics, d, latex=True) for d in table_defs]
    print(" & ".join([study_label] + cells) + " \\\\")

    print("\\bottomrule")
    print("\\end{tabular*}")
    print("% ─────────────────────────────────────────────────────────────\n")


def print_metrics_by_scenario_and_device(scenario_device_metrics: list[ScenarioDeviceMetrics], devices: list[str]) -> None:
    """Print metrics table grouped by scenario and device."""
    print_section("METRICS BY SCENARIO AND DEVICE")

    for device in devices:
        print(f"\n{device.upper()}")
        print("-" * 40)

        device_scenario_metrics = [sdm for sdm in scenario_device_metrics if sdm.device == device]

        for sdm in device_scenario_metrics:
            print(f"\n  {sdm.scenario.upper()}")
            sdm.metrics.print_console(indent="    ")
        print()


def _format_macro_value(value: int | float | str) -> str:
    """Format a value for use in a LaTeX macro."""
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return f"{value:.2f}".rstrip("0").rstrip(".")
    return str(value)


def _genuine_recording_keys(frames: list[FrameData]) -> set[tuple[str, str]]:
    """Return unique genuine recording identifiers as (device, source_type)."""
    return {
        (frame.device, frame.source_type)
        for frame in frames
        if frame.segment_type == SegmentType.GENUINE
    }


def _genuine_recording_keys_by_device(frames: list[FrameData]) -> dict[str, set[tuple[str, str]]]:
    """Return unique genuine recording identifiers grouped by device."""
    recordings: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for frame in frames:
        if frame.segment_type == SegmentType.GENUINE:
            recordings[frame.device].add((frame.device, frame.source_type))
    return recordings


def _genuine_recording_keys_by_participant(frames: list[FrameData]) -> dict[str, set[tuple[str, str]]]:
    """Return unique genuine recording identifiers grouped by participant."""
    recordings: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for frame in frames:
        if frame.segment_type == SegmentType.GENUINE:
            recordings[frame.participant].add((frame.device, frame.source_type))
    return recordings


def _enrollment_recording_count(config: dict[str, Any], participant_count: int) -> int:
    """Return the number of enrollment recordings used by the evaluation config."""
    devices = config.get("devices", [])
    preference = config.get("enrollment", {}).get("enrollment_video_preference", {})
    scenarios = preference.get("scenarios", [])
    rotations = preference.get("rotations", [])
    return participant_count * len(devices) * len(scenarios) * len(rotations)


def _composed_video_duration_seconds(config: dict[str, Any]) -> float:
    """Return the configured duration of one composed video in seconds."""
    imposter_creation = config.get("imposter_creation", {})
    return (
        float(imposter_creation.get("genuine_user_seconds", 0))
        + float(imposter_creation.get("black_screen_seconds", 0))
        + float(imposter_creation.get("impostor_seconds", 0))
    )


def _format_duration_hours_minutes(duration_seconds: float) -> str:
    """Format a duration as rounded hours and minutes for thesis text."""
    total_minutes = round(duration_seconds / 60)
    hours, minutes = divmod(total_minutes, 60)
    return f"{hours}\\,h {minutes}\\,min"


def _study_variable_macros(data: EvaluationData, config: dict[str, Any], study: str) -> list[tuple[str, int | float | str]]:
    """Build LaTeX study-variable macros that are derivable from results and config."""
    genuine_by_participant = _genuine_recording_keys_by_participant(data.frames)
    genuine_by_device = _genuine_recording_keys_by_device(data.frames)
    composed_count = len({frame.video_path for frame in data.frames})

    if study == "controlled_study":
        participant_count = len(genuine_by_participant)
        return [
            ("ParticipantCount", participant_count),
            ("ControlledStudyUsageRecordingCount", len(_genuine_recording_keys(data.frames))),
            ("ControlledStudyDesktopRecordingCount", len(genuine_by_device.get("desktop", set()))),
            ("ControlledStudyMobileRecordingCount", len(genuine_by_device.get("mobile", set()))),
            ("ControlledComposedEvaluationVideoCount", composed_count),
            ("ControlledComposedEvaluationVideoTotalDuration", _format_duration_hours_minutes(composed_count * _composed_video_duration_seconds(config))),
            ("EnrollmentRecordingCount", _enrollment_recording_count(config, participant_count)),
        ]

    if study == "in_the_wild":
        videos_per_participant = [len(recordings) for recordings in genuine_by_participant.values()]
        matching_config = config.get("processing", {}).get("matching_strategy", {}).get("config", {})
        macros: list[tuple[str, int | float | str]] = [
            ("InTheWildParticipantCount", len(genuine_by_participant)),
            ("InTheWildUsageRecordingCount", len(_genuine_recording_keys(data.frames))),
            ("InTheWildComposedEvaluationVideoCount", composed_count),
            ("InTheWildComposedEvaluationVideoTotalDuration", _format_duration_hours_minutes(composed_count * _composed_video_duration_seconds(config))),
        ]
        if "imposters_per_genuine" in matching_config:
            macros.append(("InTheWildPairingsPerGenuineVideo", matching_config["imposters_per_genuine"]))
        if videos_per_participant:
            macros.extend([
                ("MinInTheWildRecordings", min(videos_per_participant)),
                ("MaxInTheWildRecordings", max(videos_per_participant)),
                ("MeanInTheWildRecordings", sum(videos_per_participant) / len(videos_per_participant)),
            ])
        return macros

    raise ValueError(f"Unknown study: {study}")


def print_latex_study_variables(data: EvaluationData, config_path: Path, study: str) -> list[str]:
    """Print LaTeX macros for study variables derived from results.csv and config.json."""
    with open(config_path, "r") as file:
        config = json.load(file)

    macros = _study_variable_macros(data, config, study)
    lines = [
        f"\\newcommand{{\\{name}}}{{{_format_macro_value(value)}}}"
        for name, value in macros
    ]

    print("\n% LaTeX study variables derived from results.csv and config.json")
    for line in lines:
        print(line)
    print()
    return lines

"""Console reporting utilities."""
from evaluation.shared.models import DeviceMetrics, ScenarioMetrics, ScenarioDeviceMetrics, AuthenticationMetrics, FrameData, SegmentType, MetricDefinition


def print_section(title: str) -> None:
    """Print formatted section header."""
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}\n")


def print_dataset_summary(frames: list[FrameData], total_videos: int) -> None:
    """Print summary of dataset including unique genuine videos and total sessions."""
    unique_genuine_videos = set()
    for frame in frames:
        if frame.segment_type == SegmentType.GENUINE:
            unique_genuine_videos.add(frame.source_type)

    print(f"Total user videos: {len(unique_genuine_videos)}")
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
    return text.replace("%", r"\%")


def _fmt(metrics: AuthenticationMetrics, defn: MetricDefinition, latex: bool = False) -> str:
    value = metrics._resolve_field(defn.field_name)
    if value is None:
        return "--"
    cell = f"{value:{defn.format_spec}}"

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

    elif defn.field_name == "imposter_lockout_time.mean" and metrics.imposter_lockout_time.p90 is not None:
        p90 = f"{metrics.imposter_lockout_time.p90:.0f}"
        if latex:
            cell += r" {{\scriptsize (" + p90 + r")}}"
        else:
            cell += f"  ({p90})"

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

        scenario_rows = list(scenarios) + ["Overall"]
        for j, scenario in enumerate(scenario_rows):
            is_overall = scenario == "Overall"
            device_cell = device.capitalize() if j == 0 else ""
            scenario_cell = f"\\textbf{{{scenario.capitalize()}}}" if is_overall else scenario.capitalize()

            if is_overall:
                metrics = dm_lookup.get(device)
            else:
                metrics = sdm_lookup.get((scenario, device))

            if metrics:
                cells = [_fmt(metrics, d, latex=True) for d in table_defs]
            else:
                cells = ["--"] * len(table_defs)

            if is_overall:
                cells = [f"\\textbf{{{c}}}" for c in cells]

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

"""Console reporting for an evaluation run."""
from face_auth.domain import Device
from evaluation.shared.models import (
    AuthenticationMetrics,
    DeviceMetrics,
    ScenarioDeviceMetrics,
    ScenarioMetrics,
    Session,
)


def report_never_locked_out_sessions(never_locked_out: list[str]) -> None:
    """Print a warning for imposter sessions that were never locked out."""
    if never_locked_out:
        print(f"\n⚠️  WARNING: {len(never_locked_out)} session(s) where imposter was NEVER locked out:")
        for name in never_locked_out:
            print(f"    - {name}")
        print()


def report_excluded_imposter_sessions(excluded: list[str]) -> None:
    """Print a notice for imposter sessions excluded due to an earlier lockout."""
    if excluded:
        print(
            f"\nINFO: Excluded {len(excluded)} session(s) from imposter rejection metrics "
            "because the device locked during the genuine-user segment."
        )
        print()


def print_section(title: str, width: int = 80) -> None:
    """Print a formatted section header."""
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}\n")


def print_dataset_summary(sessions: list[Session]) -> None:
    """Print how many sessions were loaded, and over which participants and devices."""
    participants = {s.participant for s in sessions}
    devices = {s.device.value for s in sessions}
    print(f"{len(sessions)} sessions | {len(participants)} participants | devices: {', '.join(sorted(devices))}")
    print()


def print_metrics(metrics: AuthenticationMetrics, indent: str = "") -> None:
    """Print one metrics group to the console."""
    counts = metrics.session_counts

    frr_frac = f"  ({counts.genuine_lockouts}/{counts.genuine_sessions})" if counts.genuine_lockouts else ""
    print(f"{indent}{'FRR (%)':<20} {metrics.false_reject_rate:.1f}{frr_frac}")

    accepted = counts.imposter_sessions - counts.imposter_lockouts
    far_frac = f"  ({accepted}/{counts.imposter_sessions})" if accepted else ""
    print(f"{indent}{'FAR (%)':<20} {metrics.false_accept_rate:.1f}{far_frac}")

    trust = "N/A" if metrics.mean_genuine_trust is None else f"{metrics.mean_genuine_trust:.2f}"
    print(f"{indent}{'Mean Genuine Trust':<20} {trust}")

    _print_time(f"{indent}{'GKT (s)':<20}", metrics.genuine_lockout_time.mean, metrics.genuine_lockout_time.p90)
    _print_time(f"{indent}{'ULT (s)':<20}", metrics.imposter_lockout_time.mean, metrics.imposter_lockout_time.p90)


def _print_time(label: str, mean: float | None, p90: float | None) -> None:
    """Print a lockout-time line as 'mean (p90)', or N/A when there is no lockout."""
    if mean is None:
        print(f"{label} N/A")
        return
    p90_suffix = f"  ({p90:.0f})" if p90 is not None else ""
    print(f"{label} {mean:.0f}{p90_suffix}")


def print_metrics_by_device(device_metrics: list[DeviceMetrics]) -> None:
    """Print metrics grouped by device."""
    print_section("METRICS BY DEVICE")
    for dm in device_metrics:
        print(dm.device.value.upper())
        print_metrics(dm.metrics, indent="  ")
        print()


def print_metrics_by_scenario(scenario_metrics: list[ScenarioMetrics]) -> None:
    """Print metrics grouped by scenario."""
    print_section("METRICS BY SCENARIO")
    for sm in scenario_metrics:
        print(sm.scenario.value.upper())
        print_metrics(sm.metrics, indent="  ")
        print()


def print_metrics_by_scenario_and_device(
    scenario_device_metrics: list[ScenarioDeviceMetrics],
    devices: list[Device],
) -> None:
    """Print metrics grouped by scenario within each device."""
    print_section("METRICS BY SCENARIO AND DEVICE")
    for device in devices:
        print(f"\n{device.value.upper()}")
        print("-" * 40)
        for sdm in (s for s in scenario_device_metrics if s.device == device):
            print(f"\n  {sdm.scenario.value.upper()}")
            print_metrics(sdm.metrics, indent="    ")
        print()

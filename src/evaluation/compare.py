"""Compare two evaluation runs side by side.

Point ``VARIANT1_RUN`` and ``VARIANT2_RUN`` at two run directories and run this
module. Each row shows one metric's value in both runs, their difference, and a
✓/✗/= indicator for whether the change is an improvement.
"""
from dataclasses import dataclass
from typing import Callable, Optional

from evaluation.shared.data_loader import load_run
from evaluation.shared.metrics import calculate_metrics_by_device
from evaluation.shared.models import AuthenticationMetrics, DeviceMetrics
from evaluation.shared.reporting import print_section
from face_auth.paths import PROJECT_ROOT


VARIANT1_RUN = PROJECT_ROOT / "data/in_the_wild/results/run_2026-08-19_09-45-47_b802"
VARIANT2_RUN = PROJECT_ROOT / "data/in_the_wild/results/run_2026-08-19_09-48-01_e2e3"

VARIANT1_NAME = "b802"
VARIANT2_NAME = "e2e3"


@dataclass(frozen=True)
class MetricSpec:
    """One comparable metric: how to label it, read it, format it, and judge it."""
    label: str
    value_of: Callable[[AuthenticationMetrics], Optional[float]]
    decimals: int
    lower_is_better: bool


METRIC_SPECS: list[MetricSpec] = [
    MetricSpec("FRR (%)", lambda m: m.false_reject_rate, 1, lower_is_better=True),
    MetricSpec("FAR (%)", lambda m: m.false_accept_rate, 1, lower_is_better=True),
    MetricSpec("Mean Genuine Trust", lambda m: m.mean_genuine_trust, 2, lower_is_better=False),
    MetricSpec("GKT mean (s)", lambda m: m.genuine_lockout_time.mean, 0, lower_is_better=False),
    MetricSpec("ULT mean (s)", lambda m: m.imposter_lockout_time.mean, 0, lower_is_better=True),
    MetricSpec("ULT P90 (s)", lambda m: m.imposter_lockout_time.p90, 0, lower_is_better=True),
]


def _fmt(value: Optional[float], decimals: int) -> str:
    return "N/A" if value is None else f"{value:.{decimals}f}"


def _fmt_diff(v1: Optional[float], v2: Optional[float], decimals: int) -> str:
    if v1 is None or v2 is None:
        return "N/A"
    return f"{v2 - v1:+.{decimals}f}"


def _indicator(v1: Optional[float], v2: Optional[float], decimals: int, lower_is_better: bool) -> str:
    """Return ✓ (improved), ✗ (regressed), = (unchanged), or blank (incomparable),
    based on the difference rounded to ``decimals`` places."""
    if v1 is None or v2 is None:
        return ""
    diff = round(v2 - v1, decimals)
    if diff == 0:
        return "="
    improved = (diff < 0) == lower_is_better
    return "✓" if improved else "✗"


def _print_metric_rows(
    metrics1: AuthenticationMetrics,
    metrics2: AuthenticationMetrics,
    group_label: str,
) -> None:
    """Print one comparison row per metric for a single device group."""
    for spec in METRIC_SPECS:
        v1 = spec.value_of(metrics1)
        v2 = spec.value_of(metrics2)
        print(
            f"{spec.label:<20} {group_label:<10} "
            f"{_fmt(v1, spec.decimals):<12} {_fmt(v2, spec.decimals):<12} "
            f"{_fmt_diff(v1, v2, spec.decimals):<10} {_indicator(v1, v2, spec.decimals, spec.lower_is_better)}"
        )
    print()


def print_device_comparison(
    device_metrics1: list[DeviceMetrics],
    device_metrics2: list[DeviceMetrics],
    variant1_name: str,
    variant2_name: str,
) -> None:
    """Print a per-device comparison of two variants over their shared devices."""
    print(f"\n{'Metric':<20} {'Device':<10} {variant1_name:<12} {variant2_name:<12} {'Diff':<10}")
    print("-" * 72)

    by_device2 = {dm.device: dm.metrics for dm in device_metrics2}
    for dm1 in device_metrics1:
        metrics2 = by_device2.get(dm1.device)
        if metrics2 is None:
            continue
        _print_metric_rows(dm1.metrics, metrics2, dm1.device.value.upper())


def main():
    """Compare two evaluation runs."""
    print_section(f"COMPARISON: {VARIANT1_NAME} vs {VARIANT2_NAME}")
    print(f"Variant 1: {VARIANT1_RUN}")
    print(f"Variant 2: {VARIANT2_RUN}")

    run1 = load_run(VARIANT1_RUN)
    run2 = load_run(VARIANT2_RUN)
    print(f"\nVariant 1: {len(run1.sessions)} sessions")
    print(f"Variant 2: {len(run2.sessions)} sessions")

    device_metrics1 = calculate_metrics_by_device(run1.sessions, run1.config.devices)
    device_metrics2 = calculate_metrics_by_device(run2.sessions, run2.config.devices)

    print_section("DEVICE METRICS COMPARISON")
    print_device_comparison(device_metrics1, device_metrics2, VARIANT1_NAME, VARIANT2_NAME)

    print_section("COMPARISON COMPLETE")
    print("\nLegend:")
    print("  ✓ = Improvement")
    print("  ✗ = Degradation")
    print("  = = No change")


if __name__ == '__main__':
    main()

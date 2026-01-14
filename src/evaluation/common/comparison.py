"""Comparison utilities for evaluation results."""
from typing import Optional

from evaluation.common.domain import DeviceMetrics, ScenarioMetrics


def format_metric_value(value: Optional[float], metric_type: str) -> str:
    """Format metric value based on type."""
    if value is None:
        return "N/A"

    if metric_type == 'risk_score':
        return f"{value:.4f}"
    elif metric_type == 'lockout':
        return f"{value:.1f}"
    else:
        return f"{value:.2f}"


def calculate_diff(value1: Optional[float], value2: Optional[float], metric_type: str) -> str:
    """Calculate and format difference between two metric values."""
    if value1 is None or value2 is None:
        return "N/A"

    diff = value2 - value1

    if metric_type == 'risk_score':
        return f"{diff:+.4f}"
    elif metric_type == 'lockout':
        return f"{diff:+.1f}"
    else:
        return f"{diff:+.2f}"


def get_diff_indicator(value1: Optional[float], value2: Optional[float], lower_is_better: bool) -> str:
    """Get indicator showing if change is improvement or regression."""
    if value1 is None or value2 is None:
        return ""

    diff = value2 - value1

    if abs(diff) < 0.001:
        return "="

    # Determine if change is improvement
    is_improvement = (diff < 0 and lower_is_better) or (diff > 0 and not lower_is_better)

    return "✓" if is_improvement else "✗"


def print_metric_row(metric_name: str, category: str, v1_val: Optional[float], v2_val: Optional[float],
                     metric_type: str, lower_is_better: bool) -> None:
    """Print a single metric comparison row."""
    v1_str = format_metric_value(v1_val, metric_type)
    v2_str = format_metric_value(v2_val, metric_type)
    diff_str = calculate_diff(v1_val, v2_val, metric_type)
    indicator = get_diff_indicator(v1_val, v2_val, lower_is_better)

    print(f"{metric_name:<20} {category:<10} {v1_str:<20} {v2_str:<20} {diff_str:<15} {indicator}")


def print_device_comparison(device_metrics1: list[DeviceMetrics], device_metrics2: list[DeviceMetrics],
                            variant1_name: str, variant2_name: str, devices: list[str]) -> None:
    """Print device metrics comparison."""
    print(f"\n{'Metric':<20} {'Device':<10} {variant1_name:<20} {variant2_name:<20} {'Diff':<15}")
    print("-" * 90)

    metric_configs = [
        ('TAR (%)', 'true_accept_rate', 'percentage', False),
        ('FRR (%)', 'false_reject_rate', 'percentage', True),
        ('TRR (%)', 'true_reject_rate', 'percentage', False),
        ('FAR (%)', 'false_accept_rate', 'percentage', True),
        ('EER (%)', 'equal_error_rate', 'percentage', True),
        ('Lockout (f)', 'imposter_lockout_time', 'lockout', False)
    ]

    for device in devices:
        v1_device = next((dm for dm in device_metrics1 if dm.device == device), None)
        v2_device = next((dm for dm in device_metrics2 if dm.device == device), None)

        if not v1_device or not v2_device:
            continue

        for metric_name, metric_attr, metric_type, lower_is_better in metric_configs:
            v1_val = getattr(v1_device.metrics, metric_attr)
            v2_val = getattr(v2_device.metrics, metric_attr)
            print_metric_row(metric_name, device.upper(), v1_val, v2_val, metric_type, lower_is_better)

        print()


def print_scenario_comparison(scenario_metrics1: list[ScenarioMetrics], scenario_metrics2: list[ScenarioMetrics],
                              variant1_name: str, variant2_name: str, scenarios: list[str]) -> None:
    """Print scenario metrics comparison."""
    print(f"\n{'Metric':<20} {'Scenario':<10} {variant1_name:<20} {variant2_name:<20} {'Diff':<15}")
    print("-" * 90)

    metric_configs = [
        ('TAR (%)', 'true_accept_rate', 'percentage', False),
        ('FRR (%)', 'false_reject_rate', 'percentage', True),
        ('TRR (%)', 'true_reject_rate', 'percentage', False),
        ('FAR (%)', 'false_accept_rate', 'percentage', True),
        ('EER (%)', 'equal_error_rate', 'percentage', True),
        ('Lockout (f)', 'imposter_lockout_time', 'lockout', False)
    ]

    for scenario in scenarios:
        v1_scenario = next((sm for sm in scenario_metrics1 if sm.scenario == scenario), None)
        v2_scenario = next((sm for sm in scenario_metrics2 if sm.scenario == scenario), None)

        if not v1_scenario or not v2_scenario:
            continue

        for metric_name, metric_attr, metric_type, lower_is_better in metric_configs:
            v1_val = getattr(v1_scenario.metrics, metric_attr)
            v2_val = getattr(v2_scenario.metrics, metric_attr)
            print_metric_row(metric_name, scenario.upper(), v1_val, v2_val, metric_type, lower_is_better)

        print()
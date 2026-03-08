"""Console reporting utilities."""
from evaluation.shared.models import DeviceMetrics, ScenarioMetrics


def print_section(title: str) -> None:
    """Print formatted section header."""
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}\n")


def print_metrics_by_device(device_metrics: list[DeviceMetrics]) -> None:
    """Print metrics table grouped by device."""
    print_section("METRICS BY DEVICE")

    for dm in device_metrics:
        m = dm.metrics
        print(f"{dm.device.upper()}")
        print(f"  TAR (True Accept Rate):        {m.true_accept_rate:6.2f}%")
        print(f"  FRR (False Reject Rate):       {m.false_reject_rate:6.2f}%")
        print(f"  TRR (True Reject Rate):        {m.true_reject_rate:6.2f}%")
        print(f"  FAR (False Accept Rate):       {m.false_accept_rate:6.2f}%")
        print(f"  EER (Equal Error Rate):        {m.equal_error_rate:6.2f}%")
        if m.imposter_lockout_time is not None:
            print(f"  Mean Imposter Lockout Time:    {m.imposter_lockout_time:6.1f}s")
        else:
            print(f"  Mean Imposter Lockout Time:    N/A")
        if m.max_lockout_time is not None:
            print(f"  Max Imposter Lockout Time:     {m.max_lockout_time:6.1f}s")
        else:
            print(f"  Max Imposter Lockout Time:     N/A")
        if m.genuine_kickout_rate is not None:
            print(f"  Genuine Kickout Rate:          {m.genuine_kickout_rate:6.1f}%")
        else:
            print(f"  Genuine Kickout Rate:          N/A")
        if m.genuine_kickout_time is not None:
            print(f"  Mean Genuine Kickout Time:     {m.genuine_kickout_time:6.1f}s")
        else:
            print(f"  Mean Genuine Kickout Time:     N/A")
        print()


def print_metrics_by_scenario(scenario_metrics: list[ScenarioMetrics]) -> None:
    """Print metrics table grouped by scenario."""
    print_section("METRICS BY SCENARIO")

    for sm in scenario_metrics:
        m = sm.metrics
        print(f"{sm.scenario.upper()}")
        print(f"  TAR (True Accept Rate):        {m.true_accept_rate:6.2f}%")
        print(f"  FRR (False Reject Rate):       {m.false_reject_rate:6.2f}%")
        print(f"  TRR (True Reject Rate):        {m.true_reject_rate:6.2f}%")
        print(f"  FAR (False Accept Rate):       {m.false_accept_rate:6.2f}%")
        print(f"  EER (Equal Error Rate):        {m.equal_error_rate:6.2f}%")
        if m.imposter_lockout_time is not None:
            print(f"  Mean Imposter Lockout Time:    {m.imposter_lockout_time:6.1f}s")
        else:
            print(f"  Mean Imposter Lockout Time:    N/A")
        if m.max_lockout_time is not None:
            print(f"  Max Imposter Lockout Time:     {m.max_lockout_time:6.1f}s")
        else:
            print(f"  Max Imposter Lockout Time:     N/A")
        if m.genuine_kickout_rate is not None:
            print(f"  Genuine Kickout Rate:          {m.genuine_kickout_rate:6.1f}%")
        else:
            print(f"  Genuine Kickout Rate:          N/A")
        if m.genuine_kickout_time is not None:
            print(f"  Mean Genuine Kickout Time:     {m.genuine_kickout_time:6.1f}s")
        else:
            print(f"  Mean Genuine Kickout Time:     N/A")
        print()

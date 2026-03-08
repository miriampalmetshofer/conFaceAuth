"""Console reporting utilities."""
from evaluation.shared.models import DeviceMetrics, ScenarioMetrics, ScenarioDeviceMetrics, AuthenticationMetrics, FrameData, SegmentType


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

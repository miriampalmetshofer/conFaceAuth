"""Compare two evaluation results."""
import sys
from pathlib import Path
import pandas as pd
import re

# Add src to path to enable imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.shared.data_loader import load_evaluation_data
from evaluation.shared.metrics import calculate_metrics_by_device, calculate_metrics_by_scenario
from evaluation.shared.comparison import print_device_comparison, print_scenario_comparison
from evaluation.shared.reporting import print_section

STUDY = 'controlled_study'

# Configure the two variants to compare
VARIANT1_PATH = Path("data/" + STUDY + "/_results_archive/V03")
VARIANT2_PATH = Path("data/" + STUDY + "/_results_archive/V04")

VARIANT1_NAME = "V03"
VARIANT2_NAME = "V04"


def main():
    """Compare two evaluation results."""
    PROJECT_ROOT = Path(__file__).parent.parent.parent

    variant1_results = PROJECT_ROOT / VARIANT1_PATH / "results.csv"
    variant2_results = PROJECT_ROOT / VARIANT2_PATH / "results.csv"

    print_section(f"COMPARISON: {VARIANT1_NAME} vs {VARIANT2_NAME}")
    print(f"Variant 1: {variant1_results}")
    print(f"Variant 2: {variant2_results}")

    # Detect if data has scenarios
    has_scenarios = _detect_has_scenarios(variant1_results)
    print(f"\nData type: {'Controlled Study (with scenarios)' if has_scenarios else 'In-the-Wild (no scenarios)'}")

    print("\nLoading data...")
    data1 = load_evaluation_data(variant1_results, parse_scenario=has_scenarios)
    data2 = load_evaluation_data(variant2_results, parse_scenario=has_scenarios)

    print(f"Variant 1: {len(data1.frames)} frames from {len(data1.videos)} videos")
    print(f"Variant 2: {len(data2.frames)} frames from {len(data2.videos)} videos")

    has_scenarios = _has_scenarios(data1) and _has_scenarios(data2)
    devices = _get_unique_devices(data1)

    print("\nCalculating metrics...")
    device_metrics1 = calculate_metrics_by_device(data1.frames, devices, data1.fps)
    device_metrics2 = calculate_metrics_by_device(data2.frames, devices, data2.fps)

    print_section("DEVICE METRICS COMPARISON")
    print_device_comparison(device_metrics1, device_metrics2, VARIANT1_NAME, VARIANT2_NAME, devices)

    if has_scenarios: # only for controlled study data
        scenarios = _get_unique_scenarios(data1)
        scenario_metrics1 = calculate_metrics_by_scenario(data1.frames, scenarios, data1.fps)
        scenario_metrics2 = calculate_metrics_by_scenario(data2.frames, scenarios, data2.fps)

        print_section("SCENARIO METRICS COMPARISON")
        print_scenario_comparison(scenario_metrics1, scenario_metrics2, VARIANT1_NAME, VARIANT2_NAME, scenarios)

    print_section("COMPARISON COMPLETE")
    print("\nLegend:")
    print("  ✓ = Improvement")
    print("  ✗ = Degradation")
    print("  = = No change")


def _has_scenarios(data) -> bool:
    """Check if data contains scenario information."""
    return any(video.scenario is not None for video in data.videos)


def _get_unique_devices(data) -> list[str]:
    """Extract unique device names from data."""
    return sorted(set(frame.device for frame in data.frames))


def _get_unique_scenarios(data) -> list[str]:
    """Extract unique scenario names from data."""
    scenarios = set()
    for video in data.videos:
        if video.scenario:
            scenarios.add(video.scenario)
    return sorted(scenarios)

def _detect_has_scenarios(csv_path: Path) -> bool:
    """Detect if results contain scenario information by checking video paths."""

    df = pd.read_csv(csv_path, nrows=100)
    video_paths = df['video_path'].unique()

    # Check if any video path has scenario format (username_scenario_date_vs_...)
    scenario_pattern = r'([^/]+)_([^_]+)_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_vs_([^_]+)_'
    for vp in video_paths:
        if re.search(scenario_pattern, vp):
            return True
    return False


if __name__ == '__main__':
    main()
"""Compare two evaluation results."""
from pathlib import Path

from evaluation.common.data_loader import load_evaluation_data
from evaluation.common.metrics import calculate_metrics_by_device, calculate_metrics_by_scenario
from evaluation.common.comparison import print_device_comparison, print_scenario_comparison
from evaluation.common.reporting import print_section


# Configure the two variants to compare
VARIANT1_PATH = Path("data/controlled_study/results_archive/arcface_small_model")
VARIANT2_PATH = Path("data/controlled_study/results_archive/threshold_1")

VARIANT1_NAME = "threshold 0.95"
VARIANT2_NAME = "threshold 1"

DEVICES = ['mobile', 'desktop']
SCENARIOS = ['easy', 'angle', 'lighting']


def main():
    """Compare two evaluation results."""
    PROJECT_ROOT = Path(__file__).parent.parent.parent

    variant1_results = PROJECT_ROOT / VARIANT1_PATH / "results.csv"
    variant2_results = PROJECT_ROOT / VARIANT2_PATH / "results.csv"

    print_section(f"COMPARISON: {VARIANT1_NAME} vs {VARIANT2_NAME}")
    print(f"Variant 1: {variant1_results}")
    print(f"Variant 2: {variant2_results}")

    print("\nLoading data...")
    data1 = load_evaluation_data(variant1_results, parse_scenario=True)
    data2 = load_evaluation_data(variant2_results, parse_scenario=True)

    print(f"Variant 1: {len(data1.frames)} frames from {len(data1.videos)} videos")
    print(f"Variant 2: {len(data2.frames)} frames from {len(data2.videos)} videos")

    print("\nCalculating metrics...")
    device_metrics1 = calculate_metrics_by_device(data1.frames, DEVICES)
    device_metrics2 = calculate_metrics_by_device(data2.frames, DEVICES)

    scenario_metrics1 = calculate_metrics_by_scenario(data1.frames, SCENARIOS)
    scenario_metrics2 = calculate_metrics_by_scenario(data2.frames, SCENARIOS)

    print_section("DEVICE METRICS COMPARISON")
    print_device_comparison(device_metrics1, device_metrics2, VARIANT1_NAME, VARIANT2_NAME, DEVICES)

    print_section("SCENARIO METRICS COMPARISON")
    print_scenario_comparison(scenario_metrics1, scenario_metrics2, VARIANT1_NAME, VARIANT2_NAME, SCENARIOS)

    print_section("COMPARISON COMPLETE")
    print("\nLegend:")
    print("  ✓ = Improvement")
    print("  ✗ = Degradation")
    print("  = = No change")


if __name__ == '__main__':
    main()
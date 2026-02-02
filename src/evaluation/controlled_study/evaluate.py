"""Evaluation orchestration for controlled study."""
import sys
from pathlib import Path

# Add src to path to enable imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from evaluation.shared.data_loader import load_evaluation_data
from evaluation.shared.metrics import (
    calculate_metrics_by_device,
    calculate_metrics_by_scenario,
    calculate_metrics_by_scenario_and_device
)
from evaluation.shared.reporting import print_section, print_metrics_by_device, print_metrics_by_scenario
from evaluation.shared.visualization import (
    create_trust_timeline_all_videos,
    create_trust_timeline_by_device,
    create_summary_visualization,
    create_combined_metrics_tables,
    create_scenario_metrics_table,
    save_html,
    save_png
)

DEVICES = ['desktop', 'mobile']
SCENARIOS = ['easy', 'angle', 'lighting']

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
RESULTS_FOLDER = "data/controlled_study/_results_archive/V02"

RESULTS_PATH = PROJECT_ROOT / RESULTS_FOLDER / "results.csv"
CONFIG_PATH = PROJECT_ROOT / RESULTS_FOLDER / "config.json"

OUTPUT_PATH = PROJECT_ROOT / "src/evaluation/controlled_study/output"


def main():
    """Run controlled study evaluation."""
    print_section("CONTROLLED STUDY EVALUATION")
    print(f"Results: {RESULTS_PATH}")
    print(f"Output: {OUTPUT_PATH}")

    data = load_evaluation_data(RESULTS_PATH, parse_scenario=True)
    print(f"Loaded {len(data.frames)} frames from {len(data.videos)} videos")

    device_metrics = calculate_metrics_by_device(data.frames, DEVICES, data.fps)
    scenario_metrics = calculate_metrics_by_scenario(data.frames, SCENARIOS, data.fps)
    scenario_device_metrics = calculate_metrics_by_scenario_and_device(data.frames, SCENARIOS, DEVICES, data.fps)

    print_metrics_by_device(device_metrics)
    print_metrics_by_scenario(scenario_metrics)

    output_files = []

    fig_all = create_trust_timeline_all_videos(data, "Controlled Study", CONFIG_PATH)
    output_files.append(save_html(fig_all, OUTPUT_PATH, 'trust_timeline_all_videos.html'))

    figs_devices = create_trust_timeline_by_device(data, DEVICES, "Controlled Study", CONFIG_PATH)
    output_files.append(save_html(figs_devices, OUTPUT_PATH, 'trust_timeline_by_device.html'))

    fig_summary = create_summary_visualization(
        data,
        device_metrics,
        'Controlled Study: Authentication Performance Summary'
    )
    output_files.append(save_png(fig_summary, OUTPUT_PATH, 'summary.png'))

    fig_tables = create_combined_metrics_tables(device_metrics, scenario_device_metrics, data.frames, data.videos)
    output_files.append(save_png(fig_tables, OUTPUT_PATH, 'metrics_tables.png'))

    fig_scenario_table = create_scenario_metrics_table(scenario_metrics)
    output_files.append(save_png(fig_scenario_table, OUTPUT_PATH, 'table_scenarios_aggregated.png'))

    print_section("EVALUATION COMPLETE")
    print(f"All outputs saved to: {OUTPUT_PATH}")
    print("\nGenerated files:")
    for file in output_files:
        print(f"  - {file.name}")


if __name__ == '__main__':
    main()

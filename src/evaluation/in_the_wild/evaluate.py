"""Evaluation orchestration for in-the-wild study."""
from pathlib import Path

from evaluation.common.data_loader import load_evaluation_data
from evaluation.common.metrics import calculate_metrics_by_device
from evaluation.common.reporting import print_section, print_metrics_by_device
from evaluation.common.visualization import (
    create_risk_timeline_all_videos,
    create_risk_timeline_by_device,
    create_summary_visualization,
    create_device_metrics_table,
    save_html,
    save_png
)


DEVICES = ['mobile', 'desktop']

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
RESULTS_PATH = PROJECT_ROOT / "data/in_the_wild/results.csv"
CONFIG_PATH = PROJECT_ROOT / "data/in_the_wild/config.json"
OUTPUT_PATH = PROJECT_ROOT / "src/evaluation/in_the_wild/output"


def main():
    """Run in-the-wild study evaluation."""
    print_section("IN-THE-WILD STUDY EVALUATION")
    print(f"Results: {RESULTS_PATH}")
    print(f"Output: {OUTPUT_PATH}")

    data = load_evaluation_data(RESULTS_PATH, parse_scenario=False)
    print(f"Loaded {len(data.frames)} frames from {len(data.videos)} videos")

    device_metrics = calculate_metrics_by_device(data.frames, DEVICES)

    print_metrics_by_device(device_metrics)

    output_files = []

    fig_all = create_risk_timeline_all_videos(data, "In-the-Wild Study", CONFIG_PATH)
    output_files.append(save_html(fig_all, OUTPUT_PATH, 'risk_timeline_all_videos.html'))

    figs_devices = create_risk_timeline_by_device(data, DEVICES, "In-the-Wild Study", CONFIG_PATH)
    output_files.append(save_html(figs_devices, OUTPUT_PATH, 'risk_timeline_by_device.html'))

    fig_summary = create_summary_visualization(
        data,
        device_metrics,
        'In-the-Wild Study: Authentication Performance Summary'
    )
    output_files.append(save_png(fig_summary, OUTPUT_PATH, 'summary.png'))

    fig_device_table = create_device_metrics_table(device_metrics)
    output_files.append(save_png(fig_device_table, OUTPUT_PATH, 'table_devices.png'))

    print_section("EVALUATION COMPLETE")
    print(f"All outputs saved to: {OUTPUT_PATH}")
    print("\nGenerated files:")
    for file in output_files:
        print(f"  - {file.name}")


if __name__ == '__main__':
    main()

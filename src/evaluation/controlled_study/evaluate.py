"""Evaluation orchestration for the controlled study."""
from face_auth.domain import Scenario
from face_auth.paths import PROJECT_ROOT
from evaluation.shared.data_loader import load_run
from evaluation.shared.metrics import (
    calculate_metrics,
    calculate_metrics_by_device,
    calculate_metrics_by_scenario,
    calculate_metrics_by_scenario_and_device,
)
from evaluation.shared.reporting import (
    print_section,
    print_metrics_by_device,
    print_metrics_by_scenario,
    print_metrics_by_scenario_and_device,
    print_dataset_summary,
)
from evaluation.shared.plots.timeline import (
    create_trust_timeline_all_videos,
    create_trust_timeline_by_device,
    create_trust_timeline_by_scenario,
    create_aggregated_trust_timeline_by_scenario,
)
from evaluation.shared.plots.summary import create_summary_visualization
from evaluation.shared.plots.save import save_html, save_png
from evaluation.controlled_study.helper.significance_tests import run_significance_tests


SCENARIOS = [Scenario.EASY, Scenario.ANGLE, Scenario.LIGHTING]

RUN_DIR = PROJECT_ROOT / "data/controlled_study/results/run_2026-08-19_21-41-02_7ab5"
OUTPUT_PATH = PROJECT_ROOT / "src/evaluation/controlled_study/output"


def main():
    """Run the controlled study evaluation."""
    print_section("CONTROLLED STUDY EVALUATION")
    print(f"Run: {RUN_DIR}")
    print(f"Output: {OUTPUT_PATH}")

    run = load_run(RUN_DIR)
    sessions = run.sessions
    config = run.config
    print(f"\nLoaded {len(sessions)} sessions from {RUN_DIR.name}")

    overall_metrics = calculate_metrics(sessions)
    device_metrics = calculate_metrics_by_device(sessions, config.devices)
    scenario_metrics = calculate_metrics_by_scenario(sessions, SCENARIOS)
    scenario_device_metrics = calculate_metrics_by_scenario_and_device(sessions, SCENARIOS, config.devices)

    print_dataset_summary(sessions)
    print_metrics_by_device(device_metrics)
    print_metrics_by_scenario(scenario_metrics)
    print_metrics_by_scenario_and_device(scenario_device_metrics, config.devices)
    run_significance_tests(run, SCENARIOS, config.devices)

    output_files = []

    fig_all = create_trust_timeline_all_videos(sessions, config, "Controlled Study", overall_metrics)
    output_files.append(save_html(fig_all, OUTPUT_PATH, 'trust_timeline_all_videos.html'))

    figs_devices = create_trust_timeline_by_device(sessions, config, config.devices, "Controlled Study")
    output_files.append(save_html(figs_devices, OUTPUT_PATH, 'trust_timeline_by_device.html'))

    figs_scenarios = create_trust_timeline_by_scenario(sessions, config, SCENARIOS, "Controlled Study")
    output_files.append(save_html(figs_scenarios, OUTPUT_PATH, 'trust_timeline_by_scenario.html'))

    fig_aggregated = create_aggregated_trust_timeline_by_scenario(sessions, config, SCENARIOS, "Controlled Study")
    output_files.append(save_html(fig_aggregated, OUTPUT_PATH, 'trust_timeline_scenarios_aggregated.html'))

    fig_summary = create_summary_visualization(
        sessions,
        config,
        device_metrics,
        'Controlled Study: Authentication Performance Summary',
    )
    output_files.append(save_png(fig_summary, OUTPUT_PATH, 'summary.png'))

    print_section("EVALUATION COMPLETE")
    print(f"All outputs saved to: {OUTPUT_PATH}")
    print("\nGenerated files:")
    for file in output_files:
        print(f"  - {file.name}")


if __name__ == '__main__':
    main()

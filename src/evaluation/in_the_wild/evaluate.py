"""Evaluation orchestration for the in-the-wild study."""
from evaluation.shared.data_loader import load_run
from evaluation.shared.metrics import calculate_metrics, calculate_metrics_by_device
from evaluation.shared.reporting import (
    print_section,
    print_metrics_by_device,
    print_dataset_summary,
)
from evaluation.shared.plots.timeline import (
    create_trust_timeline_all_videos,
    create_trust_timeline_by_device,
)
from evaluation.shared.plots.summary import create_summary_visualization
from evaluation.shared.plots.save import save_html, save_png
from evaluation.in_the_wild.helper.annotation_validator import (
    validate_annotations,
    print_validation_results,
)
from face_auth.paths import PROJECT_ROOT


RUN_DIR = PROJECT_ROOT / "data/in_the_wild/results/run_2026-08-19_09-48-01_e2e3"

ANNOTATIONS_PATH = PROJECT_ROOT / "data/in_the_wild"
OUTPUT_PATH = PROJECT_ROOT / "src/evaluation/in_the_wild/output"


def main():
    """Run the in-the-wild study evaluation."""
    print_section("IN-THE-WILD STUDY EVALUATION")
    print(f"Run: {RUN_DIR}")
    print(f"Output: {OUTPUT_PATH}")

    print_section("VALIDATING ANNOTATIONS")
    validation_result = validate_annotations(ANNOTATIONS_PATH)
    print_validation_results(validation_result)

    if validation_result.invalid_files:
        print("\n⚠️  Warning: Some annotation files are invalid. Please fix them before proceeding.")
        print("Continuing with evaluation using valid annotations only.\n")

    run = load_run(RUN_DIR)
    print(f"\nLoaded {len(run.sessions)} sessions from {RUN_DIR.name}")

    sessions = run.sessions
    config = run.config

    overall_metrics = calculate_metrics(sessions)
    device_metrics = calculate_metrics_by_device(sessions, config.devices)

    print_dataset_summary(sessions)
    print_metrics_by_device(device_metrics)

    output_files = []

    fig_all = create_trust_timeline_all_videos(sessions, config, "In-the-Wild Study", overall_metrics)
    output_files.append(save_html(fig_all, OUTPUT_PATH, 'trust_timeline_all_videos.html'))

    figs_devices = create_trust_timeline_by_device(sessions, config, config.devices, "In-the-Wild Study")
    output_files.append(save_html(figs_devices, OUTPUT_PATH, 'trust_timeline_by_device.html'))

    fig_summary = create_summary_visualization(
        sessions,
        config,
        device_metrics,
        'In-the-Wild Study: Authentication Performance Summary',
    )
    output_files.append(save_png(fig_summary, OUTPUT_PATH, 'summary.png'))

    print_section("EVALUATION COMPLETE")
    print(f"All outputs saved to: {OUTPUT_PATH}")
    print("\nGenerated files:")
    for file in output_files:
        print(f"  - {file.name}")


if __name__ == '__main__':
    main()

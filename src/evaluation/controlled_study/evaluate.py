"""Evaluation script for controlled study."""
import matplotlib.pyplot as plt
from pathlib import Path

from evaluation.shared.loader import load_results_csv, load_annotations, load_annotation_schema
from evaluation.shared.processor import categorize_frames, add_grouping_columns
from evaluation.shared.metrics import (
    calculate_overall_metrics,
    calculate_segment_metrics,
    calculate_grouped_metrics,
    calculate_device_comparison
)
from evaluation.shared.visualizer import (
    create_risk_score_timeline,
    create_risk_score_timeline_with_categories,
    create_scenario_aggregated_timeline,
    save_interactive_plot,
    create_grouped_comparison_plot,
    create_device_breakdown_plot
)
from evaluation.shared.reporter import (
    print_section,
    print_overall_summary,
    print_segment_analysis,
    print_grouped_analysis,
    save_plot
)

# Study configuration
def get_project_root() -> Path:
    """Get project root directory."""
    # From src/evaluation/controlled_study/evaluate.py -> go up 3 levels to project root
    return Path(__file__).parent.parent.parent.parent

PROJECT_ROOT = get_project_root()
RESULTS_PATH = PROJECT_ROOT / "data/controlled_study/results.csv"
ANNOTATIONS_PATH = PROJECT_ROOT / "data/annotations"
ANNOTATION_SCHEMA_PATH = PROJECT_ROOT / "src/evaluation/controlled_study/annotation_schema.json"
OUTPUT_PATH = PROJECT_ROOT / "src/evaluation/controlled_study/output"
GROUPING_DIMENSIONS = ["device", "scenario"]


def main():
    """Run controlled study evaluation."""
    # Print study header
    print_section("CONTROLLED STUDY EVALUATION")
    print(f"Results: {RESULTS_PATH}")
    print(f"Annotations: {ANNOTATIONS_PATH}")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Grouping dimensions: {', '.join(GROUPING_DIMENSIONS)}")

    # Load data
    results_df = load_results_csv(RESULTS_PATH)
    annotations_df = None
    if ANNOTATIONS_PATH.exists():
        annotations_df = load_annotations(ANNOTATIONS_PATH, ANNOTATION_SCHEMA_PATH)

    # Categorize frames by segment type
    results_df = categorize_frames(results_df)

    # Merge annotations for grouping
    results_df = add_grouping_columns(
        results_df,
        GROUPING_DIMENSIONS,
        annotations_df
    )

    # Calculate overall metrics
    overall_metrics = calculate_overall_metrics(results_df)
    print_overall_summary(overall_metrics)

    # Calculate segment metrics
    segment_analysis = calculate_segment_metrics(results_df)
    print_segment_analysis(segment_analysis)

    # Calculate device-specific metrics
    device_metrics = calculate_device_comparison(results_df, 'device')
    if device_metrics.groups:
        print_grouped_analysis(device_metrics)

    # Calculate scenario-specific metrics
    scenario_metrics = calculate_grouped_metrics(results_df, 'scenario')
    if scenario_metrics.groups:
        print_grouped_analysis(scenario_metrics)

    # Generate visualizations
    output_files = []

    threshold = results_df['threshold'].iloc[0]

    # Interactive risk score timeline with category grouping
    fig_timeline_categories = create_risk_score_timeline_with_categories(results_df, threshold, category_column='scenario')
    output_files.append(
        save_interactive_plot(fig_timeline_categories, OUTPUT_PATH, 'risk_score_timeline_by_scenario.html')
    )

    # Interactive risk score timeline (all videos, original version)
    fig_timeline = create_risk_score_timeline(results_df, threshold)
    output_files.append(
        save_interactive_plot(fig_timeline, OUTPUT_PATH, 'risk_score_timeline_all.html')
    )

    # Scenario aggregated timeline
    if 'scenario' in results_df.columns:
        fig_scenario_timeline = create_scenario_aggregated_timeline(results_df, threshold)
        output_files.append(
            save_interactive_plot(fig_scenario_timeline, OUTPUT_PATH, 'scenario_aggregated_timeline.html')
        )

    # Device comparison plot
    if device_metrics.groups:
        fig_device_comparison = create_grouped_comparison_plot(device_metrics)
        output_files.append(
            save_interactive_plot(fig_device_comparison, OUTPUT_PATH, 'device_comparison.html')
        )

        # Device breakdown plot
        fig_device_breakdown = create_device_breakdown_plot(device_metrics)
        output_files.append(
            save_interactive_plot(fig_device_breakdown, OUTPUT_PATH, 'device_breakdown.html')
        )

    # Scenario comparison plot
    if scenario_metrics.groups:
        fig_scenario_comparison = create_grouped_comparison_plot(scenario_metrics)
        output_files.append(
            save_interactive_plot(fig_scenario_comparison, OUTPUT_PATH, 'scenario_comparison.html')
        )

    # Static summary plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Overall state distribution
    genuine_state = results_df[results_df['segment_type'] == 'genuine']['predicted_state'].value_counts()
    axes[0, 0].pie(genuine_state.values, labels=genuine_state.index, autopct='%1.1f%%',
                  colors=['#2ecc71', '#e74c3c'], startangle=90)
    axes[0, 0].set_title('Genuine Segment State\n(Should be Unlocked)')

    imposter_state = results_df[results_df['segment_type'] == 'imposter']['predicted_state'].value_counts()
    axes[0, 1].pie(imposter_state.values, labels=imposter_state.index, autopct='%1.1f%%',
                  colors=['#2ecc71', '#e74c3c'], startangle=90)
    axes[0, 1].set_title('Imposter Segment State\n(Should be Locked)')

    # Risk score distribution
    genuine_risk = results_df[results_df['segment_type'] == 'genuine']['risk_score']
    imposter_risk = results_df[results_df['segment_type'] == 'imposter']['risk_score']
    axes[1, 0].hist([genuine_risk, imposter_risk], bins=30, label=['Genuine', 'Imposter'],
                   color=['green', 'red'], alpha=0.6, edgecolor='black')
    axes[1, 0].axvline(threshold, color='black', linestyle='--', linewidth=2, label='Threshold')
    axes[1, 0].set_xlabel('Risk Score')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Risk Score Distribution')
    axes[1, 0].legend()
    axes[1, 0].grid(axis='y', alpha=0.3)

    # Error rates by device
    if device_metrics.groups:
        devices = list(device_metrics.groups.keys())
        far_vals = [device_metrics.groups[d].imposter.false_accept_rate for d in devices]
        frr_vals = [device_metrics.groups[d].genuine.false_reject_rate for d in devices]
        eer_vals = [(far + frr) / 2 for far, frr in zip(far_vals, frr_vals)]

        x = range(len(devices))
        width = 0.25
        axes[1, 1].bar([i - width for i in x], far_vals, width, label='FAR', color='#e74c3c', alpha=0.7)
        axes[1, 1].bar(x, frr_vals, width, label='FRR', color='#f39c12', alpha=0.7)
        axes[1, 1].bar([i + width for i in x], eer_vals, width, label='EER', color='#3498db', alpha=0.7)
        axes[1, 1].set_ylabel('Error Rate (%)')
        axes[1, 1].set_xlabel('Device Type')
        axes[1, 1].set_title('Error Rates by Device')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(devices)
        axes[1, 1].legend()
        axes[1, 1].grid(axis='y', alpha=0.3)

    plt.suptitle('Controlled Study: Authentication Performance Summary', fontsize=16, y=0.995)
    plt.tight_layout()

    output_files.append(save_plot(fig, OUTPUT_PATH, 'summary.png'))
    plt.close()

    # Print completion summary
    print_section("EVALUATION COMPLETE")
    print(f"All outputs saved to: {OUTPUT_PATH}")
    print("\nGenerated files:")
    for file in output_files:
        print(f"  - {file}")


if __name__ == '__main__':
    main()

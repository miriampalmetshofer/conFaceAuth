"""Evaluation script for controlled study."""
import matplotlib.pyplot as plt

from evaluation.controlled_study.config import CONTROLLED_STUDY_CONFIG
from evaluation.shared.loader import load_study_data
from evaluation.shared.processor import categorize_frames, add_grouping_columns
from evaluation.shared.metrics import (
    calculate_overall_metrics,
    calculate_segment_metrics,
    calculate_grouped_metrics,
    calculate_device_comparison
)
from evaluation.shared.visualizer import (
    create_risk_score_timeline,
    create_scenario_aggregated_timeline,
    save_interactive_plot,
    create_grouped_comparison_plot,
    create_device_breakdown_plot
)
from evaluation.shared.reporter import (
    print_study_header,
    print_overall_summary,
    print_segment_analysis,
    print_grouped_analysis,
    save_plot,
    print_completion_summary
)


def main():
    """Run controlled study evaluation.shared."""
    config = CONTROLLED_STUDY_CONFIG

    # Print study header
    print_study_header(config)

    # Load data
    results_df, stitch_config, annotations_df = load_study_data(config)

    # Categorize frames by segment type
    results_df = categorize_frames(
        results_df,
        fps=stitch_config.fps,
        genuine_seconds=stitch_config.genuine_user_seconds,
        black_seconds=stitch_config.black_screen_seconds,
        imposter_seconds=stitch_config.impostor_seconds
    )

    # Merge annotations for grouping
    results_df = add_grouping_columns(
        results_df,
        config.grouping_dimensions,
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

    # Interactive risk score timeline (all videos)
    threshold = results_df['threshold'].iloc[0]
    fig_timeline = create_risk_score_timeline(results_df, threshold)
    output_files.append(
        save_interactive_plot(fig_timeline, config.output_path, 'risk_score_timeline.html')
    )

    # Scenario aggregated timeline
    if 'scenario' in results_df.columns:
        fig_scenario_timeline = create_scenario_aggregated_timeline(results_df, threshold)
        output_files.append(
            save_interactive_plot(fig_scenario_timeline, config.output_path, 'scenario_aggregated_timeline.html')
        )

    # Device comparison plot
    if device_metrics.groups:
        fig_device_comparison = create_grouped_comparison_plot(device_metrics)
        output_files.append(
            save_interactive_plot(fig_device_comparison, config.output_path, 'device_comparison.html')
        )

        # Device breakdown plot
        fig_device_breakdown = create_device_breakdown_plot(device_metrics)
        output_files.append(
            save_interactive_plot(fig_device_breakdown, config.output_path, 'device_breakdown.html')
        )

    # Scenario comparison plot
    if scenario_metrics.groups:
        fig_scenario_comparison = create_grouped_comparison_plot(scenario_metrics)
        output_files.append(
            save_interactive_plot(fig_scenario_comparison, config.output_path, 'scenario_comparison.html')
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

    output_files.append(save_plot(fig, config.output_path, 'summary.png'))
    plt.close()

    # Print completion summary
    print_completion_summary(config, output_files)


if __name__ == '__main__':
    main()

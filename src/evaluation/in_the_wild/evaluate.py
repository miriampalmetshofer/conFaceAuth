"""Evaluation script for in-the-wild study."""
import matplotlib.pyplot as plt

from evaluation.in_the_wild.config import WILD_STUDY_CONFIG
from evaluation.shared.loader import load_study_data, load_annotation_schema
from evaluation.shared.processor import categorize_frames, add_grouping_columns
from evaluation.shared.metrics import (
    calculate_overall_metrics,
    calculate_segment_metrics,
    calculate_grouped_metrics
)
from evaluation.shared.visualizer import (
    create_risk_score_timeline,
    save_interactive_plot,
    create_grouped_comparison_plot
)
from evaluation.shared.reporter import (
    print_study_header,
    print_overall_summary,
    print_segment_analysis,
    print_grouped_analysis,
    save_plot,
    print_completion_summary,
    print_section
)


def main():
    """Run in-the-wild study evaluation.shared."""
    config = WILD_STUDY_CONFIG

    # Print study header
    print_study_header(config)

    # Load data
    results_df, stitch_config, annotations_df = load_study_data(config)

    # Categorize frames by segment type
    results_df = categorize_frames(results_df)

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

    # Calculate environment-specific metrics
    environment_metrics = calculate_grouped_metrics(results_df, 'environment')
    if environment_metrics.groups:
        print_grouped_analysis(environment_metrics)

    # Generate visualizations
    output_files = []

    # Interactive risk score timeline
    threshold = results_df['threshold'].iloc[0]
    fig_timeline = create_risk_score_timeline(results_df, threshold)
    output_files.append(
        save_interactive_plot(fig_timeline, config.output_path, 'risk_score_timeline.html')
    )

    # Environment comparison plot
    if environment_metrics.groups:
        fig_env_comparison = create_grouped_comparison_plot(environment_metrics)
        output_files.append(
            save_interactive_plot(fig_env_comparison, config.output_path, 'environment_comparison.html')
        )

    # Static comprehensive analysis plot
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))

    # Row 1: State distributions (pie charts)
    # Overall state distribution
    state_counts = results_df['predicted_state'].value_counts()
    axes[0, 0].pie(state_counts.values, labels=state_counts.index, autopct='%1.1f%%',
                  colors=['#2ecc71', '#e74c3c'], startangle=90)
    axes[0, 0].set_title('Overall State Distribution')

    # Genuine segment state
    genuine_state = results_df[results_df['segment_type'] == 'genuine']['predicted_state'].value_counts()
    axes[0, 1].pie(genuine_state.values, labels=genuine_state.index, autopct='%1.1f%%',
                  colors=['#2ecc71', '#e74c3c'], startangle=90)
    axes[0, 1].set_title('Genuine Segment State\n(Should be Unlocked)')

    # Imposter segment state
    imposter_state = results_df[results_df['segment_type'] == 'imposter']['predicted_state'].value_counts()
    axes[0, 2].pie(imposter_state.values, labels=imposter_state.index, autopct='%1.1f%%',
                  colors=['#2ecc71', '#e74c3c'], startangle=90)
    axes[0, 2].set_title('Imposter Segment State\n(Should be Locked)')

    # Row 2: Distributions
    # State distribution by segment (bar)
    segment_state_data = results_df.groupby(['segment_type', 'predicted_state']).size().unstack(fill_value=0)
    segment_state_data.plot(kind='bar', ax=axes[1, 0], color=['#e74c3c', '#2ecc71'])
    axes[1, 0].set_title('Authentication State by Segment Type')
    axes[1, 0].set_ylabel('Frame Count')
    axes[1, 0].set_xlabel('Segment Type')
    axes[1, 0].legend(title='State', loc='upper right')
    axes[1, 0].grid(axis='y', alpha=0.3)
    axes[1, 0].tick_params(axis='x', rotation=45)

    # Risk score distribution by segment
    genuine_risk = results_df[results_df['segment_type'] == 'genuine']['risk_score']
    imposter_risk = results_df[results_df['segment_type'] == 'imposter']['risk_score']
    axes[1, 1].hist([genuine_risk, imposter_risk], bins=30, label=['Genuine', 'Imposter'],
                   color=['green', 'red'], alpha=0.6, edgecolor='black')
    axes[1, 1].axvline(threshold, color='black', linestyle='--', linewidth=2, label='Threshold')
    axes[1, 1].set_xlabel('Risk Score')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Risk Score Distribution')
    axes[1, 1].legend()
    axes[1, 1].grid(axis='y', alpha=0.3)

    # Distance distribution by segment
    genuine_dist = results_df[results_df['segment_type'] == 'genuine']['distance']
    imposter_dist = results_df[results_df['segment_type'] == 'imposter']['distance']
    axes[1, 2].hist([genuine_dist, imposter_dist], bins=30, label=['Genuine', 'Imposter'],
                   color=['green', 'red'], alpha=0.6, edgecolor='black')
    axes[1, 2].set_xlabel('Distance')
    axes[1, 2].set_ylabel('Frequency')
    axes[1, 2].set_title('Face Distance Distribution')
    axes[1, 2].legend()
    axes[1, 2].grid(axis='y', alpha=0.3)

    # Row 3: Performance metrics
    # Authentication performance rates
    categories = ['True\nAccept', 'False\nReject', 'False\nAccept', 'True\nReject']
    rates = [
        segment_analysis.genuine.true_accept_rate,
        segment_analysis.genuine.false_reject_rate,
        segment_analysis.imposter.false_accept_rate,
        segment_analysis.imposter.true_reject_rate
    ]
    colors_bars = ['#2ecc71', '#e74c3c', '#e74c3c', '#2ecc71']

    axes[2, 0].bar(categories, rates, color=colors_bars, alpha=0.7, edgecolor='black')
    axes[2, 0].set_ylabel('Rate (%)')
    axes[2, 0].set_title('Authentication Performance Rates')
    axes[2, 0].grid(axis='y', alpha=0.3)
    axes[2, 0].set_ylim([0, 105])

    # Add value labels on bars
    for i, v in enumerate(rates):
        axes[2, 0].text(i, v + 2, f'{v:.1f}%', ha='center', va='bottom', fontweight='bold')

    # Average metrics comparison
    metrics_names = ['Avg Risk Score', 'Avg Distance']
    genuine_metrics_vals = [
        segment_analysis.genuine.avg_risk_score,
        segment_analysis.genuine.avg_distance
    ]
    imposter_metrics_vals = [
        segment_analysis.imposter.avg_risk_score,
        segment_analysis.imposter.avg_distance
    ]

    x = range(len(metrics_names))
    width = 0.35
    axes[2, 1].bar([i - width/2 for i in x], genuine_metrics_vals, width,
                   label='Genuine', color='green', alpha=0.7, edgecolor='black')
    axes[2, 1].bar([i + width/2 for i in x], imposter_metrics_vals, width,
                   label='Imposter', color='red', alpha=0.7, edgecolor='black')
    axes[2, 1].set_ylabel('Value')
    axes[2, 1].set_title('Average Metrics Comparison')
    axes[2, 1].set_xticks(x)
    axes[2, 1].set_xticklabels(metrics_names)
    axes[2, 1].legend()
    axes[2, 1].grid(axis='y', alpha=0.3)

    # Error rates summary
    error_types = ['False Accept\nRate (FAR)', 'False Reject\nRate (FRR)', 'Equal Error\nRate (EER)']
    error_rates = segment_analysis.error_rates
    error_values = [error_rates.false_accept_rate, error_rates.false_reject_rate, error_rates.equal_error_rate]

    axes[2, 2].bar(error_types, error_values, color=['#e74c3c', '#f39c12', '#3498db'],
                  alpha=0.7, edgecolor='black')
    axes[2, 2].set_ylabel('Error Rate (%)')
    axes[2, 2].set_title('Security Error Rates')
    axes[2, 2].grid(axis='y', alpha=0.3)
    axes[2, 2].set_ylim([0, max(error_values) * 1.3 if max(error_values) > 0 else 10])

    # Add value labels
    for i, v in enumerate(error_values):
        axes[2, 2].text(i, v + max(error_values) * 0.05, f'{v:.2f}%',
                       ha='center', va='bottom', fontweight='bold')

    plt.suptitle('In-the-Wild Study: Imposter Attack Analysis', fontsize=16, y=0.995)
    plt.tight_layout()

    output_files.append(save_plot(fig, config.output_path, 'comprehensive_analysis.png'))
    plt.close()

    # Generate annotation distribution plots if annotations exist
    if annotations_df is not None and len(annotations_df) > 0:
        print_section("GENERATING ANNOTATION PLOTS")

        annotation_schema = load_annotation_schema(config.schema_path)
        list_cols = annotation_schema['list_fields']
        categorical_cols = annotation_schema['single_choice_fields']
        cols_to_plot = [col for col in list_cols + categorical_cols if col in annotations_df.columns]

        num_cols = 3
        num_rows = (len(cols_to_plot) + num_cols - 1) // num_cols

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(16, 4 * num_rows))
        if num_rows == 1:
            axes = axes.reshape(1, -1)

        axes = axes.flatten()

        for ax, col in zip(axes, cols_to_plot):
            data = (
                annotations_df[col].explode()
                if col in list_cols
                else annotations_df[col]
            )

            data.value_counts().plot(kind='bar', ax=ax)
            ax.set_title(col.replace("_", " ").title())
            ax.set_ylabel("Count")
            ax.grid(axis='y', alpha=0.3)
            ax.tick_params(axis='x', rotation=45)

        # Hide unused subplots
        for ax in axes[len(cols_to_plot):]:
            ax.set_visible(False)

        plt.suptitle("In-the-Wild Study: Annotation Distribution", fontsize=16, y=0.995)
        plt.tight_layout()

        output_files.append(save_plot(fig, config.output_path, 'annotation_distribution.png'))
        plt.close()

    # Print completion summary
    print_completion_summary(config, output_files)


if __name__ == '__main__':
    main()

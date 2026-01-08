"""Unified reporting utilities for evaluation studies."""
import os
from pathlib import Path
import matplotlib.pyplot as plt

from evaluation.shared.models import (
    OverallMetrics,
    SegmentAnalysis,
    GroupedMetrics
)


def print_section(title: str, width: int = 80):
    """Print a section header with separator lines."""
    print("\n" + "="*width)
    print(title)
    print("="*width)


def print_overall_summary(metrics: OverallMetrics):
    """Print overall metrics summary."""
    print_section("OVERALL PERFORMANCE METRICS")
    print(f"Total Frames: {metrics.total_frames}")
    print(f"Unlocked Frames: {metrics.unlocked_frames} ({metrics.unlock_rate:.2f}%)")
    print(f"Locked Frames: {metrics.locked_frames} ({metrics.lock_rate:.2f}%)")
    print(f"\nRisk Score Statistics:")
    print(f"  Mean: {metrics.avg_risk_score:.4f}")
    print(f"  Median: {metrics.median_risk_score:.4f}")
    print(f"  Std Dev: {metrics.std_risk_score:.4f}")
    print(f"\nDistance Statistics:")
    print(f"  Mean: {metrics.avg_distance:.4f}")
    print(f"  Median: {metrics.median_distance:.4f}")
    print(f"  Std Dev: {metrics.std_distance:.4f}")
    print(f"\nState Changes: {metrics.state_changes}")
    print(f"Avg Frames Between Changes: {metrics.avg_frames_between_changes:.2f}")


def print_segment_analysis(analysis: SegmentAnalysis):
    """Print genuine vs imposter segment analysis."""
    print_section("GENUINE vs IMPOSTER SEGMENT ANALYSIS")

    genuine = analysis.genuine
    print("\nGenuine User Segments (Should be UNLOCKED):")
    print("-" * 60)
    print(f"  Total Frames: {genuine.total_frames}")
    print(f"  True Accept (Unlocked): {genuine.unlocked_frames} ({genuine.true_accept_rate:.2f}%)")
    print(f"  False Reject (Locked): {genuine.locked_frames} ({genuine.false_reject_rate:.2f}%)")
    print(f"  Avg Risk Score: {genuine.avg_risk_score:.4f}")
    print(f"  Avg Distance: {genuine.avg_distance:.4f}")

    imposter = analysis.imposter
    print("\nImposter Segments (Should be LOCKED):")
    print("-" * 60)
    print(f"  Total Frames: {imposter.total_frames}")
    print(f"  True Reject (Locked): {imposter.locked_frames} ({imposter.true_reject_rate:.2f}%)")
    print(f"  False Accept (Unlocked): {imposter.unlocked_frames} ({imposter.false_accept_rate:.2f}%)")
    print(f"  Avg Risk Score: {imposter.avg_risk_score:.4f}")
    print(f"  Avg Distance: {imposter.avg_distance:.4f}")

    # Combined error rates
    error_rates = analysis.error_rates
    print("\nCombined Error Rates:")
    print("-" * 60)
    print(f"  False Accept Rate (FAR): {error_rates.false_accept_rate:.2f}%")
    print(f"  False Reject Rate (FRR): {error_rates.false_reject_rate:.2f}%")
    print(f"  Approximate EER: {error_rates.equal_error_rate:.2f}%")


def print_grouped_analysis(grouped_metrics: GroupedMetrics):
    """Print analysis grouped by a dimension (device, scenario, environment, etc.).

    Args:
        grouped_metrics: GroupedMetrics dataclass
    """
    print_section(f"ANALYSIS BY {grouped_metrics.dimension_name.upper()}")

    for group_name, analysis in grouped_metrics.groups.items():
        print(f"\n{grouped_metrics.dimension_name.title()}: {group_name}")
        print("-" * 60)

        genuine = analysis.genuine
        imposter = analysis.imposter
        error_rates = analysis.error_rates

        print(f"  Genuine: TAR={genuine.true_accept_rate:.2f}%, "
              f"FRR={genuine.false_reject_rate:.2f}%, "
              f"Avg Risk={genuine.avg_risk_score:.4f}")

        print(f"  Imposter: TRR={imposter.true_reject_rate:.2f}%, "
              f"FAR={imposter.false_accept_rate:.2f}%, "
              f"Avg Risk={imposter.avg_risk_score:.4f}")

        print(f"  EER: {error_rates.equal_error_rate:.2f}%")


def save_plot(fig: plt.Figure, output_folder: Path, filename: str, dpi: int = 300):
    """Save matplotlib figure to file."""
    output_folder = Path(output_folder)
    os.makedirs(output_folder, exist_ok=True)
    output_file = output_folder / filename
    fig.savefig(output_file, dpi=dpi, bbox_inches='tight')
    print(f"Saved plot to {output_file}")
    return output_file

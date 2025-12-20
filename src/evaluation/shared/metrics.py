"""Metrics calculation utilities for authentication evaluation.shared."""
import pandas as pd
from typing import Dict
from dataclasses import asdict

from evaluation.shared.models import (
    GenuineSegmentMetrics,
    ImposterSegmentMetrics,
    OverallMetrics,
    ErrorRates,
    VideoMetrics,
    SegmentAnalysis,
    GroupedMetrics
)


def print_metrics_table(metrics: dict, title: str = "Metrics"):
    """Print metrics in a formatted table.

    Args:
        metrics: Dictionary of metrics
        title: Title for the metrics section
    """
    print(f"\n{title}")
    print("-" * 60)

    for key, value in metrics.items():
        if isinstance(value, float):
            if 'rate' in key.lower() or 'percentage' in key.lower():
                print(f"  {key.replace('_', ' ').title()}: {value:.2f}%")
            else:
                print(f"  {key.replace('_', ' ').title()}: {value:.4f}")
        else:
            print(f"  {key.replace('_', ' ').title()}: {value}")


def calculate_overall_metrics(df: pd.DataFrame) -> OverallMetrics:
    """Calculate overall authentication performance metrics.

    Args:
        df: Results dataframe

    Returns:
        OverallMetrics dataclass
    """
    total_frames = len(df)
    unlocked_frames = len(df[df['predicted_state'] == 'Unlocked'])
    locked_frames = len(df[df['predicted_state'] == 'Locked'])

    state_changes = (df['predicted_state'] != df['predicted_state'].shift()).sum() - 1

    return OverallMetrics(
        total_frames=total_frames,
        unlocked_frames=unlocked_frames,
        locked_frames=locked_frames,
        unlock_rate=(unlocked_frames / total_frames) * 100,
        lock_rate=(locked_frames / total_frames) * 100,
        avg_risk_score=df['risk_score'].mean(),
        median_risk_score=df['risk_score'].median(),
        std_risk_score=df['risk_score'].std(),
        avg_distance=df['distance'].mean(),
        median_distance=df['distance'].median(),
        std_distance=df['distance'].std(),
        state_changes=state_changes,
        avg_frames_between_changes=total_frames / max(state_changes, 1)
    )


def calculate_false_rates(genuine_metrics: dict, imposter_metrics: dict) -> Dict:
    """Calculate False Accept Rate (FAR) and False Reject Rate (FRR).

    Args:
        genuine_metrics: Metrics for genuine user segments
        imposter_metrics: Metrics for imposter segments

    Returns:
        Dictionary with FAR and FRR
    """
    far = imposter_metrics.get('false_accept_rate', 0)
    frr = genuine_metrics.get('false_reject_rate', 0)

    return {
        'false_accept_rate': far,
        'false_reject_rate': frr,
        'equal_error_rate': (far + frr) / 2  # Approximate EER
    }


def calculate_video_level_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate metrics per video.

    Args:
        df: Results dataframe

    Returns:
        DataFrame with per-video metrics
    """
    video_metrics = []

    for video_path in df['video_path'].unique():
        video_data = df[df['video_path'] == video_path]

        total_frames = len(video_data)
        unlocked_frames = len(video_data[video_data['predicted_state'] == 'Unlocked'])

        metrics = {
            'video_path': video_path,
            'total_frames': total_frames,
            'unlocked_frames': unlocked_frames,
            'locked_frames': total_frames - unlocked_frames,
            'unlock_rate': (unlocked_frames / total_frames) * 100,
            'avg_risk_score': video_data['risk_score'].mean(),
            'avg_distance': video_data['distance'].mean(),
            'face_detection_rate': (video_data['face_detected'].sum() / total_frames) * 100
        }

        # Add segment-specific metrics if available
        if 'segment_type' in video_data.columns and 'genuine_user' in video_data.columns:
            genuine_frames = video_data[video_data['segment_type'] == 'genuine']
            imposter_frames = video_data[video_data['segment_type'] == 'imposter']

            if len(genuine_frames) > 0:
                metrics['genuine_unlock_rate'] = (
                    len(genuine_frames[genuine_frames['predicted_state'] == 'Unlocked']) / len(genuine_frames)
                ) * 100

            if len(imposter_frames) > 0:
                metrics['imposter_lock_rate'] = (
                    len(imposter_frames[imposter_frames['predicted_state'] == 'Locked']) / len(imposter_frames)
                ) * 100

            metrics['genuine_user'] = video_data['genuine_user'].iloc[0]
            metrics['imposter_user'] = video_data['imposter_user'].iloc[0]

        video_metrics.append(metrics)

    return pd.DataFrame(video_metrics)


def print_segment_comparison(genuine_metrics: dict, imposter_metrics: dict):
    """Print comparison between genuine and imposter segments.

    Args:
        genuine_metrics: Metrics for genuine segments
        imposter_metrics: Metrics for imposter segments
    """
    print("\n" + "="*60)
    print("GENUINE vs IMPOSTER SEGMENT COMPARISON")
    print("="*60)

    print("\nGenuine User Segments (Should be UNLOCKED):")
    print("-" * 60)
    print(f"  Total Frames: {genuine_metrics['total_frames']}")
    print(f"  True Accept (Unlocked): {genuine_metrics['unlocked_frames']} ({genuine_metrics['true_accept_rate']:.2f}%)")
    print(f"  False Reject (Locked): {genuine_metrics['locked_frames']} ({genuine_metrics['false_reject_rate']:.2f}%)")
    print(f"  Avg Risk Score: {genuine_metrics['avg_risk_score']:.4f}")
    print(f"  Avg Distance: {genuine_metrics['avg_distance']:.4f}")

    print("\nImposter Segments (Should be LOCKED):")
    print("-" * 60)
    print(f"  Total Frames: {imposter_metrics['total_frames']}")
    print(f"  True Reject (Locked): {imposter_metrics['locked_frames']} ({imposter_metrics['true_reject_rate']:.2f}%)")
    print(f"  False Accept (Unlocked): {imposter_metrics['unlocked_frames']} ({imposter_metrics['false_accept_rate']:.2f}%)")
    print(f"  Avg Risk Score: {imposter_metrics['avg_risk_score']:.4f}")
    print(f"  Avg Distance: {imposter_metrics['avg_distance']:.4f}")

    # Calculate combined metrics
    far_frr = calculate_false_rates(genuine_metrics, imposter_metrics)
    print("\nCombined Error Rates:")
    print("-" * 60)
    print(f"  False Accept Rate (FAR): {far_frr['false_accept_rate']:.2f}%")
    print(f"  False Reject Rate (FRR): {far_frr['false_reject_rate']:.2f}%")
    print(f"  Approximate EER: {far_frr['equal_error_rate']:.2f}%")


def calculate_segment_metrics(df: pd.DataFrame) -> SegmentAnalysis:
    """Calculate authentication metrics for genuine and imposter segments.

    Args:
        df: Results dataframe with segment_type column

    Returns:
        SegmentAnalysis dataclass with genuine and imposter metrics
    """
    # Genuine segment metrics (should be unlocked)
    genuine_frames = df[df['segment_type'] == 'genuine']
    genuine_unlocked = len(genuine_frames[genuine_frames['predicted_state'] == 'Unlocked'])
    genuine_locked = len(genuine_frames[genuine_frames['predicted_state'] == 'Locked'])

    genuine_total = len(genuine_frames)
    genuine_metrics = GenuineSegmentMetrics(
        total_frames=genuine_total,
        unlocked_frames=genuine_unlocked,
        locked_frames=genuine_locked,
        avg_risk_score=genuine_frames['risk_score'].mean() if genuine_total > 0 else 0.0,
        avg_distance=genuine_frames['distance'].mean() if genuine_total > 0 else 0.0,
        true_accept_rate=(genuine_unlocked / genuine_total) * 100 if genuine_total > 0 else 0.0,
        false_reject_rate=(genuine_locked / genuine_total) * 100 if genuine_total > 0 else 0.0
    )

    # Imposter segment metrics (should be locked)
    imposter_frames = df[df['segment_type'] == 'imposter']
    imposter_locked = len(imposter_frames[imposter_frames['predicted_state'] == 'Locked'])
    imposter_unlocked = len(imposter_frames[imposter_frames['predicted_state'] == 'Unlocked'])
    imposter_total = len(imposter_frames)

    imposter_metrics = ImposterSegmentMetrics(
        total_frames=imposter_total,
        unlocked_frames=imposter_unlocked,
        locked_frames=imposter_locked,
        avg_risk_score=imposter_frames['risk_score'].mean() if imposter_total > 0 else 0.0,
        avg_distance=imposter_frames['distance'].mean() if imposter_total > 0 else 0.0,
        true_reject_rate=(imposter_locked / imposter_total) * 100 if imposter_total > 0 else 0.0,
        false_accept_rate=(imposter_unlocked / imposter_total) * 100 if imposter_total > 0 else 0.0
    )

    return SegmentAnalysis(genuine=genuine_metrics, imposter=imposter_metrics)


def calculate_grouped_metrics(df: pd.DataFrame, group_column: str) -> GroupedMetrics:
    """Calculate metrics grouped by a dimension (device, scenario, environment, etc.).

    Args:
        df: Results dataframe with segment_type column
        group_column: Column name to group by (e.g., 'device', 'scenario')

    Returns:
        GroupedMetrics dataclass
    """
    if group_column not in df.columns:
        print(f"Warning: Column '{group_column}' not found in dataframe")
        return GroupedMetrics(dimension_name=group_column, groups={})

    groups = {}

    for group_value in df[group_column].unique():
        if pd.isna(group_value):
            continue

        group_df = df[df[group_column] == group_value]
        group_metrics = calculate_segment_metrics(group_df)
        groups[str(group_value)] = group_metrics

    return GroupedMetrics(dimension_name=group_column, groups=groups)


def calculate_device_comparison(df: pd.DataFrame, device_column: str = 'device') -> GroupedMetrics:
    """Calculate metrics comparison across device types.

    Args:
        df: Results dataframe with device column
        device_column: Name of device column (default: 'device')

    Returns:
        GroupedMetrics dataclass with device comparison
    """
    return calculate_grouped_metrics(df, device_column)

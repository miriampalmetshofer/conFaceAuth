"""Metric calculation utilities."""
from typing import Optional

import numpy as np

from evaluation.shared.models import (
    AuthenticationMetrics,
    FrameData,
    SegmentType,
    DeviceMetrics,
    ScenarioMetrics,
    ScenarioDeviceMetrics,
    FrameCounts
)


def calculate_metrics(frames: list[FrameData], fps: int) -> AuthenticationMetrics:
    """Calculate authentication metrics from frames."""
    genuine_frames = [f for f in frames if f.segment_type == SegmentType.GENUINE]
    imposter_frames = [f for f in frames if f.segment_type == SegmentType.IMPOSTER]

    genuine_unlocked = sum(1 for f in genuine_frames if f.predicted_state == 'Unlocked')
    genuine_locked = sum(1 for f in genuine_frames if f.predicted_state == 'Locked')

    tar = (genuine_unlocked / len(genuine_frames) * 100) if genuine_frames else 0
    frr = (genuine_locked / len(genuine_frames) * 100) if genuine_frames else 0

    imposter_locked = sum(1 for f in imposter_frames if f.predicted_state == 'Locked')
    imposter_unlocked = sum(1 for f in imposter_frames if f.predicted_state == 'Unlocked')

    trr = (imposter_locked / len(imposter_frames) * 100) if imposter_frames else 0
    far = (imposter_unlocked / len(imposter_frames) * 100) if imposter_frames else 0

    eer = (far + frr) / 2

    all_frames = genuine_frames + imposter_frames
    mean_lockout_time, max_lockout_time = calculate_imposter_lockout_time(frames, fps)
    genuine_kickout_count, genuine_kickout_total, genuine_kickout_time = calculate_genuine_kickout_metrics(frames, fps)

    # Calculate similarity difference (genuine avg - imposter avg)
    genuine_similarities = [f.similarity for f in genuine_frames if f.face_detected]
    imposter_similarities = [f.similarity for f in imposter_frames if f.face_detected]

    if genuine_similarities and imposter_similarities:
        avg_genuine_sim = np.mean(genuine_similarities)
        avg_imposter_sim = np.mean(imposter_similarities)
        similarity_difference = avg_genuine_sim - avg_imposter_sim
    else:
        similarity_difference = None

    counts = FrameCounts(
        total_frames=len(all_frames),
        unlocked_frames=genuine_unlocked + imposter_unlocked,
        locked_frames=genuine_locked + imposter_locked
    )

    return AuthenticationMetrics(
        true_accept_rate=tar,
        false_reject_rate=frr,
        true_reject_rate=trr,
        false_accept_rate=far,
        equal_error_rate=eer,
        imposter_lockout_time=mean_lockout_time,
        max_lockout_time=max_lockout_time,
        similarity_difference=similarity_difference,
        genuine_kickout_count=genuine_kickout_count,
        genuine_kickout_total=genuine_kickout_total,
        genuine_kickout_time=genuine_kickout_time,
        counts=counts
    )


def group_frames_by_video(frames: list[FrameData]) -> dict[str, list[FrameData]]:
    """Group frames by video path."""
    videos = {}
    for frame in frames:
        if frame.video_path not in videos:
            videos[frame.video_path] = []
        videos[frame.video_path].append(frame)
    return videos


def find_first_imposter_frame_index(video_frames: list[FrameData]) -> int:
    """Find index of first imposter frame in video."""
    for i, f in enumerate(video_frames):
        if f.segment_type == SegmentType.IMPOSTER:
            return i
    return None


def find_lockout_transition(video_frames: list[FrameData], first_imposter_idx: int, fps: int) -> Optional[float]:
    """Find transition from Unlocked to Locked and return lockout time in seconds."""
    if video_frames[first_imposter_idx].predicted_state == 'Locked':
        return 0

    for i in range(first_imposter_idx, len(video_frames) - 1):
        if (video_frames[i].predicted_state == 'Unlocked' and
            video_frames[i + 1].predicted_state == 'Locked'):
            lockout_frames = video_frames[i + 1].frame - video_frames[first_imposter_idx].frame
            return lockout_frames / fps

    return None


def report_never_locked_out_videos(never_locked_out: list[str]):
    """Print warning about videos where imposter was never locked out."""
    if never_locked_out:
        print(f"\n⚠️  WARNING: {len(never_locked_out)} video(s) where imposter was NEVER locked out:")
        for vp in never_locked_out:
            print(f"    - {vp}")
        print()


def calculate_imposter_lockout_time(frames: list[FrameData], fps: int) -> tuple[Optional[float], Optional[float]]:
    """Calculate mean and max time until imposter is locked out per video in seconds.

    Returns:
        Tuple of (mean_lockout_time, max_lockout_time)
    """
    videos = group_frames_by_video(frames)

    lockout_times = []
    never_locked_out = []

    for video_path, video_frames in videos.items():
        video_frames.sort(key=lambda f: f.frame)

        first_imposter_idx = find_first_imposter_frame_index(video_frames)
        if first_imposter_idx is None:
            raise ValueError(f"No imposter frames found in video: {video_path}")

        lockout_time = find_lockout_transition(video_frames, first_imposter_idx, fps)

        if lockout_time is not None:
            lockout_times.append(lockout_time)
        else:
            never_locked_out.append(video_path)

    report_never_locked_out_videos(never_locked_out)

    if lockout_times:
        return np.mean(lockout_times), np.max(lockout_times)
    else:
        return None, None


def find_genuine_kickout_transition(video_frames: list[FrameData], fps: int) -> Optional[float]:
    """Find when genuine user gets kicked out (Unlocked -> Locked during genuine segment).

    Returns:
        Time in seconds from start of video until genuine user kickout, or None if never kicked out.
    """
    for i in range(len(video_frames) - 1):
        current_frame = video_frames[i]
        next_frame = video_frames[i + 1]

        # Check if both frames are in genuine segment and transition from Unlocked to Locked
        if (current_frame.segment_type == SegmentType.GENUINE and
            next_frame.segment_type == SegmentType.GENUINE and
            current_frame.predicted_state == 'Unlocked' and
            next_frame.predicted_state == 'Locked'):

            kickout_frames = next_frame.frame
            return kickout_frames / fps

    return None


def calculate_genuine_kickout_metrics(frames: list[FrameData], fps: int) -> tuple[Optional[int], Optional[int], Optional[float]]:
    """Calculate genuine user kickout metrics across unique genuine videos.

    Returns:
        Tuple of (kickout_count, total_unique_genuine_videos, mean_genuine_kickout_time)
        - kickout_count: number of unique genuine videos where user got kicked out
        - total_unique_genuine_videos: total number of unique genuine user videos
        - mean_genuine_kickout_time: mean time until kickout (only for videos where kickout occurred)
    """
    videos = group_frames_by_video(frames)

    # Group by source_type (unique genuine video identifier)
    unique_genuine_videos = {}

    for video_path, video_frames in videos.items():
        if not video_frames:
            continue

        # Get source_type from genuine segment frames
        genuine_source_types = set(f.source_type for f in video_frames if f.segment_type == SegmentType.GENUINE)

        if not genuine_source_types:
            continue

        # Should only be one unique source_type for genuine frames in a video
        source_type = next(iter(genuine_source_types))

        # Skip if we already processed this unique genuine video
        if source_type in unique_genuine_videos:
            continue

        video_frames.sort(key=lambda f: f.frame)
        kickout_time = find_genuine_kickout_transition(video_frames, fps)

        unique_genuine_videos[source_type] = kickout_time

    # Count kickouts and collect times
    videos_with_kickout = sum(1 for t in unique_genuine_videos.values() if t is not None)
    kickout_times = [t for t in unique_genuine_videos.values() if t is not None]
    total_unique_videos = len(unique_genuine_videos)

    # Calculate mean kickout time (only for videos where it happened)
    mean_kickout_time = np.mean(kickout_times) if kickout_times else None

    return videos_with_kickout, total_unique_videos, mean_kickout_time


def calculate_metrics_by_device(frames: list[FrameData], devices: list[str], fps: int) -> list[DeviceMetrics]:
    """Calculate metrics grouped by device."""
    device_metrics = []
    for device in devices:
        device_frames = [f for f in frames if f.device == device]
        if device_frames:
            metrics = calculate_metrics(device_frames, fps)
            device_metrics.append(DeviceMetrics(device=device, metrics=metrics))
    return device_metrics


def calculate_metrics_by_scenario(frames: list[FrameData], scenarios: list[str], fps: int) -> list[ScenarioMetrics]:
    """Calculate metrics grouped by scenario."""
    scenario_metrics = []
    for scenario in scenarios:
        scenario_frames = [f for f in frames if get_frame_scenario(f, scenarios) == scenario]
        if scenario_frames:
            metrics = calculate_metrics(scenario_frames, fps)
            scenario_metrics.append(ScenarioMetrics(scenario=scenario, metrics=metrics))
    return scenario_metrics


def get_frame_scenario(frame: FrameData, scenarios: list[str]) -> Optional[str]:
    """Get scenario for a frame from source_type."""
    for scenario in scenarios:
        if scenario in frame.source_type:
            return scenario
    return None


def calculate_metrics_by_scenario_and_device(
    frames: list[FrameData],
    scenarios: list[str],
    devices: list[str],
    fps: int
) -> list[ScenarioDeviceMetrics]:
    """Calculate metrics grouped by scenario and device combination."""
    scenario_device_metrics = []
    for scenario in scenarios:
        for device in devices:
            filtered_frames = [
                f for f in frames
                if get_frame_scenario(f, scenarios) == scenario and f.device == device
            ]
            if filtered_frames:
                metrics = calculate_metrics(filtered_frames, fps)
                scenario_device_metrics.append(
                    ScenarioDeviceMetrics(scenario=scenario, device=device, metrics=metrics)
                )
    return scenario_device_metrics

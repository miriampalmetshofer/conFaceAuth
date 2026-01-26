"""Metric calculation utilities."""
from typing import Optional

import numpy as np

from evaluation.common.domain import (
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
    lockout_time = calculate_imposter_lockout_time(frames, fps)

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
        imposter_lockout_time=lockout_time,
        counts=counts
    )


def calculate_imposter_lockout_time(frames: list[FrameData], fps: int) -> Optional[float]:
    """Calculate mean time until imposter is locked out per video in seconds."""
    videos = {}
    for frame in frames:
        if frame.video_path not in videos:
            videos[frame.video_path] = []
        videos[frame.video_path].append(frame)

    lockout_times = []
    never_locked_out = []

    for video_path, video_frames in videos.items():
        video_frames.sort(key=lambda f: f.frame)

        first_imposter_idx = None
        for i, f in enumerate(video_frames):
            if f.segment_type == SegmentType.IMPOSTER:
                first_imposter_idx = i
                break

        if first_imposter_idx is None:
            raise ValueError(f"No imposter frames found in video: {video_path}")

        # Check if imposter is immediately locked from first frame
        if video_frames[first_imposter_idx].predicted_state == 'Locked':
            lockout_times.append(0)
            continue

        # Look for transition from Unlocked to Locked
        found_lockout = False
        for i in range(first_imposter_idx, len(video_frames) - 1):
            if (video_frames[i].predicted_state == 'Unlocked' and
                video_frames[i + 1].predicted_state == 'Locked'):
                lockout_frames = video_frames[i + 1].frame - video_frames[first_imposter_idx].frame
                lockout_seconds = lockout_frames / fps
                lockout_times.append(lockout_seconds)
                found_lockout = True
                break

        if not found_lockout:
            never_locked_out.append(video_path)

    if never_locked_out:
        print(f"\n⚠️  WARNING: {len(never_locked_out)} video(s) where imposter was NEVER locked out:")
        for vp in never_locked_out:
            print(f"    - {vp}")
        print()

    return np.mean(lockout_times) if lockout_times else None


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

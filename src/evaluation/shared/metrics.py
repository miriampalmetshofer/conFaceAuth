"""Metric calculation utilities."""
from typing import Optional, NamedTuple

import numpy as np

from evaluation.shared.models import (
    AuthenticationMetrics,
    FrameData,
    SegmentType,
    TimeStat,
    SessionCounts,
    DeviceMetrics,
    ScenarioMetrics,
    ScenarioDeviceMetrics,
)

class _ImposterLockoutResult(NamedTuple):
    lockouts: int
    sessions: int
    median: Optional[float]
    p90: Optional[float]
    max: Optional[float]


class _GenuineLockoutResult(NamedTuple):
    lockouts: int
    sessions: int
    median: Optional[float]
    p90: Optional[float]


def group_frames_by_video(frames: list[FrameData]) -> dict[str, list[FrameData]]:
    """Group frames by video path."""
    videos: dict[str, list[FrameData]] = {}
    for frame in frames:
        if frame.video_path not in videos:
            videos[frame.video_path] = []
        videos[frame.video_path].append(frame)
    return videos


def find_first_imposter_frame_index(video_frames: list[FrameData]) -> Optional[int]:
    """Return index of the first imposter frame, or None if absent."""
    for i, f in enumerate(video_frames):
        if f.segment_type == SegmentType.IMPOSTER:
            return i
    return None


def find_lockout_transition(video_frames: list[FrameData], first_imposter_idx: int, fps: int) -> Optional[float]:
    """Return seconds from imposter segment start until first Unlocked→Locked transition, or None."""
    if video_frames[first_imposter_idx].predicted_state == 'Locked':
        return 0.0
    for i in range(first_imposter_idx, len(video_frames) - 1):
        if (video_frames[i].predicted_state == 'Unlocked' and
                video_frames[i + 1].predicted_state == 'Locked'):
            lockout_frames = video_frames[i + 1].frame - video_frames[first_imposter_idx].frame
            return lockout_frames / fps
    return None


def find_genuine_lockout_transition(video_frames: list[FrameData], fps: int) -> Optional[float]:
    """Return seconds from video start until genuine user is wrongly locked out, or None."""
    for i in range(len(video_frames) - 1):
        cur, nxt = video_frames[i], video_frames[i + 1]
        if (cur.segment_type == SegmentType.GENUINE and
                nxt.segment_type == SegmentType.GENUINE and
                cur.predicted_state == 'Unlocked' and
                nxt.predicted_state == 'Locked'):
            return nxt.frame / fps
    return None


def report_never_locked_out_videos(never_locked_out: list[str]) -> None:
    """Print warning for imposter sessions that were never locked out."""
    if never_locked_out:
        print(f"\n⚠️  WARNING: {len(never_locked_out)} video(s) where imposter was NEVER locked out:")
        for vp in never_locked_out:
            print(f"    - {vp}")
        print()


def _compute_genuine_trust(frames: list[FrameData]) -> Optional[float]:
    """Mean trust score across all genuine frames."""
    scores = [f.trust_score for f in frames if f.segment_type == SegmentType.GENUINE]
    return float(np.mean(scores)) if scores else None


def _compute_imposter_lockout(frames: list[FrameData], fps: int) -> _ImposterLockoutResult:
    """Compute imposter lockout count, session total, median/P90/max lockout time."""
    videos = group_frames_by_video(frames)
    lockout_times: list[float] = []
    never_locked_out: list[str] = []

    for video_path, video_frames in videos.items():
        video_frames.sort(key=lambda f: f.frame)
        first_idx = find_first_imposter_frame_index(video_frames)
        if first_idx is None:
            raise ValueError(f"No imposter frames found in video: {video_path}")
        t = find_lockout_transition(video_frames, first_idx, fps)
        if t is not None:
            lockout_times.append(t)
        else:
            never_locked_out.append(video_path)

    report_never_locked_out_videos(never_locked_out)

    if lockout_times:
        return _ImposterLockoutResult(
            lockouts=len(lockout_times),
            sessions=len(videos),
            median=float(np.median(lockout_times)),
            p90=float(np.percentile(lockout_times, 90)),
            max=float(np.max(lockout_times)),
        )
    return _ImposterLockoutResult(lockouts=0, sessions=len(videos), median=None, p90=None, max=None)


def _compute_genuine_lockout(frames: list[FrameData], fps: int) -> _GenuineLockoutResult:
    """Compute genuine lockout count, session total, and median/P90 lockout time.

    Deduplicates by source_type so each unique genuine user video is counted once.
    """
    videos = group_frames_by_video(frames)
    seen_source_types: set[str] = set()
    lockout_times: list[float] = []
    total_sessions = 0

    for video_frames in videos.values():
        genuine_source_types = {f.source_type for f in video_frames if f.segment_type == SegmentType.GENUINE}
        if not genuine_source_types:
            continue
        source_type = next(iter(genuine_source_types))
        if source_type in seen_source_types:
            continue
        seen_source_types.add(source_type)
        total_sessions += 1

        video_frames.sort(key=lambda f: f.frame)
        t = find_genuine_lockout_transition(video_frames, fps)
        if t is not None:
            lockout_times.append(t)

    if lockout_times:
        return _GenuineLockoutResult(
            lockouts=len(lockout_times),
            sessions=total_sessions,
            median=float(np.median(lockout_times)),
            p90=float(np.percentile(lockout_times, 90)),
        )
    return _GenuineLockoutResult(lockouts=0, sessions=total_sessions, median=None, p90=None)


def _compute_session_rates(imp: _ImposterLockoutResult, gen: _GenuineLockoutResult) -> tuple[float, float]:
    """Return (FRR %, FAR %) from imposter and genuine lockout results."""
    frr = (gen.lockouts / gen.sessions * 100) if gen.sessions else 0.0
    far = ((imp.sessions - imp.lockouts) / imp.sessions * 100) if imp.sessions else 0.0
    return frr, far


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def calculate_metrics(frames: list[FrameData], fps: int) -> AuthenticationMetrics:
    """Calculate all reported authentication metrics from a list of frames."""
    imp = _compute_imposter_lockout(frames, fps)
    gen = _compute_genuine_lockout(frames, fps)
    frr, far = _compute_session_rates(imp, gen)

    return AuthenticationMetrics(
        false_reject_rate=frr,
        false_accept_rate=far,
        mean_genuine_trust=_compute_genuine_trust(frames),
        imposter_lockout_time=TimeStat(median=imp.median, p90=imp.p90, max=imp.max),
        genuine_lockout_time=TimeStat(median=gen.median, p90=gen.p90, max=None),
        session_counts=SessionCounts(
            genuine_sessions=gen.sessions,
            genuine_lockouts=gen.lockouts,
            imposter_sessions=imp.sessions,
            imposter_lockouts=imp.lockouts,
        ),
    )


def calculate_metrics_by_device(frames: list[FrameData], devices: list[str], fps: int) -> list[DeviceMetrics]:
    """Calculate metrics grouped by device."""
    return [
        DeviceMetrics(device=device, metrics=calculate_metrics(
            [f for f in frames if f.device == device], fps
        ))
        for device in devices
        if any(f.device == device for f in frames)
    ]


def get_frame_scenario(frame: FrameData, scenarios: list[str]) -> Optional[str]:
    """Return scenario name if found in frame source_type, else None."""
    for scenario in scenarios:
        if scenario in frame.source_type:
            return scenario
    return None


def calculate_metrics_by_scenario(frames: list[FrameData], scenarios: list[str], fps: int) -> list[ScenarioMetrics]:
    """Calculate metrics grouped by scenario."""
    return [
        ScenarioMetrics(scenario=scenario, metrics=calculate_metrics(
            [f for f in frames if get_frame_scenario(f, scenarios) == scenario], fps
        ))
        for scenario in scenarios
        if any(get_frame_scenario(f, scenarios) == scenario for f in frames)
    ]


def calculate_metrics_by_scenario_and_device(
    frames: list[FrameData],
    scenarios: list[str],
    devices: list[str],
    fps: int,
) -> list[ScenarioDeviceMetrics]:
    """Calculate metrics grouped by scenario and device combination."""
    results = []
    for scenario in scenarios:
        for device in devices:
            filtered = [
                f for f in frames
                if get_frame_scenario(f, scenarios) == scenario and f.device == device
            ]
            if filtered:
                results.append(ScenarioDeviceMetrics(
                    scenario=scenario,
                    device=device,
                    metrics=calculate_metrics(filtered, fps),
                ))
    return results

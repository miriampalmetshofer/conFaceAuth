"""Metric calculation over authentication sessions.

Metrics are computed from :class:`Session` objects. Each session is one composed
video with a genuine segment, a black transition, and an imposter segment. All
times are derived from ``timestamp_ms`` and reported in seconds.
"""
from __future__ import annotations

from typing import NamedTuple, Optional

import numpy as np

from face_auth.domain import Device, Scenario
from evaluation.shared.grouping import group_sessions_ordered
from evaluation.shared.reporting import (
    report_excluded_imposter_sessions,
    report_never_locked_out_sessions,
)
from evaluation.shared.models import (
    AuthenticationMetrics,
    DeviceMetrics,
    ScenarioDeviceMetrics,
    ScenarioMetrics,
    Segment,
    SessionCounts,
    Session,
    TimeStat,
)


class _ImposterLockoutResult(NamedTuple):
    lockouts: int
    sessions: int
    excluded_sessions: int
    mean: Optional[float]
    p90: Optional[float]
    max: Optional[float]


class _GenuineLockoutResult(NamedTuple):
    lockouts: int
    sessions: int
    mean: Optional[float]
    p90: Optional[float]


def first_lockout_seconds(segment: Segment) -> Optional[float]:
    """Seconds from a segment's start until its first authenticated to locked drop.

    Returns 0.0 if the segment starts already locked out, or ``None`` if the
    segment never transitions out of the authenticated state. For an imposter
    segment this is the unauthorized-user lockout time (ULT); callers should first
    check :func:`genuine_stays_authenticated` for eligibility.
    """
    samples = segment.samples
    if not samples[0].is_authenticated:
        return 0.0
    for current, following in zip(samples, samples[1:]):
        if current.is_authenticated and not following.is_authenticated:
            return (following.timestamp_ms - segment.start_ms) / 1000.0
    return None


def genuine_stays_authenticated(session: Session) -> bool:
    """Whether the genuine user is authenticated for the whole genuine segment.

    A session is eligible for imposter-rejection metrics only when this holds.
    """
    return all(sample.is_authenticated for sample in session.genuine.samples)


def _compute_genuine_trust(sessions: list[Session]) -> Optional[float]:
    """Mean trust score across all genuine samples of the given sessions."""
    scores = [
        sample.trust_score
        for session in sessions
        for sample in session.genuine.samples
    ]
    return float(np.mean(scores)) if scores else None


def _compute_imposter_lockout(sessions: list[Session]) -> _ImposterLockoutResult:
    """Compute imposter lockout count, eligible session total, and lockout times.

    A session is eligible only if the genuine user stayed authenticated through the
    genuine segment. Lockout time is measured from the imposter segment's start.
    """
    lockout_times: list[float] = []
    never_locked_out: list[str] = []
    excluded: list[str] = []
    eligible_sessions = 0

    for session in sessions:
        if not genuine_stays_authenticated(session):
            excluded.append(session.csv_path.name)
            continue
        eligible_sessions += 1
        t = first_lockout_seconds(session.imposter)
        if t is not None:
            lockout_times.append(t)
        else:
            never_locked_out.append(session.csv_path.name)

    report_excluded_imposter_sessions(excluded)
    report_never_locked_out_sessions(never_locked_out)

    if lockout_times:
        return _ImposterLockoutResult(
            lockouts=len(lockout_times),
            sessions=eligible_sessions,
            excluded_sessions=len(excluded),
            mean=float(np.mean(lockout_times)),
            p90=float(np.percentile(lockout_times, 90)),
            max=float(np.max(lockout_times)),
        )
    return _ImposterLockoutResult(
        lockouts=0,
        sessions=eligible_sessions,
        excluded_sessions=len(excluded),
        mean=None,
        p90=None,
        max=None,
    )


def _compute_genuine_lockout(sessions: list[Session]) -> _GenuineLockoutResult:
    """Compute genuine lockout count, session total, and mean/P90 lockout time.

    Deduplicates by genuine source stem so each unique genuine recording is counted
    once even when it appears in several composed videos against different imposters.
    """
    seen_stems: set[str] = set()
    lockout_times: list[float] = []
    total_sessions = 0

    for session in sessions:
        stem = session.genuine_identity.source_stem
        if stem in seen_stems:
            continue
        seen_stems.add(stem)
        total_sessions += 1

        t = first_lockout_seconds(session.genuine)
        if t is not None:
            lockout_times.append(t)

    if lockout_times:
        return _GenuineLockoutResult(
            lockouts=len(lockout_times),
            sessions=total_sessions,
            mean=float(np.mean(lockout_times)),
            p90=float(np.percentile(lockout_times, 90)),
        )
    return _GenuineLockoutResult(lockouts=0, sessions=total_sessions, mean=None, p90=None)


def _compute_session_rates(imp: _ImposterLockoutResult, gen: _GenuineLockoutResult) -> tuple[float, float]:
    """Return (FRR %, FAR %) from imposter and genuine lockout results."""
    frr = (gen.lockouts / gen.sessions * 100) if gen.sessions else 0.0
    far = ((imp.sessions - imp.lockouts) / imp.sessions * 100) if imp.sessions else 0.0
    return frr, far


def calculate_metrics(sessions: list[Session]) -> AuthenticationMetrics:
    """Calculate all reported authentication metrics from a list of sessions."""
    imp = _compute_imposter_lockout(sessions)
    gen = _compute_genuine_lockout(sessions)
    frr, far = _compute_session_rates(imp, gen)

    return AuthenticationMetrics(
        false_reject_rate=frr,
        false_accept_rate=far,
        mean_genuine_trust=_compute_genuine_trust(sessions),
        imposter_lockout_time=TimeStat(mean=imp.mean, p90=imp.p90, max=imp.max),
        genuine_lockout_time=TimeStat(mean=gen.mean, p90=gen.p90, max=None),
        session_counts=SessionCounts(
            genuine_sessions=gen.sessions,
            genuine_lockouts=gen.lockouts,
            imposter_sessions=imp.sessions,
            imposter_lockouts=imp.lockouts,
        ),
    )


def calculate_metrics_by_device(sessions: list[Session], devices: list[Device]) -> list[DeviceMetrics]:
    """Calculate metrics grouped by device, in the given device order."""
    return [
        DeviceMetrics(device=device, metrics=calculate_metrics(group))
        for device, group in group_sessions_ordered(sessions, devices, lambda s: s.device)
    ]


def calculate_metrics_by_scenario(sessions: list[Session], scenarios: list[Scenario]) -> list[ScenarioMetrics]:
    """Calculate metrics grouped by the genuine user's recording scenario, in scenario order."""
    return [
        ScenarioMetrics(scenario=scenario, metrics=calculate_metrics(group))
        for scenario, group in group_sessions_ordered(
            sessions, scenarios, lambda s: s.genuine_identity.scenario
        )
    ]


def calculate_metrics_by_scenario_and_device(
    sessions: list[Session],
    scenarios: list[Scenario],
    devices: list[Device],
) -> list[ScenarioDeviceMetrics]:
    """Calculate metrics grouped by scenario and device, scenario-major then device order."""
    return [
        ScenarioDeviceMetrics(scenario=scenario, device=device, metrics=calculate_metrics(group))
        for scenario, scenario_sessions in group_sessions_ordered(
            sessions, scenarios, lambda s: s.genuine_identity.scenario
        )
        for device, group in group_sessions_ordered(
            scenario_sessions, devices, lambda s: s.device
        )
    ]

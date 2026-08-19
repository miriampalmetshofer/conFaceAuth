"""Domain models for evaluation.

The models mirror how results are written to disk. One run directory contains a
``config.json`` and, per participant and device, one CSV per composed video. Each
composed video is a single authentication session stitched from a genuine-user
segment, a short black-frame segment, and an imposter segment. The hierarchy is

    Run
     └── Session               (one composed-video CSV)
          └── Segment          (genuine / black / imposter run of rows)
               └── FrameResult (one CSV row)

Time is carried as ``timestamp_ms`` throughout; there is no frame index or fps.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

from face_auth.domain import Device, Pool, Scenario


class SegmentRole(Enum):
    """Role of a segment within a composed authentication session."""
    GENUINE = "genuine"
    BLACK = "black"
    IMPOSTER = "imposter"


# The literal source_type written for the black transition segment.
BLACK_SOURCE_TYPE = "black_frames"


@dataclass(frozen=True)
class FrameResult:
    """One CSV row — a single sampled frame's authentication verdict.

    Fields mirror the CSV columns exactly, with no derived values.
    """
    timestamp_ms: float
    is_authenticated: bool
    similarity: float
    trust_score: float
    face_detected: bool
    source_type: str


@dataclass(frozen=True)
class Identity:
    """The person behind a segment, parsed from its source video stem.

    ``scenario`` is populated for the controlled study and is ``None`` for the
    in-the-wild study, whose recordings carry no scenario.
    """
    name: str
    source_stem: str
    scenario: Optional[Scenario]


@dataclass(frozen=True)
class Segment:
    """A contiguous run of samples sharing one source_type, with one role
    (genuine, black, or imposter)."""
    role: SegmentRole
    source_type: str
    samples: list[FrameResult]

    @property
    def start_ms(self) -> float:
        """Timestamp of the segment's first sample."""
        return self.samples[0].timestamp_ms

    @property
    def end_ms(self) -> float:
        """Timestamp of the segment's last sample."""
        return self.samples[-1].timestamp_ms


@dataclass(frozen=True)
class Session:
    """One composed-video CSV — a single authentication run.

    A session pairs a genuine user against one imposter with a black transition in
    between. The genuine and imposter identities are resolved once at load time.
    """
    csv_path: Path
    participant: str
    device: Device
    genuine: Segment
    black: Optional[Segment]
    imposter: Segment
    genuine_identity: Identity
    imposter_identity: Identity


@dataclass(frozen=True)
class RunConfig:
    """The slice of ``config.json`` the evaluation needs.

    ``raw`` holds the full config dict for fields not promoted to typed attributes here.
    """
    pool: Pool
    devices: list[Device]
    threshold: float
    authentication_hz: float
    genuine_user_seconds: float
    impostor_seconds: float
    black_screen_seconds: float
    raw: dict


@dataclass(frozen=True)
class Run:
    """One ``run_<timestamp>_<suffix>`` results directory."""
    run_dir: Path
    config: RunConfig
    sessions: list[Session]


@dataclass
class TimeStat:
    """Time statistic bundling mean, P90, and optional max, in seconds."""
    mean: Optional[float]
    p90: Optional[float]
    max: Optional[float]


@dataclass
class SessionCounts:
    """Raw session counts — supporting data for FRR/FAR fraction display only."""
    genuine_sessions: int
    genuine_lockouts: int       # genuine sessions where user was wrongly locked out
    imposter_sessions: int      # eligible imposter sessions, excluding earlier lockouts
    imposter_lockouts: int      # imposter sessions where imposter was successfully locked out


@dataclass
class AuthenticationMetrics:
    """Core authentication performance metrics for one group of sessions.

    ``session_counts`` is supporting data used only to show FRR/FAR as fractions.
    """
    false_reject_rate: float
    false_accept_rate: float
    mean_genuine_trust: Optional[float]
    imposter_lockout_time: TimeStat
    genuine_lockout_time: TimeStat
    session_counts: SessionCounts


@dataclass
class DeviceMetrics:
    """Metrics aggregated by device."""
    device: Device
    metrics: AuthenticationMetrics


@dataclass
class ScenarioMetrics:
    """Metrics aggregated by the genuine user's recording scenario."""
    scenario: Scenario
    metrics: AuthenticationMetrics


@dataclass
class ScenarioDeviceMetrics:
    """Metrics aggregated by scenario and device combination."""
    scenario: Scenario
    device: Device
    metrics: AuthenticationMetrics

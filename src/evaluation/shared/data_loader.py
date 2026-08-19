"""Loading a results run directory into the evaluation domain model.

A run directory is laid out as

    run_<timestamp>_<suffix>/
        config.json
        <participant>/<device>/<genuine_stem>_vs_<imposter_stem>.csv

Each CSV is one composed authentication session with three back-to-back segments
identified by the ``source_type`` column: a genuine-user segment, a short
``black_frames`` segment, and an imposter segment. The genuine stem is the
``source_type`` before the black transition, the imposter stem the one after.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from face_auth.domain import Device, Pool, Scenario
from face_auth.video_parser.impl.controlled_study_parser import ControlledStudyParser
from face_auth.video_parser.impl.in_the_wild_study_parser import InTheWildStudyParser
from evaluation.shared.models import (
    BLACK_SOURCE_TYPE,
    Identity,
    Run,
    RunConfig,
    FrameResult,
    Segment,
    SegmentRole,
    Session,
)


def parse_bool(value: object) -> bool:
    """Parse a Boolean value from a CSV field."""
    if isinstance(value, bool):
        return value

    normalized = str(value).strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"Invalid Boolean value: {value}")


def resolve_identity(source_stem: str, pool: Pool) -> Identity:
    """Resolve the person behind a source video stem for the given pool.

    The in-the-wild study has no scenario; the controlled study carries one in the stem.
    """
    if pool == Pool.CONTROLLED_STUDY:
        match = ControlledStudyParser.regex.match(source_stem)
        if not match:
            raise ValueError(f"Cannot parse controlled-study source stem: {source_stem}")
        return Identity(
            name=match.group("name"),
            source_stem=source_stem,
            scenario=Scenario(match.group("scenario").lower()),
        )

    if pool == Pool.IN_THE_WILD:
        match = InTheWildStudyParser.regex.match(source_stem)
        if not match:
            raise ValueError(f"Cannot parse in-the-wild source stem: {source_stem}")
        return Identity(
            name=match.group("name"),
            source_stem=source_stem,
            scenario=None,
        )

    raise ValueError(f"Unknown pool: {pool}")


def load_run_config(config_path: Path) -> RunConfig:
    """Load and validate the run's ``config.json`` into a typed RunConfig."""
    with open(config_path, 'r') as f:
        raw = json.load(f)

    imposter = raw['imposter_creation']
    return RunConfig(
        pool=Pool(raw['pool']),
        devices=[Device(d) for d in raw['devices']],
        threshold=raw['authentication']['backend']['threshold'],
        authentication_hz=raw['authentication']['authentication_hz'],
        genuine_user_seconds=imposter['genuine_user_seconds'],
        impostor_seconds=imposter['impostor_seconds'],
        black_screen_seconds=imposter['black_screen_seconds'],
        raw=raw,
    )


def _read_samples(csv_path: Path) -> list[FrameResult]:
    """Read one composed-video CSV into an ordered list of samples."""
    df = pd.read_csv(csv_path)
    return [
        FrameResult(
            timestamp_ms=float(row.timestamp_ms),
            is_authenticated=parse_bool(row.is_authenticated),
            similarity=float(row.similarity),
            trust_score=float(row.trust_score),
            face_detected=parse_bool(row.face_detected),
            source_type=str(row.source_type),
        )
        for row in df.itertuples(index=False)
    ]


def _split_into_segments(samples: list[FrameResult]) -> list[tuple[str, list[FrameResult]]]:
    """Split ordered samples into contiguous runs sharing one source_type.

    Returns a list of (source_type, samples) blocks in timeline order.
    """
    blocks: list[tuple[str, list[FrameResult]]] = []
    for sample in samples:
        if blocks and blocks[-1][0] == sample.source_type:
            blocks[-1][1].append(sample)
        else:
            blocks.append((sample.source_type, [sample]))
    return blocks


def _build_session(csv_path: Path, participant: str, device: Device, pool: Pool) -> Session:
    """Assemble one Session from a composed-video CSV.

    The CSV is expected to hold exactly one genuine segment, an optional
    ``black_frames`` segment, and one imposter segment, in that timeline order.
    """
    samples = _read_samples(csv_path)
    blocks = _split_into_segments(samples)

    genuine_block = None
    black_block = None
    imposter_block = None

    for source_type, block_samples in blocks:
        if source_type == BLACK_SOURCE_TYPE:
            if black_block is not None:
                raise ValueError(f"Multiple black segments in {csv_path}")
            black_block = (source_type, block_samples)
        elif genuine_block is None:
            genuine_block = (source_type, block_samples)
        elif imposter_block is None:
            imposter_block = (source_type, block_samples)
        else:
            raise ValueError(f"Unexpected extra segment '{source_type}' in {csv_path}")

    if genuine_block is None or imposter_block is None:
        raise ValueError(f"Missing genuine or imposter segment in {csv_path}")

    genuine = Segment(SegmentRole.GENUINE, genuine_block[0], genuine_block[1])
    imposter = Segment(SegmentRole.IMPOSTER, imposter_block[0], imposter_block[1])
    black = (
        Segment(SegmentRole.BLACK, black_block[0], black_block[1])
        if black_block is not None else None
    )

    return Session(
        csv_path=csv_path,
        participant=participant,
        device=device,
        genuine=genuine,
        black=black,
        imposter=imposter,
        genuine_identity=resolve_identity(genuine.source_type, pool),
        imposter_identity=resolve_identity(imposter.source_type, pool),
    )


def load_run(run_dir: Path) -> Run:
    """Load a full run directory into a Run.

    Walks ``<run_dir>/<participant>/<device>/*.csv``, reading the run config once
    and building one Session per CSV.
    """
    config = load_run_config(run_dir / "config.json")

    sessions: list[Session] = []
    for participant_dir in sorted(p for p in run_dir.iterdir() if p.is_dir()):
        participant = participant_dir.name
        for device_dir in sorted(d for d in participant_dir.iterdir() if d.is_dir()):
            device = Device(device_dir.name)
            for csv_path in sorted(device_dir.glob("*.csv")):
                sessions.append(_build_session(csv_path, participant, device, config.pool))

    return Run(run_dir=run_dir, config=config, sessions=sessions)

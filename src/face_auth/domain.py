"""Domain models shared across the study pipeline.

These types are the study-domain vocabulary (recording scenarios, head rotations,
pools, devices, participants) plus the value objects that flow between features
(composed videos, enrollment data, per-frame and per-video results). They are
study-specific rather than reusable core machinery, so they live here at the
package root rather than in core/.
"""
from dataclasses import dataclass
from datetime import date
from enum import Enum
from pathlib import Path
from typing import Iterator, List

import numpy as np
from pydantic import BaseModel, Field, ConfigDict

from face_auth.core.authenticator.models import AuthenticationResult
from face_auth.core.models import Frame
from face_auth.imposter_creation.iterators.frame_iterator import FrameIterator

VIDEO_EXTENSIONS = ("mp4", "MP4")


class Pool(Enum):
    """Available pool types."""
    CONTROLLED_STUDY = "controlled_study"
    IN_THE_WILD = "in_the_wild"


class Scenario(Enum):
    """Video recording scenarios."""
    EASY = "easy"
    ANGLE = "angle"
    LIGHTING = "lighting"


class HeadRotation(Enum):
    """Head rotation directions for video scenarios."""
    CLOCKWISE = "cw"
    COUNTERCLOCKWISE = "ccw"


class Device(Enum):
    """Device types for video recording."""
    MOBILE = "mobile"
    DESKTOP = "desktop"


class Participant(BaseModel):
    """A study participant."""
    name: str = Field(..., min_length=1)

    model_config = ConfigDict(frozen=True)


class ProcessingContext(BaseModel):
    """Context for processing a participant on a specific device."""
    participant: Participant
    device: Device
    pool: Pool

    model_config = ConfigDict(frozen=True)


@dataclass
class Video:
    """Represents a video file with parsed metadata."""
    path: Path
    recording_date: date
    participant: Participant


@dataclass
class ImposterSamplePair:
    """Pair of genuine user video with matching imposter videos for stitching."""
    genuine_video: Video
    imposter_video: Video


@dataclass(frozen=True, kw_only=True)
class IndexedFrame(Frame):
    """A core Frame carrying its position on a composed video's continuous timeline."""
    index: int


@dataclass
class ComposedVideo:
    """Virtual video composed from multiple frame iterators, with no physical file.

    A composed video is an in-memory sequence of frame iterators rather than a file
    on disk, so it carries an identifying name, its iterators, and the frame rate
    all its segments share. The frame rate lets consumers convert between frame
    indices and real time.
    """
    name: str
    iterators: List['FrameIterator']
    fps: float

    def frames(self) -> 'Iterator[IndexedFrame]':
        """Yield every frame as one continuous, timestamped stream.

        The segments are walked back to back and share a single 1-based frame
        index and timeline, so the stitched video behaves like one recording
        rather than three separate clips.
        """
        index = 1
        for iterator in self.iterators:
            for image in iterator:
                yield IndexedFrame(
                    image=image,
                    timestamp_ms=(index / self.fps) * 1000,
                    source_name=iterator.source_name,
                    index=index,
                )
                index += 1


@dataclass
class AuthenticatedFrame:
    """One frame of a composed video paired with its authentication verdict.

    The frame's index and source come from the frame itself, so nothing is
    duplicated alongside the result.
    """
    frame: IndexedFrame
    result: AuthenticationResult


@dataclass
class EnrollmentData:
    """Container for enrollment data."""

    folder: Path
    embeddings: List[np.ndarray]


@dataclass
class VideoResult:
    """Container for video processing results."""

    video: ComposedVideo
    frame_results: List[AuthenticatedFrame]

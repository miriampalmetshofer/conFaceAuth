"""Service for creating composed imposter videos using frame iterators."""

import cv2
from pathlib import Path
from typing import List

from face_auth.config.logging_config import get_logger
from face_auth.config.models import StitchConfig
from face_auth.core.imposter_video_creation import VideoFrameIterator, BlackFrameGenerator, FrameIterator
from face_auth.core.processing.models import ImposterSamplePair, ComposedVideo

logger = get_logger(__name__)


class ImposterVideoCreationService:
    """Service for creating composed imposter videos using frame iterators."""

    def __init__(self, stitch_config: StitchConfig):
        """Initialize with stitching configuration.

        Args:
            stitch_config: Configuration for imposter creation
        """
        self.config = stitch_config

    def create(self, pair: ImposterSamplePair) -> ComposedVideo:
        """Create composed imposter video from frame iterators.

        Args:
            pair: Containing genuine and imposter sample

        Returns:
            ComposedVideo with frame iterators (no physical file)
        """
        logger.debug(
            f"Creating composed video: {pair.genuine_video.path.name} + "
            f"{pair.imposter_video.path.name}"
        )

        iterators = self._create_iterators(pair)
        virtual_path = self._build_virtual_path(pair)

        return ComposedVideo(
            path=virtual_path,
            recording_date=pair.genuine_video.recording_date,
            participant=pair.genuine_video.participant,
            iterators=iterators,
            cacheable_iterator=iterators[0]  # Genuine video - reused across multiple imposter pairs
        )

    def _create_iterators(self, pair):
        width, height = self._get_video_dimensions(pair.genuine_video.path)

        iterators: List[FrameIterator] = [
            VideoFrameIterator(
                video_path=pair.genuine_video.path,
                duration_seconds=self.config.genuine_user_seconds,
                fps=self.config.fps
            ),
            BlackFrameGenerator(
                width=width,
                height=height,
                num_frames=int(self.config.black_screen_seconds * self.config.fps)
            ),
            VideoFrameIterator(
                video_path=pair.imposter_video.path,
                duration_seconds=self.config.impostor_seconds,
                fps=self.config.fps
            )
        ]
        total_frames = sum(it.get_frame_count() for it in iterators)
        logger.debug(
            f"Composed video will have {total_frames} frames "
            f"({iterators[0].get_frame_count()} genuine + "
            f"{iterators[1].get_frame_count()} black + "
            f"{iterators[2].get_frame_count()} imposter)"
        )

        return iterators

    def _get_video_dimensions(self, video_path: Path) -> tuple[int, int]:
        """Get video width and height."""
        cap = cv2.VideoCapture(str(video_path))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        return width, height

    def _build_virtual_path(self, pair: ImposterSamplePair) -> Path:
        """Build virtual path for composed video (for identification only)."""
        genuine_stem = pair.genuine_video.path.stem
        imposter_stem = pair.imposter_video.path.stem
        filename = f"{genuine_stem}_vs_{imposter_stem}.composed"
        return Path(f"<composed>/{filename}")

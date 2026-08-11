"""Service for creating composed imposter videos using frame iterators."""

import cv2
from pathlib import Path
from typing import List

from face_auth.config.logging_config import get_logger
from face_auth.config.models import Participant, ImposterConfig
from face_auth.imposter_video_creation.iterators.frame_iterator import FrameIterator
from face_auth.imposter_video_creation.iterators.impl.black_frame_iterator import BlackFrameGenerator
from face_auth.imposter_video_creation.iterators.impl.video_frame_iterator import VideoFrameIterator
from face_auth.imposter_video_creation.matching.video_matching_strategy import VideoMatchingStrategy
from face_auth.processing.models import ImposterSamplePair, ComposedVideo, Video
from face_auth.services.video_validation import validate_video

logger = get_logger(__name__)


class ImposterVideoCreationService:
    """Service for creating composed imposter videos using frame iterators."""

    def __init__(
            self,
            stitch_config: ImposterConfig,
            matching_strategy: VideoMatchingStrategy,
    ):
        """Initialize with stitching configuration and matching strategy.

        Args:
            stitch_config: Configuration for imposter creation
            matching_strategy: Strategy for pairing genuine videos with imposter videos
        """
        self.config = stitch_config
        self.matching_strategy = matching_strategy

    def create_imposter_videos(
            self,
            all_videos: List[Video],
            genuine_user: Participant,
            allowed_participants: List[Participant],
    ) -> List[ComposedVideo]:
        """Match genuine videos with imposters and compose the resulting videos.

        Args:
            all_videos: All discovered videos
            genuine_user: Genuine user participant
            allowed_participants: Participants allowed to serve as imposters

        Returns:
            List of composed videos (no physical files)

        Raises:
            ValueError: If any video does not meet FPS or duration requirements
        """
        pairs = self.matching_strategy.match(all_videos, genuine_user, allowed_participants)

        logger.info(f"Creating imposter videos from {len(pairs)} pair(s)")

        logger.info("Validating video requirements (FPS, duration)...")
        for pair in pairs:
            validate_video(pair.genuine_video.path, self.config.fps, self.config.genuine_user_seconds)
            validate_video(pair.imposter_video.path, self.config.fps, self.config.impostor_seconds)

        imposter_videos = []
        for i, pair in enumerate(pairs, 1):
            logger.info(
                f"Creating imposter video {i}/{len(pairs)}: "
                f"{pair.genuine_video.path.name} + {pair.imposter_video.path.name}"
            )
            iterators = self._create_iterators(pair)

            imposter_videos.append(ComposedVideo(
                name=self._build_name(pair),
                iterators=iterators
            ))

        return imposter_videos


    def _create_iterators(self, pair):
        """Create frame iterators for a composed video. Currently this is fixed to genuine + black frames + imposter."""
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
        total_frames = sum(len(it) for it in iterators)
        logger.debug(
            f"Composed video will have {total_frames} frames "
            f"({len(iterators[0])} genuine + "
            f"{len(iterators[1])} black + "
            f"{len(iterators[2])} imposter)"
        )

        return iterators

    def _get_video_dimensions(self, video_path: Path) -> tuple[int, int]:
        """Get video width and height."""
        cap = cv2.VideoCapture(str(video_path))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        return width, height

    def _build_name(self, pair: ImposterSamplePair) -> str:
        """Build the identifying name for a composed video."""
        genuine_stem = pair.genuine_video.path.stem
        imposter_stem = pair.imposter_video.path.stem
        return f"{genuine_stem}_vs_{imposter_stem}"

from typing import List

from face_auth.config import Participant
from face_auth.core.processing import Video
from face_auth.core.processing.matching.strategy.video_matching_strategy import VideoMatchingStrategy
from face_auth.core.processing.models import ImposterSamplePair
from face_auth.config.logging_config import get_logger

logger = get_logger(__name__)

class AllVideosMatchingStrategy(VideoMatchingStrategy):
    """Match all imposter videos with each genuine video for in-the-wild study."""

    def match(self, all_videos: List[Video], genuine_user: Participant,
              allowed_participants: List[Participant]) -> List[ImposterSamplePair]:
        """Match genuine user videos with all available imposter videos.

        Args:
            all_videos: All discovered videos
            genuine_user: Genuine user participant
            allowed_participants: List of allowed participants from config

        Returns:
            List of ImposterSamplePair objects
        """
        genuine_videos, imposter_videos = self._separate_genuine_and_imposter(
            all_videos, genuine_user, allowed_participants
        )

        # Match each genuine video with ALL imposters
        pairs = []
        for genuine_video in genuine_videos:
            if not imposter_videos:
                logger.warning(f"No imposter videos available for {genuine_video.path.name}")
                continue

            for imposter_video in imposter_videos:
                pairs.append(ImposterSamplePair(
                    genuine_video=genuine_video,
                    imposter_video=imposter_video
                ))
                logger.info(
                    f"Matched {genuine_video.path.name} with {imposter_video.path.name}"
                )

        return pairs
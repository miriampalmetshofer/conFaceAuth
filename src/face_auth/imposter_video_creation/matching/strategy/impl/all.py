from typing import List

from face_auth.config import Participant
from face_auth.authentication.imposter_video_creation.matching.strategy.video_matching_strategy import VideoMatchingStrategy
from face_auth.processing import Video
from face_auth.config.logging_config import get_logger
from face_auth.processing import ImposterSamplePair

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
            List of ImposterAttackScenario objects
        """
        genuine_videos, imposter_videos = self._separate_genuine_and_imposter(
            all_videos, genuine_user, allowed_participants
        )

        # Match each genuine video with ALL imposters
        scenarios = []
        for genuine_video in genuine_videos:
            if not imposter_videos:
                logger.warning(f"No imposter videos available for {genuine_video.path.name}")
                continue

            for imposter_video in imposter_videos:
                scenarios.append(ImposterSamplePair(
                    genuine_video=genuine_video,
                    imposter_video=imposter_video
                ))
                logger.debug(
                    f"Matched {genuine_video.path.name} with {imposter_video.path.name}"
                )

        logger.info(f"Created {len(scenarios)} match(es) (all imposters)")
        return scenarios
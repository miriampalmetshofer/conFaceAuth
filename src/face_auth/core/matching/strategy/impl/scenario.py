from typing import List

from face_auth.core.matching.strategy.video_matching_strategy import VideoMatchingStrategy
from face_auth.core.processing.models import ControlledStudyVideo, ImposterSamplePair
from face_auth.config.models import Participant
from face_auth.config.logging_config import get_logger

logger = get_logger(__name__)

class ScenarioMatchingStrategy(VideoMatchingStrategy):
    """Match videos by scenario (easy, angle, lighting) for controlled study."""

    def match(self, all_videos: List[ControlledStudyVideo], genuine_user: Participant,
              allowed_participants: List[Participant]) -> List[ImposterSamplePair]:
        """Match genuine user videos with imposter videos of same scenario.

        Args:
            all_videos: All discovered videos (must be ControlledStudyVideo)
            genuine_user: Genuine user participant
            allowed_participants: List of allowed participants from config

        Returns:
            List of ImposterSamplePair objects
        """
        genuine_videos, imposter_videos = self._separate_genuine_and_imposter(
            all_videos, genuine_user, allowed_participants
        )

        # Match each genuine video with imposters of same scenario
        pairs = []
        for genuine_video in genuine_videos:
            matching_imposters = [
                video for video in imposter_videos
                if video.scenario == genuine_video.scenario
            ]

            if not matching_imposters:
                logger.warning(
                    f"No matching imposters found for {genuine_video.path.name} "
                )
            else:
                for matching_imposter in matching_imposters:
                    pairs.append(ImposterSamplePair(
                        genuine_video=genuine_video,
                        imposter_video=matching_imposter
                    ))
                    logger.info(
                        f"Matched {genuine_video.path.name} ({genuine_video.scenario.value}) "
                        f"with {matching_imposter.path.name}"
                    )

        return pairs
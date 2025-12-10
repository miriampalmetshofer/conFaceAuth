"""Video matching strategies for pairing genuine videos with imposter samples."""
from abc import ABC, abstractmethod
from typing import Protocol, List
from face_auth.core.processing.models import Video, ControlledStudyVideo, ImposterSamplePair
from face_auth.config.logging_config import get_logger

logger = get_logger(__name__)


class VideoMatchingStrategy(ABC):
    """Protocol for video matching strategies."""

    @abstractmethod
    def match(self, all_videos: List[Video], genuine_user_name: str) -> List[ImposterSamplePair]:
        """Match genuine user videos with imposter videos.

        Args:
            all_videos: All discovered videos
            genuine_user_name: Name of the genuine user

        Returns:
            List of ImposterSamplePair objects
        """

    def separate_genuine_and_imposter(self, all_videos, genuine_user_name):
        genuine_videos = []
        imposter_videos = []
        for video in all_videos:
            if video.participant.name == genuine_user_name:
                genuine_videos.append(video)
            else:
                imposter_videos.append(video)
        logger.info(f"Found {len(genuine_videos)} genuine video(s) for {genuine_user_name}")
        logger.info(f"Found {len(imposter_videos)} potential imposter video(s)")
        return genuine_videos, imposter_videos



class ScenarioMatchingStrategy(VideoMatchingStrategy):
    """Match videos by scenario (easy, angle, lighting)."""

    def match(self, all_videos: List[ControlledStudyVideo], genuine_user_name: str) -> List[ImposterSamplePair]:
        """Match genuine user videos with imposter videos of same scenario.

        Args:
            all_videos: All discovered videos
            genuine_user_name: Name of the genuine user

        Returns:
            List of ImposterSamplePair objects, one per genuine video
        """
        genuine_videos, imposter_videos = self.separate_genuine_and_imposter(all_videos, genuine_user_name)

        # Match each genuine video with imposters of same scenario
        pairs = []
        for genuine_video in genuine_videos:

            matching_imposters = [
                video for video in imposter_videos
                if video.scenario == genuine_video.scenario
            ]

            if not matching_imposters:
                # Debug: show what imposters exist
                imposter_scenarios = {f"{v.participant.name} ({v.scenario.value})" for v in imposter_videos}
                logger.warning(
                    f"No matching imposters found for {genuine_video.path.name} "
                    f"(scenario: {genuine_video.scenario.value}). "
                    f"Available imposters: {', '.join(sorted(imposter_scenarios)) if imposter_scenarios else 'none'}"
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



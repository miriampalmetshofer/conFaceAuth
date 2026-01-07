"""Video matching strategies for pairing genuine videos with imposter samples."""
import random
from abc import ABC, abstractmethod
from typing import List
from face_auth.core.processing.models import Video, ControlledStudyVideo, ImposterSamplePair
from face_auth.config.models import Participant
from face_auth.config.logging_config import get_logger

logger = get_logger(__name__)


class VideoMatchingStrategy(ABC):
    """Protocol for video matching strategies."""

    @abstractmethod
    def match(self, all_videos: List[Video], genuine_user: Participant,
              allowed_participants: List[Participant]) -> List[ImposterSamplePair]:
        """Match genuine user videos with imposter videos.

        Args:
            all_videos: All discovered videos
            genuine_user: Genuine user participant
            allowed_participants: List of allowed participants from config

        Returns:
            List of ImposterSamplePair objects
        """

    def _separate_genuine_and_imposter(self, all_videos, genuine_user, allowed_participants):
        genuine_videos = []
        imposter_videos = []

        # Extract names for comparison
        genuine_name = genuine_user.name
        allowed_names = {p.name for p in allowed_participants}

        for video in all_videos:
            if video.participant.name == genuine_name:
                genuine_videos.append(video)
            elif video.participant.name in allowed_names:
                imposter_videos.append(video)

        logger.info(f"Found {len(genuine_videos)} genuine video(s) for {genuine_name}")
        logger.info(f"Found {len(imposter_videos)} potential imposter video(s)")
        logger.info(f"Filtered to participants: {', '.join(sorted(allowed_names))}")

        return genuine_videos, imposter_videos



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


class RandomSamplingMatchingStrategy(VideoMatchingStrategy):
    """Match each genuine video with K randomly selected imposters."""

    def __init__(self, imposters_per_genuine: int, random_seed: int = 42):
        """Initialize random sampling strategy.

        Args:
            imposters_per_genuine: Number of imposters to match with each genuine video
            random_seed: Seed for reproducibility (default: 42)
        """
        self.imposters_per_genuine = imposters_per_genuine
        random.seed(random_seed)

    def match(self, all_videos: List[Video], genuine_user: Participant,
              allowed_participants: List[Participant]) -> List[ImposterSamplePair]:
        """Match genuine user videos with K randomly selected imposter videos.

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

        pairs = []
        for genuine_video in genuine_videos:
            if not imposter_videos:
                logger.warning(f"No imposter videos available for {genuine_video.path.name}")
                continue

            num_to_sample = min(self.imposters_per_genuine, len(imposter_videos))
            sampled_imposters = random.sample(imposter_videos, num_to_sample)

            for imposter_video in sampled_imposters:
                pairs.append(ImposterSamplePair(
                    genuine_video=genuine_video,
                    imposter_video=imposter_video
                ))

            logger.info(
                f"Matched {genuine_video.path.name} with {num_to_sample} random imposters "
                f"(requested: {self.imposters_per_genuine}, available: {len(imposter_videos)})"
            )

        return pairs

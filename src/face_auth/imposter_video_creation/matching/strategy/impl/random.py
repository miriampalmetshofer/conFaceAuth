from typing import List

from face_auth.config import Participant
from face_auth.authentication.imposter_video_creation.matching.strategy.video_matching_strategy import VideoMatchingStrategy
from face_auth.processing import Video
from face_auth.processing import ImposterSamplePair
from face_auth.config.logging_config import get_logger

logger = get_logger(__name__)


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

            logger.debug(
                f"Matched {genuine_video.path.name} with {num_to_sample} random imposters "
                f"(requested: {self.imposters_per_genuine}, available: {len(imposter_videos)})"
            )

        logger.info(f"Created {len(pairs)} random match(es)")
        return pairs
"""Stage for matching genuine user videos with imposter videos."""

from typing import List

from face_auth.config import Participant
from face_auth.config.logging_config import get_logger
from face_auth.core.processing.models import Video, ImposterSamplePair
from face_auth.core.processing.video_matching import VideoMatchingStrategy

logger = get_logger(__name__)


class VideoMatchingStage:
    """Stage 2: Match genuine user videos with appropriate imposter videos."""

    def __init__(self, matching_strategy: VideoMatchingStrategy):
        """Initialize with matching strategy.

        Args:
            matching_strategy: Strategy for matching videos
        """
        self.matching_strategy = matching_strategy

    def execute(self, all_videos: List[Video], genuine_user: Participant,
                allowed_participants: List[Participant]) -> List[ImposterSamplePair]:
        """Match genuine user videos with imposter videos.

        Args:
            all_videos: All discovered videos from device
            genuine_user: Participant for genuine user
            allowed_participants: List of all allowed participants from config

        Returns:
            List of imposter sample pairs

        Raises:
            ValueError: If no genuine user videos found or no matches created
        """
        logger.info(f"Matching videos for genuine user: {genuine_user.name}")

        pairs = self.matching_strategy.match(all_videos, genuine_user, allowed_participants)

        if not pairs:
            raise ValueError(
                f"No imposter sample pairs created for {genuine_user.name}. "
                "Ensure genuine user has videos and matching imposters exist."
            )

        logger.info(f"Created {len(pairs)} imposter sample pair(s)")
        return pairs

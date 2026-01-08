"""Video matching strategy for pairing genuine videos with imposter samples."""
import random
from abc import ABC, abstractmethod
from typing import List
from face_auth.core.processing.models import Video, ImposterSamplePair
from face_auth.config.models import Participant
from face_auth.config.logging_config import get_logger

logger = get_logger(__name__)


class VideoMatchingStrategy(ABC):
    """Protocol for video matching strategy."""

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




import os
import glob
from datetime import datetime

from face_auth.models import ParticipantInfo
from face_auth.processing.models import VideoInfo, Scenario
from face_auth.utils.logging_config import get_logger

logger = get_logger(__name__)


class VideoDiscovery:
    """Discovers videos from filesystem for a participant and parses metadata."""

    def __init__(self, participant: ParticipantInfo):
        """Initialize video discovery for a participant."""
        self.participant = participant

    def discover_videos(self, base_path: str) -> list[VideoInfo]:
        """Discover all videos for the participant.

        Args:
            base_path: Base directory containing participant videos (e.g., "controlled_study/desktop")

        Returns:
            List of VideoInfo objects with parsed metadata
        """
        device_folder = os.path.join(base_path, self.participant.device)

        # Find all .mp4 and .MP4 files for this participant
        video_patterns = [
            os.path.join(device_folder, f"{self.participant.name}_*.mp4"),
            os.path.join(device_folder, f"{self.participant.name}_*.MP4"),
        ]

        video_paths = []
        for pattern in video_patterns:
            video_paths.extend(glob.glob(pattern))
        logger.info(f"Found {len(video_paths)} videos for {self.participant.name} ({self.participant.device})")

        video_infos = self._parse_video_files(video_paths)

        return video_infos

    def _parse_video_files(self, video_paths) -> list[VideoInfo]:
        """Parse video filenames to extract metadata."""
        video_infos = []
        for video_path in video_paths:
            try:
                video_info = self._extract_info_from_video(video_path)
                video_infos.append(video_info)
            except ValueError as e:
                logger.warning(f"Could not parse filename: {os.path.basename(video_path)} - {e}")
        return video_infos

    def _extract_info_from_video(self, video_path: str) -> VideoInfo:
        """Parse video filename to extract metadata.

        Expected format: {name}_{scenario}_{YYYY-MM-DD_HH-MM-SS}.mp4
        Example: miriam_easy_2025-10-31_08-01-47.mp4
        """
        parts = self._split_filename(os.path.basename(video_path))
        scenario = self._parse_scenario(parts[1])
        recording_date = self._parse_date(parts[2])

        return VideoInfo(
            path=video_path,
            participant=self.participant,
            scenario=scenario,
            recording_date=recording_date
        )

    def _split_filename(self, filename: str) -> list[str]:
        """Split filename and validate it has the expected number of parts."""
        name_without_ext = os.path.splitext(filename)[0]
        parts = name_without_ext.split('_')

        if len(parts) < 4:
            raise ValueError(f"Filename has too few parts (expected at least 4): {filename}")

        return parts

    def _parse_scenario(self, scenario_str: str) -> Scenario:
        """Convert scenario string to Scenario enum."""
        try:
            return Scenario(scenario_str.lower())
        except ValueError:
            raise ValueError(f"Invalid scenario '{scenario_str}'")

    def _parse_date(self, date_str: str) -> datetime.date:
        """Parse date from YYYY-MM-DD format."""
        try:
            return datetime.strptime(date_str, "%Y-%m-%d").date()
        except ValueError as e:
            raise ValueError(f"Could not parse date '{date_str}'") from e

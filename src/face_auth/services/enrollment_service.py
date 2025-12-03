"""Enrollment service for managing participant enrollment data."""

import os
from typing import List
import numpy as np

from face_auth.config.models import EnrollmentConfig, PathsConfig
from face_auth.config.models import ParticipantConfig
from face_auth.core.embedder import Embedder
from face_auth.detection import FaceDetector, FaceExtractor
from face_auth.processing.video_discovery import VideoDiscovery
from face_auth.processing.video_parser import EnrollmentVideoParser
from face_auth.enrollment import (
    EnrollmentOrchestrator,
    VideoFrameExtractor,
    HeadPoseEstimator,
    DirectionClassifier,
    NormalDistributionSampler,
    EnrollmentFrameSaver
)
from face_auth.io import EnrollmentLoader
from face_auth.services.models import EnrollmentData
from face_auth.utils.logging_config import get_logger

logger = get_logger(__name__)


class EnrollmentException(Exception):
    """Exception raised for enrollment-related errors."""
    pass


class EnrollmentService:
    """Handles all enrollment-related operations."""

    def __init__(
        self,
        config: EnrollmentConfig,
        paths_config: PathsConfig,
        face_detector: FaceDetector,
        face_extractor: FaceExtractor,
        embedder: Embedder
    ):
        """Initialize enrollment service.

        Args:
            config: Enrollment configuration
            paths_config: Paths configuration
            face_detector: Face detector instance
            face_extractor: Face extractor instance
            embedder: Embedding generator instance
        """
        self.config = config
        self.paths = paths_config
        self.detector = face_detector
        self.extractor = face_extractor
        self.embedder = embedder

    def ensure_enrollment(
        self,
        participant: ParticipantConfig,
        device: str
    ) -> EnrollmentData:
        """Ensure enrollment exists, create if needed.

        Args:
            participant: Participant information
            device: Device identifier

        Returns:
            EnrollmentData with folder path and embeddings

        Raises:
            EnrollmentException: If enrollment cannot be created or loaded
        """
        enrollment_folder = self._get_enrollment_folder(participant, device)

        if self._enrollment_exists(enrollment_folder):
            logger.info(f"Loading existing enrollment for {participant.name} ({device})")
            return self._load_enrollment(enrollment_folder)
        else:
            logger.info(f"=== ENROLLING: {participant.name} ({device}) ===")
            return self._create_enrollment(participant, device, enrollment_folder)

    def _get_enrollment_folder(self, participant: ParticipantConfig, device: str) -> str:
        """Get enrollment folder path for participant."""
        participant_folder = os.path.join(
            self.paths.enrollment_base_path,
            device,
            participant.name
        )

        # Find enrollment video to determine folder name
        discovery = VideoDiscovery(participant, EnrollmentVideoParser())
        enrollment_videos = discovery.discover(participant_folder)

        if not enrollment_videos:
            raise EnrollmentException(
                f"\n{'!' * 60}\n"
                f"ERROR: No enrollment video found!\n"
                f"Searched in: {participant_folder}\n"
                f"Expected pattern: {participant.name}_enrollment_*\n"
                f"Participant: '{participant.name}' | Device: '{device}'\n"
                f"{'!' * 60}\n"
            )

        enrollment_video = enrollment_videos[0]
        video_name = os.path.splitext(os.path.basename(enrollment_video.path))[0]
        return os.path.join(participant_folder, video_name)

    def _enrollment_exists(self, enrollment_folder: str) -> bool:
        """Check if enrollment folder exists and contains images."""
        if not os.path.exists(enrollment_folder):
            return False

        try:
            files = os.listdir(enrollment_folder)
            return any(f.endswith(".jpg") for f in files)
        except Exception:
            return False

    def _create_enrollment(
        self,
        participant: ParticipantConfig,
        device: str,
        enrollment_folder: str
    ) -> EnrollmentData:
        """Create new enrollment.

        Args:
            participant: Participant information
            device: Device identifier
            enrollment_folder: Path to enrollment folder

        Returns:
            EnrollmentData with created enrollment

        Raises:
            EnrollmentException: If enrollment creation fails
        """
        # Discover enrollment video
        participant_folder = os.path.join(
            self.paths.enrollment_base_path,
            device,
            participant.name
        )
        discovery = VideoDiscovery(participant, EnrollmentVideoParser())
        enrollment_videos = discovery.discover(participant_folder)

        if not enrollment_videos:
            raise EnrollmentException("No enrollment video found")

        enrollment_video = enrollment_videos[0]
        logger.info(f"Using enrollment video: {enrollment_video.filename}")

        # Build enrollment orchestrator
        orchestrator = self._build_enrollment_orchestrator()

        # Process enrollment video
        orchestrator.process_enrollment_video(
            enrollment_video.path,
            self.config.frames_per_direction,
            enrollment_folder
        )

        # Load and return
        return self._load_enrollment(enrollment_folder)

    def _load_enrollment(self, folder: str) -> EnrollmentData:
        """Load existing enrollment from folder.

        Args:
            folder: Path to enrollment folder

        Returns:
            EnrollmentData with loaded embeddings

        Raises:
            EnrollmentException: If loading fails
        """
        try:
            loader = EnrollmentLoader(self.embedder, self.detector, self.extractor)
            embeddings = loader.load_embeddings(folder)
            return EnrollmentData(folder=folder, embeddings=embeddings)
        except Exception as e:
            raise EnrollmentException(f"Failed to load enrollment: {e}")

    def _build_enrollment_orchestrator(self) -> EnrollmentOrchestrator:
        """Build enrollment orchestrator with configured components."""
        return EnrollmentOrchestrator(
            frame_extractor=VideoFrameExtractor(self.config.frame_sampling_interval),
            pose_estimator=HeadPoseEstimator(),
            direction_classifier=DirectionClassifier(
                self.config.yaw_threshold,
                self.config.pitch_threshold
            ),
            frame_sampler=NormalDistributionSampler(
                self.config.distribution_mean_fraction,
                self.config.distribution_stddev_fraction,
                self.config.sampling_seed
            ),
            frame_saver=EnrollmentFrameSaver()
        )

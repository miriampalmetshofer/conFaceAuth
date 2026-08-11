"""Enrollment service for managing participant enrollment data."""

from collections import Counter
from pathlib import Path
import shutil
from typing import List

import numpy as np

from face_auth.config.models import EnrollmentConfig, PathsConfig, ProcessingContext, EnrollmentVideoPreference
from face_auth.core.embedder.backend.embedder_backend import EmbedderBackend
from face_auth.processing.video_parser.impl.enrollment_video_parser import EnrollmentVideoParser
from face_auth.processing.models import EnrollmentVideo
from face_auth.core.enrollment.backend.enrollment_factory import create_enrollment_backend
from face_auth.core.enrollment.models import SelectedEnrollmentFrame
from face_auth.services.models import EnrollmentData
from face_auth.processing.video_utils import save_image_rgb, load_image_rgb
from face_auth.config.logging_config import get_logger

logger = get_logger(__name__)


class EnrollmentService:
    """Handles all enrollment-related operations."""

    def __init__(
        self,
        config: EnrollmentConfig,
        embedder: EmbedderBackend
    ):
        """Initialize enrollment service.

        Args:
            config: Enrollment configuration
            embedder: Embedding generator instance
        """
        self.config = config
        self.embedder = embedder

    def enroll_participant(
        self,
        context: ProcessingContext,
        paths: PathsConfig,
    ) -> EnrollmentData:
        """Resolve a participant's enrollment folder from the study paths and enroll them.

        Discovers the participant's videos by the study path convention, applies the
        re-enrollment cache policy, then delegates to the path-explicit core.

        Args:
            context: Processing context with participant and device
            paths: Paths configuration locating the study's enrollment videos

        Returns:
            EnrollmentData with folder path and embeddings

        Raises:
            ValueError: If enrollment cannot be created or loaded
        """
        discovered_videos = self._discover_enrollment_videos(context, paths)

        selected_videos = self._filter_enrollment_videos(
            discovered_videos,
            self.config.enrollment_video_preference
        )

        enrollment_folder = self._get_enrollment_folder(context, selected_videos, paths)

        if self.config.force_reenrollment and enrollment_folder.exists():
            logger.info(f"Deleting existing enrollment folder: {enrollment_folder}")
            shutil.rmtree(enrollment_folder)

        if self._enrollment_exists(enrollment_folder):
            logger.info(f"Loading existing enrollment for {context.participant.name} ({context.device})")
            return self.load_enrollment(enrollment_folder)
        else:
            logger.info(f"=== ENROLLING: {context.participant.name} ({context.device}) ===")
            return self.enroll_from_videos(selected_videos, enrollment_folder)

    def _discover_enrollment_videos(
        self,
        context: ProcessingContext,
        paths: PathsConfig,
    ) -> List[EnrollmentVideo]:
        """Discover enrollment videos for participant.

        Args:
            context: Processing context with participant and device
            paths: Paths configuration locating the study's enrollment videos

        Returns:
            List of enrollment videos found

        Raises:
            ValueError: If no enrollment videos are found
        """
        participant_folder = paths.enrollment_base_path / context.device.value / context.participant.name
        enrollment_videos = EnrollmentVideoParser().parse_folder(participant_folder)

        if not enrollment_videos:
            raise ValueError(
                f"\n{'!' * 60}\n"
                f"ERROR: No enrollment video found!\n"
                f"Searched in: {participant_folder}\n"
                f"Expected pattern: {context.participant.name}_enrollment_*\n"
                f"Participant: '{context.participant.name}' | Device: '{context.device}'\n"
                f"{'!' * 60}\n"
            )

        return enrollment_videos

    def _get_enrollment_folder(
        self,
        context: ProcessingContext,
        enrollment_videos: List[EnrollmentVideo],
        paths: PathsConfig,
    ) -> Path:
        """Get enrollment folder path from videos.

        Args:
            context: Processing context with participant and device
            enrollment_videos: Selected enrollment videos
            paths: Paths configuration locating the study's enrollment videos

        Returns:
            Path to enrollment folder
        """
        participant_folder = paths.enrollment_base_path / context.device.value / context.participant.name

        # Extract unique scenarios and rotations from selected videos
        scenarios = sorted(set(v.scenario.value for v in enrollment_videos))
        rotations = sorted(set(v.head_rotation.value for v in enrollment_videos))

        # Build predictable folder name: {name}_enrollment_{scenarios}_{rotations}
        scenarios_str = "+".join(scenarios)
        rotations_str = "+".join(rotations)

        folder_name = f"{context.participant.name}_enrollment_{scenarios_str}_{rotations_str}"
        return participant_folder / folder_name

    def _filter_enrollment_videos(
        self,
        available_videos: List[EnrollmentVideo],
        preference: EnrollmentVideoPreference
    ) -> List[EnrollmentVideo]:
        """Select enrollment videos based on configured preference.

        Args:
            available_videos: List of all available enrollment videos
            preference: Configuration specifying scenario and rotations to use

        Returns:
            List of selected enrollment videos matching preference

        Raises:
            ValueError: If any required video is missing
        """
        selected_videos = []

        # Select videos for each combination of scenario and rotation
        for scenario in preference.scenarios:
            for rotation in preference.rotations:
                matching_videos = [
                    v for v in available_videos
                    if v.scenario == scenario and v.head_rotation == rotation
                ]
                if not matching_videos:
                    raise ValueError(
                        f"No enrollment video found for scenario '{scenario.value}' with rotation '{rotation.value}'"
                    )
                # Use first match if multiple videos exist for same scenario+rotation
                selected_videos.append(matching_videos[0])

        logger.info(
            f"Selected {len(selected_videos)} enrollment video(s): "
            f"{', '.join([v.path.name for v in selected_videos])}"
        )

        return selected_videos

    def _enrollment_exists(self, enrollment_folder: Path) -> bool:
        """Check if the enrollment folder contains any images."""
        return any(enrollment_folder.glob("*.jpg"))

    def enroll_from_videos(
        self,
        enrollment_videos: List[EnrollmentVideo],
        enrollment_folder: Path
    ) -> EnrollmentData:
        """Build a new enrollment from the given videos into a folder.

        Selects enrollment frames, saves them to the folder, and returns the
        resulting embeddings. Takes explicit videos and folder, so it does not
        depend on the study path convention.

        Args:
            enrollment_videos: Enrollment videos to process
            enrollment_folder: Folder to write the enrollment images into

        Returns:
            EnrollmentData with created enrollment

        Raises:
            ValueError: If enrollment creation fails
        """
        logger.info(
            f"Processing {len(enrollment_videos)} enrollment video(s): "
            f"{', '.join([v.path.name for v in enrollment_videos])}"
        )

        backend = create_enrollment_backend(self.config)
        selections = backend.select_frames(
            enrollment_videos=enrollment_videos,
            frames_per_direction_per_video=self.config.frames_per_direction,
        )
        self._save_frames(selections, enrollment_folder)

        return self.load_enrollment(enrollment_folder)

    def load_enrollment(self, folder: Path) -> EnrollmentData:
        """Load existing enrollment from folder.

        Args:
            folder: Path to enrollment folder

        Returns:
            EnrollmentData with loaded embeddings

        Raises:
            ValueError: If loading fails
        """
        try:
            embeddings = self._load_embeddings(folder)
            return EnrollmentData(folder=folder, embeddings=embeddings)
        except Exception as e:
            raise ValueError(f"Failed to load enrollment: {e}")

    def _load_embeddings(self, enrollment_folder: Path) -> List[np.ndarray]:
        """Load enrollment images from a folder and compute their embeddings.

        Raises:
            FileNotFoundError: If the folder is missing or contains no images
        """
        if not any(enrollment_folder.glob("*.jpg")):
            raise FileNotFoundError(
                f"No images found in the enrollment folder: {enrollment_folder}. "
                f"Please ensure the folder exists and contains images."
            )

        image_files = sorted(enrollment_folder.glob("*.jpg"))
        logger.info(f"Found {len(image_files)} images in the enrollment folder")

        embeddings = []
        for image_path in image_files:
            embedding = self._embed_image(image_path)
            if embedding is not None:
                embeddings.append(embedding)

        logger.info(f"Successfully computed {len(embeddings)} enrollment embeddings")
        return embeddings

    def _embed_image(self, image_path: Path) -> np.ndarray | None:
        """Load a single enrollment image and return its embedding, or None if unusable."""
        image_rgb = load_image_rgb(image_path)
        if image_rgb is None:
            logger.warning(f"Could not read image {image_path}. Skipping")
            return None

        embedding_result = self.embedder.embed(image_rgb)
        if not embedding_result.face_detected:
            logger.warning(f"No face detected in enrollment image {image_path}. Skipping")
            return None

        return embedding_result.embedding

    def _save_frames(
        self,
        selections: list[SelectedEnrollmentFrame],
        output_folder: Path,
    ) -> None:
        """Save selected frames named by head direction into the enrollment folder."""
        output_folder.mkdir(parents=True, exist_ok=True)

        counters = Counter()
        for selection in selections:
            direction = selection.assigned_direction
            counters[direction] += 1
            source_name = selection.frame.source_name or "frame"

            filename = (
                f"{source_name}_"
                f"{direction.value}_"
                f"{counters[direction] - 1:03d}.jpg"
            )
            save_image_rgb(selection.image_rgb, output_folder / filename)

        logger.info(f"Saved {len(selections)} enrollment frames to {output_folder}")

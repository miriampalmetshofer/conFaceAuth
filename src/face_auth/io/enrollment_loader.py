import os
import cv2
import numpy as np

from face_auth.core.embedder import Embedder
from face_auth.detection import FaceDetector, FaceExtractor
from face_auth.utils.logging_config import get_logger

logger = get_logger(__name__)


class EnrollmentLoader:
    """Loads enrollment images from disc and computes their embeddings."""

    def __init__(
        self,
        embedder: Embedder,
        face_detector: FaceDetector,
        face_extractor: FaceExtractor
    ):
        """Initialize embedding loader.

        Args:
            embedder: Embedder instance for generating embeddings
            face_detector: FaceDetector instance for detecting faces
            face_extractor: FaceExtractor instance for extracting face regions
        """
        self.embedder = embedder
        self.face_detector = face_detector
        self.face_extractor = face_extractor

    def load_embeddings(self, enrollment_folder: str) -> list[np.ndarray]:
        """Load enrollment images and compute their embeddings."""
        self._validate_enrollment_folder(enrollment_folder)

        image_files = os.listdir(enrollment_folder)
        logger.info(f"Found {len(image_files)} images in the enrollment folder")

        embeddings = []
        for filename in image_files:
            image_path = os.path.join(enrollment_folder, filename)
            embedding = self._get_enrollment_embedding(image_path)
            if embedding is not None:
                embeddings.append(embedding)

        logger.info(f"Successfully computed {len(embeddings)} enrollment embeddings")

        return embeddings

    def _validate_enrollment_folder(self, enrollment_folder: str) -> None:
        """Validate that enrollment folder exists and is not empty."""
        if not os.path.exists(enrollment_folder) or not os.listdir(enrollment_folder):
            raise FileNotFoundError(
                f"No images found in the enrollment folder: {enrollment_folder}. "
                f"Please ensure the folder exists and contains images."
            )

    def _get_enrollment_embedding(self, image_path: str) -> np.ndarray | None:
        """Process a single enrollment image and return its embedding."""
        image = cv2.imread(image_path)
        if image is None:
            logger.warning(f"Could not read image {image_path}. Skipping")
            return None

        detection_result = self.face_extractor.detect_and_extract(image, self.face_detector)
        if detection_result is None:
            logger.warning(f"No face detected in enrollment image {image_path}. Skipping")
            return None

        embedding = self.embedder.get_embedding(detection_result.face_image)
        if embedding is None:
            logger.warning(f"Failed to compute embedding for {image_path}. Skipping")
            return None

        return embedding

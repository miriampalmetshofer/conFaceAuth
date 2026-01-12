import cv2
import numpy as np
from pathlib import Path

from face_auth.config.logging_config import get_logger
from face_auth.core.embedder import Embedder

logger = get_logger(__name__)


class EnrollmentLoader:
    """Loads enrollment images from disc and computes their embeddings."""

    def __init__(self, embedder: Embedder):
        """Initialize embedding loader.

        Args:
            embedder: Embedder instance for generating embeddings
        """
        self.embedder = embedder

    def load_embeddings(self, enrollment_folder: Path) -> list[np.ndarray]:
        """Load enrollment images and compute their embeddings."""
        self._validate_enrollment_folder(enrollment_folder)

        image_files = list(enrollment_folder.iterdir())
        logger.info(f"Found {len(image_files)} images in the enrollment folder")

        embeddings = []
        for image_path in image_files:
            embedding = self._get_enrollment_embedding(image_path)
            if embedding is not None:
                embeddings.append(embedding)

        logger.info(f"Successfully computed {len(embeddings)} enrollment embeddings")

        return embeddings

    def _validate_enrollment_folder(self, enrollment_folder: Path) -> None:
        """Validate that enrollment folder exists and is not empty."""
        if not enrollment_folder.exists() or not any(enrollment_folder.iterdir()):
            raise FileNotFoundError(
                f"No images found in the enrollment folder: {enrollment_folder}. "
                f"Please ensure the folder exists and contains images."
            )

    def _get_enrollment_embedding(self, image_path: Path) -> np.ndarray | None:
        """Process a single enrollment image and return its embedding."""
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            logger.warning(f"Could not read image {image_path}. Skipping")
            return None

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        embedding_result = self.embedder.get_embedding(image_rgb)

        if not embedding_result.face_detected:
            logger.warning(f"No face detected in enrollment image {image_path}. Skipping")
            return None

        return embedding_result.embedding

import os
import cv2
from face_auth.logging_config import get_logger

logger = get_logger(__name__)


class EnrollmentService:
    """Handles loading and processing enrollment images to generate embeddings."""

    def __init__(self, embedder, face_detector):
        self.embedder = embedder
        self.face_detector = face_detector

    def load_enrollment_embeddings(self, enrollment_folder: str) -> list:
        """Load enrollment images and compute their embeddings."""
        if not os.path.exists(enrollment_folder) or not os.listdir(enrollment_folder):
            raise FileNotFoundError(
                f"No images found in the enrollment folder: {enrollment_folder}. "
                f"Please ensure the folder exists and contains images."
            )

        embeddings = []
        image_files = os.listdir(enrollment_folder)
        logger.info(f"Found {len(image_files)} images in the enrollment folder")

        for filename in image_files:
            image_path = os.path.join(enrollment_folder, filename)
            image = cv2.imread(image_path)

            if image is None:
                logger.warning(f"Could not read image {filename}. Skipping")
                continue

            result = self.face_detector.detect_and_crop(image)

            if result is None:
                logger.warning(f"No face detected in enrollment image {filename}. Skipping")
                continue

            face, _ = result
            embedding = self.embedder.get_embedding(face)

            if embedding is not None:
                embeddings.append(embedding)
            else:
                logger.warning(f"Failed to compute embedding for {filename}. Skipping")

        logger.info(f"Successfully computed {len(embeddings)} enrollment embeddings")
        return embeddings
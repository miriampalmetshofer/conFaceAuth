import numpy as np
from pathlib import Path

from face_auth.config.logging_config import get_logger
from face_auth.core.embedder.backend.embedder_backend import EmbedderBackend
from face_auth.processing.video_utils import load_image_rgb

logger = get_logger(__name__)


def load_enrollment_embeddings(
    enrollment_folder: Path,
    embedder: EmbedderBackend,
) -> list[np.ndarray]:
    """Load enrollment images from disk and compute their embeddings.

    Args:
        enrollment_folder: Folder containing enrollment images
        embedder: Backend for generating embeddings

    Returns:
        Embeddings for every image that yielded a detected face

    Raises:
        FileNotFoundError: If the folder is missing or contains no images
    """
    _validate_enrollment_folder(enrollment_folder)

    image_files = sorted(enrollment_folder.glob("*.jpg"))
    logger.info(f"Found {len(image_files)} images in the enrollment folder")

    embeddings = []
    for image_path in image_files:
        embedding = _embed_enrollment_image(image_path, embedder)
        if embedding is not None:
            embeddings.append(embedding)

    logger.info(f"Successfully computed {len(embeddings)} enrollment embeddings")

    return embeddings


def _validate_enrollment_folder(enrollment_folder: Path) -> None:
    """Validate that the enrollment folder exists and is not empty."""
    if not enrollment_folder.exists() or not any(enrollment_folder.glob("*.jpg")):
        raise FileNotFoundError(
            f"No images found in the enrollment folder: {enrollment_folder}. "
            f"Please ensure the folder exists and contains images."
        )


def _embed_enrollment_image(image_path: Path, embedder: EmbedderBackend) -> np.ndarray | None:
    """Load a single enrollment image and return its embedding, or None if unusable."""
    image_rgb = load_image_rgb(image_path)
    if image_rgb is None:
        logger.warning(f"Could not read image {image_path}. Skipping")
        return None

    embedding_result = embedder.embed(image_rgb)

    if not embedding_result.face_detected:
        logger.warning(f"No face detected in enrollment image {image_path}. Skipping")
        return None

    return embedding_result.embedding

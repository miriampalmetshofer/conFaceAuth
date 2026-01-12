from typing import Iterator
import numpy as np
from face_auth.core.imposter_video_creation.frame_iterator import FrameIterator
from face_auth.config.logging_config import get_logger

logger = get_logger(__name__)

class BlackFrameGenerator(FrameIterator):
    """Generates black frames matching specified dimensions."""

    def __init__(self, width: int, height: int, num_frames: int):
        """Initialize black frame generator.

        Args:
            width: Frame width in pixels
            height: Frame height in pixels
            num_frames: Number of black frames to generate
        """
        self.width = width
        self.height = height
        self.num_frames = num_frames

        logger.debug(
            f"BlackFrameGenerator: {width}x{height}, "
            f"{num_frames} frames"
        )

    def __iter__(self) -> Iterator[np.ndarray]:
        """Yield black frames."""
        black_frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        for _ in range(self.num_frames):
            yield black_frame.copy()

    def get_frame_count(self) -> int:
        """Return total number of frames this iterator will yield."""
        return self.num_frames

    def get_source_name(self) -> str:
        """Return name of the source (black frames have no source)."""
        return "black_frames"

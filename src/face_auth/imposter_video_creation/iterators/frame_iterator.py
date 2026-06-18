"""Frame iterators for composing video streams without physical file creation."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Iterator, Optional
import numpy as np
from face_auth.config.logging_config import get_logger

logger = get_logger(__name__)


class FrameIterator(ABC):
    """Abstract base for frame iterators that yield video frames."""

    @abstractmethod
    def __iter__(self) -> Iterator[np.ndarray]:
        """Return iterator that yields frames."""
        pass

    @abstractmethod
    def get_frame_count(self) -> int:
        """Return total number of frames this iterator will yield."""
        pass

    @abstractmethod
    def get_source_name(self) -> str:
        """Return name of the source (for debugging/logging)."""
        pass

    @property
    @abstractmethod
    def video_path(self) -> Optional[Path]:
        """Return the video path if this iterator reads from a file, None otherwise."""
        pass

    def get_source_frame_index(self, iterator_offset: int) -> Optional[int]:
        """Return the zero-based source frame index for an iterator offset."""
        return None

    def get_embedding_cache_metadata(self) -> Dict[str, Any]:
        """Return source metadata that affects the pixels sent to the embedder."""
        return {}

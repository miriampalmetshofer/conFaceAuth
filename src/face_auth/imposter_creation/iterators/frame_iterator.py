"""Frame iterators for composing video streams without physical file creation."""

from abc import ABC, abstractmethod
from typing import Iterator
import numpy as np
from face_auth.logging_config import get_logger

logger = get_logger(__name__)


class FrameIterator(ABC):
    """Abstract base for frame iterators that yield video frames."""

    @abstractmethod
    def __iter__(self) -> Iterator[np.ndarray]:
        """Return iterator that yields frames."""
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Return total number of frames this iterator will yield."""
        pass

    @property
    @abstractmethod
    def source_name(self) -> str:
        """Name of the source (for debugging/logging)."""
        pass

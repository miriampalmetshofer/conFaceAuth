"""Base interface for video parsers."""
from abc import ABC, abstractmethod
from pathlib import Path

from face_auth.processing.models import Video


class VideoParser(ABC):
    """Pluggable parser for a specific type of video file."""

    @abstractmethod
    def matches(self, path: Path) -> bool:
        """Return true if the file matches this parser."""

    @abstractmethod
    def parse(self, path: Path) -> Video:
        """Parse the file and return a Video instance."""

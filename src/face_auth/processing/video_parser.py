"""Plugin architecture for parsing different video types."""
import re
from abc import ABC, abstractmethod
from pathlib import Path
from datetime import datetime

from face_auth.processing.models import Video, EnrollmentVideo, Scenario, HeadRotation


class VideoParser(ABC):
    """Pluggable parser for a specific type of video file."""

    @abstractmethod
    def matches(self, path: Path) -> bool:
        """Return true if this parser can parse this file."""

    @abstractmethod
    def parse(self, path: Path) -> Video:
        """Parse the file and return a Video instance."""


class UsageVideoParser(VideoParser):
    """Parser for regular videos: {name}_{scenario}_{date}_{time}.mp4"""

    regex = re.compile(
        r"^(?P<name>[^_]+)_(?P<scenario>[^_]+)_(?P<date>\d{4}-\d{2}-\d{2})_(?P<time>\d{2}-\d{2}-\d{2})$"
    )

    def matches(self, path: Path) -> bool:
        return bool(self.regex.match(path.stem))

    def parse(self, path: Path) -> Video:
        match = self.regex.match(path.stem)
        groups = match.groupdict()

        scenario = Scenario(groups["scenario"].lower())
        datetime_obj = datetime.strptime(f"{groups['date']} {groups['time']}", "%Y-%m-%d %H-%M-%S")

        return Video(
            path=path,
            scenario=scenario,
            recording_date=datetime_obj.date(),
        )


class EnrollmentVideoParser(VideoParser):
    """Parser for enrollment videos: {name}_enrollment_{scenario}_{variant}_{date}_{time}.mp4"""

    regex = re.compile(
        r"^(?P<name>[^_]+)_enrollment_(?P<scenario>[^_]+)_(?P<head_rotation>[^_]+)_(?P<date>\d{4}-\d{2}-\d{2})_(?P<time>\d{2}-\d{2}-\d{2})$"
    )

    def matches(self, path: Path) -> bool:
        return bool(self.regex.match(path.stem))

    def parse(self, path: Path) -> EnrollmentVideo:
        match = self.regex.match(path.stem)
        groups = match.groupdict()

        scenario = Scenario(groups["scenario"].lower())
        head_rotation = HeadRotation(groups["head_rotation"].lower())
        datetime_obj = datetime.strptime(f"{groups['date']} {groups['time']}", "%Y-%m-%d %H-%M-%S")

        return EnrollmentVideo(
            path=path,
            scenario=scenario,
            head_rotation=head_rotation,
            recording_date=datetime_obj.date(),
        )

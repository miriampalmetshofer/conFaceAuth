"""Parser for enrollment videos."""
import re
from datetime import datetime
from pathlib import Path

from face_auth.config.models import Participant
from face_auth.processing.models import EnrollmentVideo, HeadRotation, Scenario
from face_auth.processing.video_parser.parser import VideoParser


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
            participant=Participant(name=groups["name"]),
        )

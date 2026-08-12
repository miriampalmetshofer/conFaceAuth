"""Parser for controlled study videos."""
import re
from datetime import datetime
from pathlib import Path

from face_auth.domain import Participant, Scenario
from face_auth.domain import Video
from face_auth.video_parser.models import ControlledStudyVideo
from face_auth.video_parser.parser import VideoParser


class ControlledStudyParser(VideoParser):
    """Parser for regular videos: {name}_{scenario}_{date}_{time}.mp4"""

    regex = re.compile(
        r"^(?P<name>[^_]+)_(?P<scenario>[^_]+)_(?P<date>\d{4}-\d{2}-\d{2})_(?P<time>\d{2}-\d{2}-\d{2})$"
    )

    def can_parse(self, path: Path) -> bool:
        return bool(self.regex.match(path.stem))

    def parse(self, path: Path) -> Video:
        match = self.regex.match(path.stem)
        groups = match.groupdict()

        scenario = Scenario(groups["scenario"].lower())
        datetime_obj = datetime.strptime(f"{groups['date']} {groups['time']}", "%Y-%m-%d %H-%M-%S")

        return ControlledStudyVideo(
            path=path,
            scenario=scenario,
            recording_date=datetime_obj.date(),
            participant=Participant(name=groups["name"]),
        )

"""Parser for in-the-wild study videos."""
import re
from datetime import datetime
from pathlib import Path

from face_auth.config.models import Participant
from face_auth.processing.models import Video
from face_auth.processing.video_parser.parser import VideoParser


class InTheWildStudyParser(VideoParser):
    """Parser for videos from the In the Wild study: {name}_{date}_{time}.mp4"""

    regex = re.compile(
        r"^(?P<name>[^_]+)_(?P<date>\d{4}-\d{2}-\d{2})_(?P<time>\d{2}-\d{2}-\d{2})$"
    )

    def can_parse(self, path: Path) -> bool:
        return bool(self.regex.match(path.stem))

    def parse(self, path: Path) -> Video:
        match = self.regex.match(path.stem)
        groups = match.groupdict()

        datetime_obj = datetime.strptime(f"{groups['date']} {groups['time']}", "%Y-%m-%d %H-%M-%S")

        return Video(
            path=path,
            recording_date=datetime_obj.date(),
            participant=Participant(name=groups["name"]),
        )

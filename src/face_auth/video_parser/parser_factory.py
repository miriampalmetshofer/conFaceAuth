"""Factory for creating the video parser matching the configured pool."""
from face_auth.domain import Pool
from face_auth.video_parser.impl.controlled_study_parser import ControlledStudyParser
from face_auth.video_parser.impl.in_the_wild_study_parser import InTheWildStudyParser
from face_auth.video_parser.parser import VideoParser


def create_parser(pool: Pool) -> VideoParser:
    """Create the video parser for a study pool.

    Args:
        pool: Study pool selecting which parser to build

    Returns:
        Video parser instance

    Raises:
        ValueError: If the pool is not recognized
    """
    if pool == Pool.CONTROLLED_STUDY:
        return ControlledStudyParser()

    if pool == Pool.IN_THE_WILD:
        return InTheWildStudyParser()

    raise ValueError(f"Unknown pool type: {pool}")

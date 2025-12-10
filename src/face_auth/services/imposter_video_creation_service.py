"""Service for creating imposter videos through stitching."""

from pathlib import Path

from face_auth.config.logging_config import get_logger
from face_auth.config.models import StitchConfig
from face_auth.core.processing.models import Video, ControlledStudyVideo, ImposterSamplePair
from face_auth.core.imposter_sample_creation.video_stitcher import VideoStitcher

logger = get_logger(__name__)


class ImposterVideoCreationService:
    """Service for creating imposter videos by stitching genuine and imposter samples."""

    def __init__(self, stitch_config: StitchConfig):
        """Initialize with stitching configuration.

        Args:
            stitch_config: Configuration for video stitching
            stitch_config: Configuration for imposter creation
        """
        self.stitcher = VideoStitcher(stitch_config)
        self.stitch_config = stitch_config

    def create(
            self,
            pair: ImposterSamplePair
    ) -> Video:
        """Create single imposter video by stitching.

        Args:
            pair: Containing genuine and imposter sample

        Returns:
            Created imposter video object
        """
        output_dir = Path(self.stitch_config.temp_output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        output_filename = self._build_output_file_name(pair)
        output_path = output_dir / output_filename

        logger.debug(f"Creating imposter video: {output_filename}")

        self.stitcher.stitch(
            pair.genuine_video.path,
            pair.imposter_video.path,
            output_path
        )

        # Return video with same type as genuine video
        if isinstance(pair.genuine_video, ControlledStudyVideo):
            return ControlledStudyVideo(
                path=output_path,
                recording_date=pair.genuine_video.recording_date,
                participant=pair.genuine_video.participant,
                scenario=pair.genuine_video.scenario
            )
        else:
            return Video(
                path=output_path,
                recording_date=pair.genuine_video.recording_date,
                participant=pair.genuine_video.participant
            )

    def _build_output_file_name(self, pair: ImposterSamplePair) -> str:
        """Build unique output filename using full video names to prevent collisions."""
        genuine_stem = pair.genuine_video.path.stem
        imposter_stem = pair.imposter_video.path.stem
        return f"{genuine_stem}_vs_{imposter_stem}.mp4"

    def cleanup(self) -> None:
        """Clean up temporary directory for specific participant and device.
        """
        output_dir = Path(self.stitch_config.temp_output_path)
        if output_dir.exists():
            import shutil
            shutil.rmtree(output_dir)
            logger.debug(f"Cleaned up temporary directory: {output_dir}")

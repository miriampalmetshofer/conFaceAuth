"""Video processing service for authentication pipeline."""

from pathlib import Path

from face_auth.config.models import AuthenticationConfig
from face_auth.core.authentication.embedder import Embedder
from face_auth.core.detection import FaceDetector, FaceExtractor
from face_auth.core.authentication.continuous_authenticator import ContinuousAuthenticator
from face_auth.core.authentication.frame_authenticator import FrameAuthenticator
from face_auth.core.processing.video_processor import VideoProcessor
from face_auth.core.processing.models import Video
from face_auth.services.models import EnrollmentData, VideoResult
from face_auth.config.logging_config import get_logger

logger = get_logger(__name__)


class VideoProcessingService:
    """Handles video processing through authentication pipeline."""

    def __init__(
        self,
        config: AuthenticationConfig,
        face_detector: FaceDetector,
        face_extractor: FaceExtractor,
        embedder: Embedder
    ):
        """Initialize video processing service.

        Args:
            config: Authentication configuration
            face_detector: Face detector instance
            face_extractor: Face extractor instance
            embedder: Embedding generator instance
        """
        self.config = config
        self.detector = face_detector
        self.extractor = face_extractor
        self.embedder = embedder

    def process_video(
        self,
        video: Video,
        enrollment_data: EnrollmentData,
        skip_frames: int
    ) -> VideoResult:
        """Process single video and return authentication results.

        Args:
            video: Video information
            enrollment_data: Enrollment data with embeddings
            skip_frames: Process every Nth frame

        Returns:
            VideoResult with frame-by-frame authentication results
        """
        # Build continuous authenticator for this video
        authenticator = ContinuousAuthenticator(
            enrollment_embeddings=enrollment_data.embeddings,
            window_size=self.config.window_size,
            threshold=self.config.threshold,
            similarity_percentile=self.config.similarity_percentile,
            alpha=self.config.alpha
        )

        # Build frame authenticator
        frame_authenticator = FrameAuthenticator(
            detector=self.detector,
            extractor=self.extractor,
            embedder=self.embedder,
            authenticator=authenticator,
            no_face_penalty=self.config.no_face_penalty
        )

        # Build video processor
        video_processor = VideoProcessor(
            frame_authenticator=frame_authenticator,
            debug_output_folder=Path("debug/no_face_frames")
        )

        # Process video
        frame_results = video_processor.process_video_frames(
            video_path=video.path,
            skip_frames=skip_frames
        )

        return VideoResult(
            video=video,
            frame_results=frame_results
        )

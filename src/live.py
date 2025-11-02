from face_auth.video_processor import VideoProcessor
from face_auth.continuous_authenticator import ContinuousAuthenticator
from face_auth.config_manager import ConfigManager
from face_auth.embedder import Embedder
from face_auth.enrollment_service import EnrollmentService
from face_auth.face_detector import FaceDetector
from face_auth.frame_processor import FrameProcessor
from face_auth.logging_config import setup_logging

setup_logging()

config = ConfigManager("live_config.json")

face_detector = FaceDetector(
    detector_name=config.get("detector")
)
embedder = Embedder(
    embedder_name=config.get("embedder")
)
enrollment_service = EnrollmentService(
    embedder=embedder,
    face_detector=face_detector
)

enrollment_embeddings = enrollment_service.load_enrollment_embeddings(
    config.get('enrollment').get("enrollment_folder")
)

continuous_authenticator = ContinuousAuthenticator(
    enrollment_embeddings,
    window_size=config.get("window_size"),
    threshold=config.get("threshold"),
    similarity_percentile=config.get("similarity_percentile"),
    alpha=config.get("alpha"),
)

frame_processor = FrameProcessor(
    face_detector=face_detector,
    embedder=embedder,
    continuous_authenticator=continuous_authenticator,
    no_face_penalty=config.get("no_face_penalty")
)

processor = VideoProcessor(
    frame_processor=frame_processor,
    config=config.config
)
processor.process_live_stream(skip_frames=config.get("skip_frames"))

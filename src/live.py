from face_auth.video_processor import VideoProcessor
from face_auth.authenticator import Authenticator
from face_auth.config_manager import ConfigManager
from face_auth.embedder import Embedder
from face_auth.enrollment_service import EnrollmentService
from face_auth.face_detector import FaceDetector

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

authenticator = Authenticator(
    enrollment_embeddings,
    window_size=config.get("window_size"),
    threshold=config.get("threshold"),
    similarity_computation=config.get("similarity_computation"),
    alpha=config.get("alpha"),
)
processor = VideoProcessor(
    face_detector,
    embedder,
    authenticator,
    config=config.config,
)
processor.process_live_stream(skip_frames=config.get("skip_frames"))

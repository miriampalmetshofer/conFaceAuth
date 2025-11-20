from face_auth.processing import VideoProcessor
from face_auth.core import ContinuousAuthenticator, Embedder, FaceDetector, FrameProcessor
from face_auth.io import ConfigManager
from face_auth.enrollment import service as enrollment_service
from face_auth.utils import setup_logging

setup_logging()

config = ConfigManager("live_config.json")

face_detector = FaceDetector(
    detector_name=config.get("detector")
)
embedder = Embedder(
    embedder_name=config.get("embedder")
)

enrollment_embeddings = enrollment_service.load_enrollment_embeddings(
    config.get('enrollment').get("enrollment_folder"),
    embedder,
    face_detector
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

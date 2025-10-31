from face_auth.video_processor import VideoProcessor
from face_auth.authenticator import Authenticator
from face_auth.config_manager import ConfigManager
from face_auth.embedder import Embedder
from face_auth.enrollment_service import EnrollmentService
from face_auth.face_detector import FaceDetector
from face_auth.frame_authenticator import FrameAuthenticator
from face_auth.result_writer import ResultWriter
from face_auth.debug_frame_saver import DebugFrameSaver
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

authenticator = Authenticator(
    enrollment_embeddings,
    window_size=config.get("window_size"),
    threshold=config.get("threshold"),
    similarity_computation=config.get("similarity_computation"),
    alpha=config.get("alpha"),
)

frame_authenticator = FrameAuthenticator(
    face_detector=face_detector,
    embedder=embedder,
    authenticator=authenticator,
    no_face_penalty=config.get("no_face_penalty")
)

result_writer = ResultWriter(config=config.config)
debug_saver = DebugFrameSaver()

processor = VideoProcessor(
    frame_authenticator=frame_authenticator,
    result_writer=result_writer,
    debug_saver=debug_saver
)
processor.process_live_stream(skip_frames=config.get("skip_frames"))

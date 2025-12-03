from face_auth.processing import VideoProcessor
from face_auth.core import (
    ContinuousAuthenticator,
    Embedder,
    FrameProcessor,
    FACENET_INPUT_WIDTH,
    FACENET_INPUT_HEIGHT
)
from face_auth.detection import FaceDetector, FaceExtractor
from face_auth.io import ConfigManager
from face_auth.enrollment.enrollment_loader import EnrollmentLoader
from face_auth.utils import setup_logging

setup_logging()

config = ConfigManager("live_config.json")

face_detector = FaceDetector(
    detector_backend=config.get("detector")
)
face_extractor = FaceExtractor(
    target_width=FACENET_INPUT_WIDTH,
    target_height=FACENET_INPUT_HEIGHT
)
embedder = Embedder(
    model_name=config.get("embedder")
)

enrollment_loader = EnrollmentLoader(embedder, face_detector, face_extractor)
enrollment_embeddings = enrollment_loader.load_embeddings_from_folder(
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
    detector=face_detector,
    extractor=face_extractor,
    embedder=embedder,
    authenticator=continuous_authenticator,
    no_face_penalty=config.get("no_face_penalty")
)

processor = VideoProcessor(
    frame_processor=frame_processor,
    config=config.config,
    debug_output_folder="debug/no_face_frames"
)
processor.process_live_stream(skip_frames=config.get("skip_frames"))

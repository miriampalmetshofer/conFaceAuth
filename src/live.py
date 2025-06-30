from face_auth.authentication_manager import AuthenticationManager
from face_auth.authenticator import Authenticator
from face_auth.config_manager import ConfigManager
from face_auth.embedder import EmbeddingManager
from face_auth.face_detector import FaceDetector

config = ConfigManager("live_config.json")

face_detector = FaceDetector(
    detector_name=config.get("detector")
)
embedding_manager = EmbeddingManager(
    embedder_name=config.get("embedder")
)
embedding_manager.initialize_embeddings_from_enrollment_images(config.get('enrollment').get("enrollment_folder"),
                                                               face_detector=face_detector)

authenticator = Authenticator(
    embedding_manager.embeddings,
    window_size=config.get("window_size"),
    threshold=config.get("threshold"),
    similarity_computation=config.get("similarity_computation"),
    alpha=config.get("alpha"),
)
processor = AuthenticationManager(
    face_detector,
    embedding_manager,
    authenticator,
    config=config.config,
)
processor.process_live_stream(skip_frames=config.get("skip_frames"))

from face_auth import (
    Authenticator,
    EnrollmentManager,
    EmbeddingManager,
    FaceDetector,
    ConfigManager,
    VideoProcessor
)

config = ConfigManager("config.json")

enrollment = EnrollmentManager(
    enrollment_folder=config.get("enrollment_folder")
)

face_detector = FaceDetector(
    detector_name=config.get("detector")
)
embedding_manager = EmbeddingManager(
    embedder_name=config.get("embedder")
)

authenticator = Authenticator(
    enrollment.embeddings,
    window_size=config.get("window_size"),
    threshold=config.get("threshold"),
    similarity_computation=config.get("similarity_computation"),
)
processor = VideoProcessor(face_detector, embedding_manager, authenticator, threshold=0.6)
processor.process_live_stream(skip_frames=30)  # Adjust skip_frames for speed/performance

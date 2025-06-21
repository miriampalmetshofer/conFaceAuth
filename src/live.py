from face_auth import (
    Authenticator,
    EmbeddingManager,
    FaceDetector,
    ConfigManager,
    VideoProcessor
)

config = ConfigManager("config.json")

face_detector = FaceDetector(
    detector_name=config.get("detector")
)
embedding_manager = EmbeddingManager(
    embedder_name=config.get("embedder")
)
embedding_manager.initialize_embeddings_from_enrollment_images('../data/runs/run1/enrollments/miriam_desktop',
                                                               face_detector=face_detector)

authenticator = Authenticator(
    embedding_manager.embeddings,
    window_size=config.get("window_size"),
    threshold=config.get("threshold"),
    similarity_computation=config.get("similarity_computation"),
)
processor = VideoProcessor(
    face_detector,
    embedding_manager,
    authenticator,
    config=config.config,
)
processor.process_live_stream(skip_frames=30)  # Adjust skip_frames for speed/performance

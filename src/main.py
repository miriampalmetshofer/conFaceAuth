from face_auth import (
    Authenticator,
    EnrollmentManager,
    EmbeddingManager,
    FaceDetector,
    VideoProcessor,
    ConfigManager
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

processor = VideoProcessor(
    face_detector=face_detector,
    embedding_manager=embedding_manager,
    authenticator=authenticator,
    threshold=config.get("threshold"),
    config=config.config
)

processor.process_video(
    video_path=config.get("video_path"),
    annotations_csv_path=config.get("annotations_csv_path"),
    output_path=config.get("output_path"),
    skip_frames=config.get("skip_frames"),
    results_csv_path=config.get("results_csv_path")
)

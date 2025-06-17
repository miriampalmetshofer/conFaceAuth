from face_auth import (
    Authenticator,
    EnrollmentManager,
    EmbeddingManager,
    FaceDetector,
    VideoProcessor,
    ConfigManager
)

config = ConfigManager("config.json")

###
#
# Enrollment
#
###
print("INITIALIZING ENROLLMENT-MANAGER")
enrollment_manager = EnrollmentManager(
    enrollment_video=config.get("enrollment").get("enrollment_video_path"),
    enrollment_folder=config.get("enrollment").get("enrollment_folder")
)
print("STARTING enrollment")
enrollment_manager.enroll(
    frames_per_direction=config.get("enrollment").get("enrollment_frames_per_direction")
)

###
#
# Authentication Setup
#
###
print("INITIALIZING FACE-DETECTOR")
face_detector = FaceDetector(
    detector_name=config.get("detector")
)
print("INITIALIZING EMBEDDING-MANAGER")
embedding_manager = EmbeddingManager(
    embedder_name=config.get("embedder")
)
print("INITIALIZING EMBEDDINGS FROM ENROLLMENT IMAGES")
embedding_manager.initialize_embeddings_from_enrollment_images(face_detector=face_detector,
                                                              enrollment_folder=config.get("enrollment").get(
                                                                  "enrollment_folder"))
print("INITIALIZING AUTHENTICATOR")
authenticator = Authenticator(
    enrollment_embeddings=embedding_manager.embeddings,
    window_size=config.get("window_size"),
    threshold=config.get("threshold"),
    similarity_computation=config.get("similarity_computation"),
)

###
#
# Video Processing
#
###
print("INITIALIZING VIDEO-PROCESSOR")
processor = VideoProcessor(
    face_detector=face_detector,
    embedding_manager=embedding_manager,
    authenticator=authenticator,
    config=config.config
)
print("PROCESSING VIDEO")
processor.process_video(
    video_path=config.get("video_path"),
    annotations_csv_path=config.get("annotations_csv_path"),
    skip_frames=config.get("skip_frames"),
    results_csv_path=config.get("results_csv_path")
)

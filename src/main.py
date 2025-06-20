import os
from face_auth import (
    Authenticator,
    EnrollmentManager,
    EmbeddingManager,
    FaceDetector,
    VideoProcessor,
    ConfigManager
)

# Load and resolve config
config = ConfigManager("bulk_config.json")
base_path = config.get("base_path")
video_folder = config.get("video_folder").format(base_path=base_path)
annotations_csv_path = config.get("annotations_file").format(base_path=base_path)
results_csv_path = config.get("results_file").format(base_path=base_path)

participants = config.get("participants")
devices = ["desktop", "mobile"]

for device in devices:
    for participant in participants:
        sessions = participant["sessions"]
        name = participant["name"]
        for session in sessions:
            video_filename = f"{name}_{session}_{device}.mp4"
            video_path = os.path.join(video_folder, device, video_filename)

            enrollment_video_path = os.path.join(video_folder, device, f"{name}_enrollment_{device}.mp4")
            enrollment_folder = os.path.join(base_path, "enrollments", f"{name}_{device}")

            if os.path.exists(enrollment_folder) and any(f.endswith(".jpg") for f in os.listdir(enrollment_folder)):
                print(f"SKIPPING ENROLLMENT {name} ({device}) — already exists.")
            else:
                print(f"\n=== ENROLLING: {name} ({device}) ===")
                enrollment_manager = EnrollmentManager(
                    enrollment_video=enrollment_video_path,
                    enrollment_folder=enrollment_folder
                )
                enrollment_manager.enroll(
                    frames_per_direction=config.get("enrollment_frames_per_direction")
                )

            print("FACE-DETECTOR")
            face_detector = FaceDetector(detector_name=config.get("detector"))

            print("EMBEDDING-MANAGER")
            embedding_manager = EmbeddingManager(embedder_name=config.get("embedder"))

            print("EMBEDDINGS FROM ENROLLMENT IMAGES")
            embedding_manager.initialize_embeddings_from_enrollment_images(
                face_detector=face_detector,
                enrollment_folder=enrollment_folder
            )

            print("AUTHENTICATOR")
            authenticator = Authenticator(
                enrollment_embeddings=embedding_manager.embeddings,
                window_size=config.get("window_size"),
                threshold=config.get("threshold"),
                similarity_computation=config.get("similarity_computation"),
            )

            print("VIDEO PROCESSOR")
            processor = VideoProcessor(
                face_detector=face_detector,
                embedding_manager=embedding_manager,
                authenticator=authenticator,
                config=config.config
            )

            print(f"\n--- PROCESSING: {video_filename} ---")
            processor.process_video(
                video_path=video_path,
                annotations_csv_path=annotations_csv_path,
                skip_frames=config.get("skip_frames"),
                results_csv_path=results_csv_path
            )

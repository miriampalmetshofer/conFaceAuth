import os
import sys
import glob
from face_auth.video_processor import VideoProcessor
from face_auth.authenticator import Authenticator
from face_auth.config_manager import ConfigManager
from face_auth.embedder import Embedder
from face_auth.enrollment_service import EnrollmentService
from face_auth.enrollment_manager import EnrollmentManager
from face_auth.face_detector import FaceDetector
from face_auth.frame_authenticator import FrameAuthenticator
from face_auth.result_writer import ResultWriter
from face_auth.debug_frame_saver import DebugFrameSaver


def main(config_file):
    config = ConfigManager(config_file)
    base_path = config.get("base_path")
    enrollment_base_path = config.get("enrollment_base_path")
    video_folder = config.get("video_folder").format(base_path=base_path)
    results_csv_path = config.get("results_file").format(base_path=base_path)

    participants = config.get("participants")
    devices = config.get("devices", ["mobile", "desktop"])
    pool_name = config.get("pool", "unknown")

    print(f"\n{'='*60}")
    print(f"Processing Pool: {pool_name.upper()}")
    print(f"Base Path: {base_path}")
    print(f"Enrollment Path: {enrollment_base_path}")
    print(f"Devices: {', '.join(devices)}")
    print(f"{'='*60}\n")

    # Check if results file already exists
    if os.path.exists(results_csv_path):
        print(f"Results file already exists at: {results_csv_path}")
        confirm = input("Do you want to delete this file and continue? (y/N): ").strip().lower()
        if confirm == 'y':
            os.remove(results_csv_path)
            print("File deleted.")
        else:
            print("File NOT deleted. Stopping execution.")
            return

    # Process videos for each participant
    for device in devices:
        for participant in participants:
            name = participant["name"]

            # Discover videos for this participant and device
            # - Controlled study: {name}_{mode}_{timestamp}.mp4
            # - In the wild: {name}_{timestamp}.mp4 (no mode)
            device_folder = os.path.join(video_folder, device)

            video_pattern = os.path.join(device_folder, f"{name}_*.mp4")
            video_pattern_upper = os.path.join(device_folder, f"{name}_*.MP4")
            video_files = glob.glob(video_pattern) + glob.glob(video_pattern_upper)

            if not video_files:
                print(f"INFO: No videos found for {name} on {device}")
                continue

            print(f"\n{'='*60}")
            print(f"Participant: {name} | Device: {device}")
            print(f"Found {len(video_files)} video(s)")
            print(f"{'='*60}\n")

            participant_enrollment_folder = os.path.join(enrollment_base_path, device, name)

            if not os.path.exists(participant_enrollment_folder):
                raise FileNotFoundError(
                    f"\n{'!'*60}\n"
                    f"ERROR: Enrollment folder not found!\n"
                    f"Path: {participant_enrollment_folder}\n"
                    f"Participant: '{name}' | Device: '{device}'\n"
                    f"{'!'*60}\n"
                )

            enrollment_video_pattern = os.path.join(participant_enrollment_folder, f"{name}_enrollment_*.mp4")
            enrollment_video_pattern_upper = os.path.join(participant_enrollment_folder, f"{name}_enrollment_*.MP4")
            enrollment_videos = glob.glob(enrollment_video_pattern) + glob.glob(enrollment_video_pattern_upper)

            if not enrollment_videos:
                raise FileNotFoundError(
                    f"\n{'!'*60}\n"
                    f"ERROR: No enrollment video found!\n"
                    f"Searched in: {participant_enrollment_folder}\n"
                    f"Expected pattern: {name}_enrollment_*.mp4 or .MP4\n"
                    f"Participant: '{name}' | Device: '{device}'\n"
                    f"{'!'*60}\n"
                )

            # Use the first enrollment video found
            enrollment_video_path = enrollment_videos[0]
            enrollment_video_name = os.path.splitext(os.path.basename(enrollment_video_path))[0]
            enrollment_folder = os.path.join(participant_enrollment_folder, enrollment_video_name)

            # Enrollment phase
            if os.path.exists(enrollment_folder) and any(f.endswith(".jpg") for f in os.listdir(enrollment_folder)):
                print(f"SKIPPING ENROLLMENT {name} ({device}) — already exists.")
            else:
                print(f"\n=== ENROLLING: {name} ({device}) ===")
                print(f"Using enrollment video: {os.path.basename(enrollment_video_path)}")
                enrollment_manager = EnrollmentManager(
                    enrollment_video=enrollment_video_path,
                    enrollment_folder=enrollment_folder
                )
                enrollment_manager.enroll(
                    frames_per_direction=config.get("enrollment_frames_per_direction")
                )

            # Initialize components (once per participant/device)
            print("\nInitializing components...")
            face_detector = FaceDetector(detector_name=config.get("detector"))
            embedder = Embedder(embedder_name=config.get("embedder"))
            enrollment_service = EnrollmentService(
                embedder=embedder,
                face_detector=face_detector
            )

            enrollment_embeddings = enrollment_service.load_enrollment_embeddings(enrollment_folder)

            authenticator = Authenticator(
                enrollment_embeddings=enrollment_embeddings,
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

            for video_path in video_files:
                video_filename = os.path.basename(video_path)
                print(f"\n--- PROCESSING: {video_filename} ---")
                processor.process_video(
                    video_path=video_path,
                    skip_frames=config.get("skip_frames"),
                    results_csv_path=results_csv_path
                )

    print(f"\n{'='*60}")
    print(f"Processing complete! Results saved to: {results_csv_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    # Allow config file to be passed as command line argument
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    else:
        print("No config file specified. Available configs:")
        print("  1. controlled_study_config.json")
        print("  2. in_the_wild_config.json")
        choice = input("\nSelect config (1 or 2): ").strip()

        if choice == "1":
            config_file = "controlled_study_config.json"
        elif choice == "2":
            config_file = "in_the_wild_config.json"
        else:
            print("Invalid choice. Using controlled_study_config.json")
            config_file = "controlled_study_config.json"

    main(config_file)

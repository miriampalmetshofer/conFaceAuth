import os
import glob
from dataclasses import dataclass

from face_auth.video_processor import VideoProcessor
from face_auth.continuous_authenticator import ContinuousAuthenticator
from face_auth.config_manager import ConfigManager
from face_auth.embedder import Embedder
from face_auth.face_detector import FaceDetector
from face_auth.frame_processor import FrameProcessor
from face_auth import enrollment_service


@dataclass
class ParticipantInfo:
    """Represents participant identification information."""
    name: str
    device: str


def discover_videos(base_path: str, participant: ParticipantInfo) -> list[str]:
    """Discover video files for a participant and device."""
    device_folder = os.path.join(base_path, participant.device)
    video_pattern = os.path.join(device_folder, f"{participant.name}_*4")

    return glob.glob(video_pattern)


def find_enrollment_video(enrollment_base_path: str, participant: ParticipantInfo) -> str:
    """Find and validate enrollment video for participant. Returns enrollment video path. Returns the first found video."""
    participant_enrollment_folder = os.path.join(enrollment_base_path, participant.device, participant.name)

    if not os.path.exists(participant_enrollment_folder):
        raise FileNotFoundError(
            f"\n{'!' * 60}\n"
            f"ERROR: Enrollment folder not found!\n"
            f"Path: {participant_enrollment_folder}\n"
            f"Participant: '{participant.name}' | Device: '{participant.device}'\n"
            f"{'!' * 60}\n"
        )

    enrollment_video_pattern = os.path.join(participant_enrollment_folder, f"{participant.name}_enrollment_*4")
    enrollment_videos = glob.glob(enrollment_video_pattern)

    if not enrollment_videos:
        raise FileNotFoundError(
            f"\n{'!' * 60}\n"
            f"ERROR: No enrollment video found!\n"
            f"Searched in: {participant_enrollment_folder}\n"
            f"Expected pattern: {participant.name}_enrollment_*4\n"
            f"Participant: '{participant.name}' | Device: '{participant.device}'\n"
            f"{'!' * 60}\n"
        )

    return enrollment_videos[0]


def get_enrollment_folder_name(enrollment_video_path: str, enrollment_base_path: str,
                               participant: ParticipantInfo) -> str:
    """Derive enrollment folder path from video path."""
    participant_enrollment_folder = os.path.join(enrollment_base_path, participant.device, participant.name)
    enrollment_video_name = os.path.splitext(os.path.basename(enrollment_video_path))[0]

    return os.path.join(participant_enrollment_folder, enrollment_video_name)


def setup_enrollment(enrollment_base_path: str,
                     participant: ParticipantInfo,
                     frames_per_direction: int,
                     logger) -> str:
    """Setup enrollment for participant. Returns enrollment folder path."""
    enrollment_video_path = find_enrollment_video(enrollment_base_path, participant)
    enrollment_folder = get_enrollment_folder_name(enrollment_video_path, enrollment_base_path, participant)

    enrollment_folder_is_already_filled = os.path.exists(enrollment_folder) and any(
        f.endswith(".jpg") for f in os.listdir(enrollment_folder))
    if enrollment_folder_is_already_filled:
        logger.info(f"Skipping enrollment for {participant.name} ({participant.device}) — already exists")
    else:
        logger.info(f"=== ENROLLING: {participant.name} ({participant.device}) ===")
        logger.info(f"Using enrollment video: {os.path.basename(enrollment_video_path)}")
        enrollment_frames = enrollment_service.get_enrollment_frames_for_video(enrollment_video_path, frames_per_direction)
        enrollment_service.save_enrollment_frames_to_folder(enrollment_frames, enrollment_folder)

    return enrollment_folder


def process_participant(participant: ParticipantInfo, base_path: str,
                        enrollment_base_path: str, results_csv_path: str,
                        config: ConfigManager, logger):
    """Process all videos for a single participant on a device."""
    video_files = discover_videos(base_path, participant)
    if not video_files:
        logger.info(f"No videos found for {participant.name} on {participant.device}")
        return

    logger.info(f"{'=' * 60}")
    logger.info(f"Participant: {participant.name} | Device: {participant.device}")
    logger.info(f"Found {len(video_files)} video(s)")
    logger.info(f"{'=' * 60}")

    enrollment_folder = setup_enrollment(
        enrollment_base_path,
        participant,
        config.get("enrollment_frames_per_direction"),
        logger
    )

    logger.info("Initializing shared components...")
    face_detector = FaceDetector(detector_name=config.get("detector"))
    embedder = Embedder(embedder_name=config.get("embedder"))
    enrollment_embeddings = enrollment_service.load_enrollment_embeddings(enrollment_folder, embedder, face_detector)

    for video_path in video_files:
        video_filename = os.path.basename(video_path)
        logger.info(f"--- PROCESSING: {video_filename} ---")

        continuous_authenticator = ContinuousAuthenticator(
            enrollment_embeddings=enrollment_embeddings,
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

        processor.process_video(
            video_path=video_path,
            skip_frames=config.get("skip_frames"),
            results_csv_path=results_csv_path,
            participant=participant
        )

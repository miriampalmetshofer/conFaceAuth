import os

from face_auth.config.models import ParticipantConfig
from face_auth.processing.video_discovery import VideoDiscovery
from face_auth.processing.video_processor import VideoProcessor
from face_auth.processing.video_parser import UsageVideoParser, EnrollmentVideoParser
from face_auth.core.authenticator import ContinuousAuthenticator
from face_auth.core.embedder import Embedder
from face_auth.detection import FaceDetector, FaceExtractor
from face_auth.core.constants import FACENET_INPUT_WIDTH, FACENET_INPUT_HEIGHT
from face_auth.io.config_manager import ConfigManager
from face_auth.core.frame_processor import FrameProcessor
from face_auth.enrollment import (
    EnrollmentOrchestrator,
    VideoFrameExtractor,
    HeadPoseEstimator,
    DirectionClassifier,
    NormalDistributionSampler,
    EnrollmentFrameSaver,
    FRAME_SAMPLING_INTERVAL,
    YAW_THRESHOLD,
    PITCH_THRESHOLD,
    DISTRIBUTION_MEAN_FRACTION,
    DISTRIBUTION_STDDEV_FRACTION,
    SAMPLING_SEED,
)
from face_auth.io import EnrollmentLoader




def setup_enrollment(participant: ParticipantConfig,
                     enrollment_base_path: str,
                     frames_per_direction: int,
                     logger) -> str:
    """Setup enrollment for participant. Returns enrollment folder path."""
    participant_enrollment_folder = os.path.join(enrollment_base_path, participant.device, participant.name)

    enrollment_discovery = VideoDiscovery(participant, EnrollmentVideoParser())
    enrollment_videos = enrollment_discovery.discover(participant_enrollment_folder)

    if not enrollment_videos:
        raise FileNotFoundError(
            f"\n{'!' * 60}\n"
            f"ERROR: No enrollment video found!\n"
            f"Searched in: {participant_enrollment_folder}\n"
            f"Expected pattern: {participant.name}_enrollment_*\n"
            f"Participant: '{participant.name}' | Device: '{participant.device}'\n"
            f"{'!' * 60}\n"
        )

    enrollment_video = enrollment_videos[0]
    participant_folder = os.path.join(enrollment_base_path, participant.device, participant.name)
    video_name = os.path.splitext(os.path.basename(enrollment_video.path))[0]
    enrollment_folder = os.path.join(participant_folder, video_name)

    enrollment_folder_is_already_filled = os.path.exists(enrollment_folder) and any(
        f.endswith(".jpg") for f in os.listdir(enrollment_folder))
    if enrollment_folder_is_already_filled:
        logger.info(f"Skipping enrollment for {participant.name} ({participant.device}) — already exists")
    else:
        logger.info(f"=== ENROLLING: {participant.name} ({participant.device}) ===")
        logger.info(f"Using enrollment video: {enrollment_video.filename}")

        # Create enrollment orchestrator with all components
        frame_extractor = VideoFrameExtractor(FRAME_SAMPLING_INTERVAL)
        pose_estimator = HeadPoseEstimator()
        direction_classifier = DirectionClassifier(YAW_THRESHOLD, PITCH_THRESHOLD)
        frame_sampler = NormalDistributionSampler(
            DISTRIBUTION_MEAN_FRACTION,
            DISTRIBUTION_STDDEV_FRACTION,
            SAMPLING_SEED
        )
        frame_saver = EnrollmentFrameSaver()

        orchestrator = EnrollmentOrchestrator(
            frame_extractor,
            pose_estimator,
            direction_classifier,
            frame_sampler,
            frame_saver
        )

        orchestrator.process_enrollment_video(
            enrollment_video.path,
            frames_per_direction,
            enrollment_folder
        )

    return enrollment_folder


def process_participant(participant: ParticipantConfig, base_path: str,
                        enrollment_base_path: str, results_csv_path: str,
                        config: ConfigManager, logger):
    """Process all videos for a single participant on a device."""
    participant_video_folder = os.path.join(base_path, participant.device)

    video_discovery = VideoDiscovery(participant, UsageVideoParser())
    videos = video_discovery.discover(participant_video_folder)

    if not videos:
        logger.warn(f"No videos found for {participant.name} on {participant.device}")
        return

    logger.info(f"{'=' * 60}")
    logger.info(f"Participant: {participant.name} | Device: {participant.device}")
    logger.info(f"Found {len(videos)} video(s)")
    logger.info(f"{'=' * 60}")

    enrollment_folder = setup_enrollment(
        participant,
        enrollment_base_path,
        config.get("enrollment_frames_per_direction"),
        logger
    )

    logger.info("Initializing shared components...")
    face_detector = FaceDetector(detector_backend=config.get("detector"))
    face_extractor = FaceExtractor(
        target_width=FACENET_INPUT_WIDTH,
        target_height=FACENET_INPUT_HEIGHT
    )
    embedder = Embedder(model_name=config.get("embedder"))

    enrollment_loader = EnrollmentLoader(embedder, face_detector, face_extractor)
    enrollment_embeddings = enrollment_loader.load_embeddings(enrollment_folder)

    for video in videos:
        logger.info(f"--- PROCESSING: {video.filename} ---")
        logger.info(f"    Scenario: {video.scenario.value} | Date: {video.recording_date}")

        continuous_authenticator = ContinuousAuthenticator(
            enrollment_embeddings=enrollment_embeddings,
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

        processor.process_video(
            video_path=video.path,
            skip_frames=config.get("skip_frames"),
            results_csv_path=results_csv_path,
            participant=participant
        )

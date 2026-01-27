#!/usr/bin/env python3
"""Live face authentication demo using webcam."""

import sys
from pathlib import Path

# Add src directory to Python path
project_root = Path(__file__).parent
src_path = project_root / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))

import json
import cv2
import logging
import argparse

from face_auth.authentication.embedder.embedder import Embedder
from face_auth.authentication.core.frame_authenticator import FrameAuthenticator
from face_auth.authentication.core.backend.authenticator_factory import (
    create_authenticator,
    AuthenticatorBackendType
)
from face_auth.authentication.core.backend.impl.temporal_decay.models import TemporalDecayConfig


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LiveFaceAuth:
    """Live face authentication using webcam."""

    def __init__(self, config_path: str = "live_config.json"):
        """Initialize live face authentication.

        Args:
            config_path: Path to configuration file
        """
        logger.info("Loading configuration...")
        self.config = self._load_config(config_path)

        logger.info("Initializing InsightFace embedder...")
        self.embedder = Embedder(
            model_name=self.config["embedder"]["model"],
            model_config=self.config["embedder"]["config"]
        )

        # Initialize InsightFace app directly for bbox extraction
        from insightface.app import FaceAnalysis
        embedder_config = self.config["embedder"]["config"]
        self.face_app = FaceAnalysis(
            name=embedder_config.get("model_name", "buffalo_sc"),
            providers=['CPUExecutionProvider']
        )
        det_size = tuple(embedder_config.get("det_size", [640, 640]))
        self.face_app.prepare(ctx_id=-1, det_size=det_size)

        self.enrollment_embeddings = None
        self.authenticator = None

    def _load_config(self, config_path: str) -> dict:
        """Load configuration from JSON file."""
        with open(config_path, 'r') as f:
            return json.load(f)

    def process_enrollment_video(self):
        """Process enrollment video to extract embeddings."""
        enrollment_video = self.config["enrollment_video"]
        logger.info(f"Processing enrollment video: {enrollment_video}")

        if not Path(enrollment_video).exists():
            raise FileNotFoundError(
                f"Enrollment video not found: {enrollment_video}\n"
                f"Please update the 'enrollment_video' path in live_config.json"
            )

        frame_interval = self.config["enrollment"].get("frame_sampling_interval", 30)
        embeddings = []

        cap = cv2.VideoCapture(enrollment_video)
        if not cap.isOpened():
            raise IOError(f"Could not open enrollment video: {enrollment_video}")

        frame_count = 0
        processed_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Sample frames at intervals
            if frame_count % frame_interval != 0:
                continue

            # Convert to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Get embedding
            embedding_result = self.embedder.get_embedding(frame_rgb)

            if embedding_result.face_detected:
                embeddings.append(embedding_result.embedding)
                processed_count += 1
                logger.info(f"Enrollment frame {processed_count}: face detected")

        cap.release()

        if len(embeddings) == 0:
            raise ValueError(
                f"No faces detected in enrollment video!\n"
                f"Please ensure the enrollment video contains clear face frames."
            )

        logger.info(f"Enrollment complete: {len(embeddings)} embeddings extracted from {frame_count} frames")
        self.enrollment_embeddings = embeddings

        # Initialize authenticator with enrollment embeddings
        temporal_decay_config = TemporalDecayConfig(
            threshold=self.config["temporal_decay"]["threshold"],
            similarity_percentile=self.config["temporal_decay"]["similarity_percentile"],
            k_weight=self.config["temporal_decay"]["k_weight"],
            k_decay=self.config["temporal_decay"]["k_decay"],
            initial_confidence=self.config["temporal_decay"]["initial_confidence"]
        )

        authenticator_backend = create_authenticator(
            AuthenticatorBackendType.TEMPORAL_DECAY,
            temporal_decay_config,
            self.enrollment_embeddings
        )

        fps = self.config.get("fps")
        self.authenticator = FrameAuthenticator(self.embedder, authenticator_backend, fps, use_wall_clock_time=True)
        logger.info("Authenticator initialized")

    def run_live_authentication(self):
        """Run live authentication using webcam."""
        if self.authenticator is None:
            raise RuntimeError("Must process enrollment video first!")

        camera_index = self.config.get("camera_index")
        logger.info(f"Attempting to open camera {camera_index}...")

        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            error_msg = (
                f"Could not open camera {camera_index}!\n"
                f"Update 'camera_index' in live_config.json to use a different camera."
            )
            raise IOError(error_msg)

        # Set camera resolution (optional)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        threshold = self.config["temporal_decay"]["threshold"]

        logger.info("Starting live authentication. Press 'q' to quit.")

        frame_index = 1
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.error("Failed to read frame from camera")
                    break

                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)

                # Authenticate frame
                auth_result = self.authenticator.authenticate(frame, frame_index)
                frame_index += 1

                # Get bounding box using InsightFace directly
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                faces = self.face_app.get(frame_rgb)

                # Draw overlays
                display_frame = self._draw_overlays(
                    frame,
                    auth_result,
                    faces,
                    threshold
                )

                # Show frame
                cv2.imshow('Live Face Authentication', display_frame)

                # Check for quit key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("Quitting...")
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()

    def _draw_overlays(self, frame, auth_result, faces, threshold):
        """Draw authentication status and bounding boxes on frame."""
        display_frame = frame.copy()
        h, w = display_frame.shape[:2]

        # Determine colors based on authentication status
        is_unlocked = auth_result.status.value == "Unlocked"
        status_color = (0, 255, 0) if is_unlocked else (0, 0, 255)  # Green/Red

        # Draw bounding box if face detected
        if faces and len(faces) > 0:
            face = faces[0]  # Take first face
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]

            # Draw rectangle
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), status_color, 2)

            # Draw face detection confidence
            det_score = face.det_score
            cv2.putText(
                display_frame,
                f"Det: {det_score:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                status_color,
                2
            )

        # Draw status panel (top-left)
        panel_height = 120
        panel_width = 300
        overlay = display_frame.copy()
        cv2.rectangle(overlay, (0, 0), (panel_width, panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, display_frame, 0.5, 0, display_frame)

        # Status text
        status_text = "UNLOCKED" if is_unlocked else "LOCKED"
        cv2.putText(
            display_frame,
            status_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            status_color,
            2
        )

        # Trust score
        trust_score = auth_result.trust
        cv2.putText(
            display_frame,
            f"Trust: {trust_score:.4f}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        # Threshold
        cv2.putText(
            display_frame,
            f"Threshold: {threshold:.4f}",
            (10, 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        # Similarity (if available)
        if auth_result.similarity is not None:
            cv2.putText(
                display_frame,
                f"Similarity: {auth_result.similarity:.4f}",
                (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

        # Draw trust bar (bottom)
        bar_height = 30
        bar_y = h - bar_height - 10
        bar_width = w - 20
        bar_x = 10

        # Background bar
        cv2.rectangle(
            display_frame,
            (bar_x, bar_y),
            (bar_x + bar_width, bar_y + bar_height),
            (50, 50, 50),
            -1
        )

        # Trust level bar
        trust_normalized = max(0, min(1, trust_score))  # Clamp to [0, 1]
        trust_bar_width = int(bar_width * trust_normalized)
        cv2.rectangle(
            display_frame,
            (bar_x, bar_y),
            (bar_x + trust_bar_width, bar_y + bar_height),
            status_color,
            -1
        )

        # Threshold line
        threshold_x = bar_x + int(bar_width * threshold)
        cv2.line(
            display_frame,
            (threshold_x, bar_y),
            (threshold_x, bar_y + bar_height),
            (255, 255, 0),  # Yellow
            3
        )

        # Bar labels
        cv2.putText(
            display_frame,
            "0.0",
            (bar_x, bar_y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1
        )
        cv2.putText(
            display_frame,
            "1.0",
            (bar_x + bar_width - 20, bar_y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1
        )

        # Instructions
        cv2.putText(
            display_frame,
            "Press 'q' to quit",
            (w - 200, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1
        )

        return display_frame


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Live face authentication demo")

    parser.add_argument(
        "--config",
        default="live_config.json",
        help="Path to configuration file (default: live_config.json)"
    )

    args = parser.parse_args()

    try:
        # Initialize
        live_auth = LiveFaceAuth(args.config)

        # Process enrollment
        live_auth.process_enrollment_video()

        # Run live authentication
        live_auth.run_live_authentication()

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()

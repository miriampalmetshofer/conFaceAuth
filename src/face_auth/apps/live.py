#!/usr/bin/env python3
"""Live face authentication demo using a webcam.

Thin driver over the shared face-auth components. It loads a typed config, enrolls
by sampling frames evenly across an enrollment video,
and authenticates every camera frame through the same FrameAuthenticator the batch
pipeline uses. The face box drawn on screen comes from the authentication result, so
the demo never runs face detection a second time.
"""

import argparse
import json
import time
from importlib.resources import files
from pathlib import Path

import cv2
from pydantic import BaseModel, Field, ConfigDict

from face_auth.configs.models import EmbedderModelConfig
from face_auth.core.authenticator.backend.authenticator_factory import create_authenticator
from face_auth.core.authenticator.backend.impl.temporal_decay.models import TemporalDecayConfig
from face_auth.core.authenticator.models import AuthenticationResult
from face_auth.core.embedder.backend.embedder_factory import create_embedder
from face_auth.core.models import Frame
from face_auth.authentication.frame_authenticator import FrameAuthenticator
from face_auth.enrollment.service import EnrollmentService
from face_auth.logging_config import setup_logging, get_logger

logger = get_logger(__name__)


class LiveConfig(BaseModel):
    """Live demo configuration parsed from JSON.

    Reuses the shared embedder and temporal-decay schemas so the demo runs the same
    models as the batch study, and adds only the fields specific to a webcam.
    """

    enrollment_video: Path = Field(..., description="Video the enrolled user is sampled from")
    enrollment_frames: int = Field(..., gt=0, description="Frames sampled evenly across the video")
    camera_index: int = Field(..., ge=0, description="Index of the webcam OpenCV opens")
    embedder: EmbedderModelConfig
    authenticator: TemporalDecayConfig

    model_config = ConfigDict(frozen=True)


def load_live_config(config_path: str) -> LiveConfig:
    """Load and validate the live demo configuration from a JSON file.

    Relative paths in the config are resolved against the current working
    directory, so run the demo from the repository root.
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    with open(config_file, "r") as f:
        return LiveConfig.model_validate(json.load(f))


class LiveFaceAuth:
    """Runs webcam capture and renders the authentication overlay."""

    def __init__(self, config: LiveConfig):
        self.config = config
        self.embedder = create_embedder(
            config=config.embedder,
            cache_enabled=False,
            cache_dir=None,
        )
        self.frame_authenticator = None
        self.start_time = None

    def enroll(self):
        """Enroll the user and build the per-frame authenticator."""
        logger.info("Enrolling from configured enrollment video...")
        enrollment_embeddings = EnrollmentService.enroll_from_video_uniform(
            embedder=self.embedder,
            video_path=self.config.enrollment_video,
            num_frames=self.config.enrollment_frames,
        )
        # create_authenticator reads only the backend config; the live loop
        # authenticates every camera frame, so there is no sampling rate to supply.
        authenticator = create_authenticator(
            backend=self.config.authenticator,
            enrollment_embeddings=enrollment_embeddings,
        )
        self.frame_authenticator = FrameAuthenticator(
            embedder=self.embedder,
            authenticator=authenticator,
        )

    def run(self):
        """Run live authentication until the user quits."""
        cap = self._open_camera()
        threshold = self.config.authenticator.threshold

        logger.info("Starting live authentication. Press 'q' to quit.")
        try:
            while True:
                ret, frame_bgr = cap.read()
                if not ret:
                    logger.error("Failed to read frame from camera")
                    break

                # Mirror the frame so it reads like a mirror to the user.
                frame_bgr = cv2.flip(frame_bgr, 1)
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

                auth_result = self._authenticate(frame_rgb)
                display_frame = self._draw_overlays(frame_bgr, auth_result, threshold)

                cv2.imshow("Live Face Authentication", display_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    logger.info("Quitting...")
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()

    def _authenticate(self, frame_rgb) -> AuthenticationResult:
        """Authenticate one camera frame stamped with elapsed wall-clock time."""
        if self.start_time is None:
            self.start_time = time.time()
        timestamp_ms = (time.time() - self.start_time) * 1000
        return self.frame_authenticator.update_authentication_with_new_evidence(
            Frame(image=frame_rgb, timestamp_ms=timestamp_ms)
        )

    def _open_camera(self) -> cv2.VideoCapture:
        """Open the configured webcam at the requested resolution."""
        logger.info(f"Opening camera {self.config.camera_index}...")
        cap = cv2.VideoCapture(self.config.camera_index)
        if not cap.isOpened():
            raise IOError(
                f"Could not open camera {self.config.camera_index}. "
                f"Update 'camera_index' in the live config to use a different camera."
            )
        return cap

    def _draw_overlays(
        self,
        frame_bgr,
        auth_result: AuthenticationResult,
        threshold: float,
    ):
        """Draw authentication status, the trust bar, and the face box on the frame."""
        display_frame = frame_bgr.copy()
        h, w = display_frame.shape[:2]

        is_unlocked = auth_result.is_authenticated
        status_color = (0, 255, 0) if is_unlocked else (0, 0, 255)  # Green/Red

        self._draw_face_box(display_frame, auth_result.bounding_box, status_color)
        self._draw_status_panel(display_frame, auth_result, threshold, status_color)
        self._draw_trust_bar(display_frame, auth_result.trust, threshold, status_color, h, w)

        cv2.putText(
            display_frame, "Press 'q' to quit", (w - 200, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1
        )
        return display_frame

    def _draw_face_box(self, frame, bounding_box, color):
        """Draw the detected face rectangle, if a face was found."""
        if bounding_box is None:
            return
        x1, y1 = bounding_box.x, bounding_box.y
        x2, y2 = x1 + bounding_box.width, y1 + bounding_box.height
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    def _draw_status_panel(self, frame, auth_result, threshold, status_color):
        """Draw the top-left panel with lock status, trust, threshold, similarity."""
        panel = frame.copy()
        cv2.rectangle(panel, (0, 0), (300, 120), (0, 0, 0), -1)
        cv2.addWeighted(panel, 0.5, frame, 0.5, 0, frame)

        status_text = "UNLOCKED" if auth_result.is_authenticated else "LOCKED"
        cv2.putText(
            frame, status_text, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 2
        )
        cv2.putText(
            frame, f"Trust: {auth_result.trust:.4f}", (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
        )
        cv2.putText(
            frame, f"Threshold: {threshold:.4f}", (10, 85),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
        )
        if auth_result.similarity is not None:
            cv2.putText(
                frame, f"Similarity: {auth_result.similarity:.4f}", (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
            )

    def _draw_trust_bar(self, frame, trust_score, threshold, status_color, h, w):
        """Draw the bottom trust bar with the threshold marker."""
        bar_height = 30
        bar_x = 10
        bar_y = h - bar_height - 10
        bar_width = w - 20

        cv2.rectangle(
            frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height),
            (50, 50, 50), -1
        )
        trust_normalized = max(0, min(1, trust_score))
        trust_bar_width = int(bar_width * trust_normalized)
        cv2.rectangle(
            frame, (bar_x, bar_y), (bar_x + trust_bar_width, bar_y + bar_height),
            status_color, -1
        )
        threshold_x = bar_x + int(bar_width * threshold)
        cv2.line(
            frame, (threshold_x, bar_y), (threshold_x, bar_y + bar_height),
            (255, 255, 0), 3
        )
        cv2.putText(
            frame, "0.0", (bar_x, bar_y - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1
        )
        cv2.putText(
            frame, "1.0", (bar_x + bar_width - 20, bar_y - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1
        )


def main():
    """Entry point for the live authentication demo."""
    parser = argparse.ArgumentParser(description="Live face authentication demo")
    parser.add_argument(
        "--config",
        default=str(files("face_auth.configs") / "live_config.json"),
        help="Path to the live configuration file",
    )
    args = parser.parse_args()

    setup_logging()

    try:
        config = load_live_config(args.config)
        live_auth = LiveFaceAuth(config)
        live_auth.enroll()
        live_auth.run()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()

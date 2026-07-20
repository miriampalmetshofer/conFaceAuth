"""MediaPipe face detector backend (BlazeFace, Tasks API)."""

from typing import Optional
import numpy as np
import mediapipe as mp

from face_auth.core.detection.backend.detector_backend import DetectorBackend
from face_auth.core.detection.models import BoundingBox, DetectionResult


class MediaPipeBackend(DetectorBackend):
    """Face detector using MediaPipe FaceDetector (BlazeFace)."""

    def __init__(self, model_path: str = "src/face_auth/core/detection/backend/impl/blaze_face_short_range.tflite"):
        """Initialize MediaPipe BlazeFace detector."""
        BaseOptions = mp.tasks.BaseOptions
        FaceDetector = mp.tasks.vision.FaceDetector
        FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = FaceDetectorOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.IMAGE,
            min_detection_confidence=0.4,
        )

        self._detector = FaceDetector.create_from_options(options)

    def get_backend_name(self) -> str:
        """Return the detector backend name used in configuration."""
        return "mediapipe"

    def detect(self, image_rgb: np.ndarray) -> Optional[DetectionResult]:
        """Detect and extract the largest face in an RGB image."""
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=image_rgb
        )

        result = self._detector.detect(mp_image)

        if not result.detections:
            return None

        best_bbox: Optional[BoundingBox] = None
        best_area = 0

        for detection in result.detections:
            bbox = detection.bounding_box

            x = bbox.origin_x
            y = bbox.origin_y
            width = bbox.width
            height = bbox.height
            area = width * height

            if area > best_area:
                best_area = area
                best_bbox = BoundingBox(
                    x=max(0, x),
                    y=max(0, y),
                    width=width,
                    height=height,
                )

        if best_bbox is None:
            return None

        face_region = image_rgb[
            best_bbox.y:best_bbox.y + best_bbox.height,
            best_bbox.x:best_bbox.x + best_bbox.width
        ]

        return DetectionResult(face_image=face_region, bounding_box=best_bbox)

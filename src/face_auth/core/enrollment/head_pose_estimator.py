from typing import Optional
import math
import numpy as np
import cv2
from face_auth.core.enrollment.models import HeadPose
import mediapipe as mp

NOSE_TIP_LANDMARK = 1
CHIN_LANDMARK = 152
LEFT_EYE_CORNER_LANDMARK = 33
RIGHT_EYE_CORNER_LANDMARK = 263
LEFT_MOUTH_CORNER_LANDMARK = 61
RIGHT_MOUTH_CORNER_LANDMARK = 291

FACE_MODEL_POINTS = np.array([
    [0.0, 0.0, 0.0],           # Nose tip
    [0.0, -330.0, -65.0],      # Chin
    [-225.0, 170.0, -135.0],   # Left eye corner
    [225.0, 170.0, -135.0],    # Right eye corner
    [-150.0, -150.0, -125.0],  # Left mouth corner
    [150.0, -150.0, -125.0]    # Right mouth corner
])

class HeadPoseEstimator:
    """Estimates head pose angles from facial landmarks using MediaPipe."""

    def __init__(self, model_path: str = "src/face_auth/core/enrollment/face_landmarker.task"):
        """Initialize head pose estimator with MediaPipe FaceLandmarker (Tasks API)."""
        BaseOptions = mp.tasks.BaseOptions
        FaceLandmarker = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.VIDEO,
            num_faces=1,
        )

        self._landmarker = FaceLandmarker.create_from_options(options)
        self._frame_timestamp_ms = 0

        self._landmark_indices = [
            NOSE_TIP_LANDMARK,
            CHIN_LANDMARK,
            LEFT_EYE_CORNER_LANDMARK,
            RIGHT_EYE_CORNER_LANDMARK,
            LEFT_MOUTH_CORNER_LANDMARK,
            RIGHT_MOUTH_CORNER_LANDMARK,
        ]

    def estimate_pose(self, frame_rgb: np.ndarray) -> Optional[HeadPose]:
        """Estimate head pose from frame.

        Args:
            frame_rgb: Frame in RGB format

        Returns:
            HeadPose with pitch, yaw, roll angles, or None if no face detected
        """
        # Create MediaPipe Image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame_rgb
        )

        # Process with FaceLandmarker
        results = self._landmarker.detect_for_video(mp_image, self._frame_timestamp_ms)
        self._frame_timestamp_ms += 33  # ~30 fps

        if not results.face_landmarks:
            return None

        landmarks = results.face_landmarks[0]
        return self._compute_pose_from_landmarks(landmarks, frame_rgb.shape)

    def _compute_pose_from_landmarks(
        self,
        landmarks,
        image_shape: tuple[int, ...]
    ) -> Optional[HeadPose]:
        """Compute head pose angles from facial landmarks."""
        h, w = image_shape[:2]

        # Convert normalized landmark coordinates to pixel coordinates
        # landmarks is now a list of NormalizedLandmark objects
        face_2d = []
        for idx in self._landmark_indices:
            if idx < len(landmarks):
                x = int(landmarks[idx].x * w)
                y = int(landmarks[idx].y * h)
                face_2d.append([x, y])

        if len(face_2d) < 6:
            return None

        face_2d = np.array(face_2d, dtype=np.float64)

        # Camera intrinsic matrix (approximate)
        focal_length = w
        cam_matrix = np.array([
            [focal_length, 0, w / 2],
            [0, focal_length, h / 2],
            [0, 0, 1]
        ])

        # Assume no lens distortion
        dist_matrix = np.zeros((4, 1), dtype=np.float64)

        # Solve PnP to get rotation and translation vectors
        success, rotation_vector, _ = cv2.solvePnP(
            FACE_MODEL_POINTS,
            face_2d,
            cam_matrix,
            dist_matrix
        )

        if not success:
            return None

        # Convert rotation vector to rotation matrix
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

        # Calculate Euler angles
        angles = self._rotation_matrix_to_euler_angles(rotation_matrix)

        # Normalize pitch angle to [-90, 90] range
        pitch = float(angles[0])
        if pitch > 90:
            pitch -= 180
        elif pitch < -90:
            pitch += 180

        yaw = float(angles[1])
        roll = float(angles[2])

        return HeadPose(pitch=pitch, yaw=yaw, roll=roll)

    def _rotation_matrix_to_euler_angles(self, rotation_matrix: np.ndarray) -> np.ndarray:
        """Convert rotation matrix to Euler angles in degrees."""
        x = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        y = math.atan2(
            -rotation_matrix[2, 0],
            math.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
        )
        z = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        return np.array([x, y, z]) * 180.0 / math.pi
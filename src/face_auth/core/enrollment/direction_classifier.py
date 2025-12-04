import math
from typing import Optional
import cv2
import mediapipe as mp
import numpy as np

from face_auth.core.enrollment import HeadPose, HeadDirection

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

    def __init__(self):
        """Initialize head pose estimator with MediaPipe Face Mesh."""
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True
        )

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
        results = self.face_mesh.process(frame_rgb)

        if not results.multi_face_landmarks:
            return None

        landmarks = results.multi_face_landmarks[0]
        return self._compute_pose_from_landmarks(landmarks, frame_rgb.shape)

    def _compute_pose_from_landmarks(
        self,
        landmarks,
        image_shape: tuple[int, ...]
    ) -> Optional[HeadPose]:
        """Compute head pose angles from facial landmarks."""
        h, w = image_shape[:2]

        # Convert normalized landmark coordinates to pixel coordinates
        face_2d = []
        for idx in self._landmark_indices:
            if idx < len(landmarks.landmark):
                x = int(landmarks.landmark[idx].x * w)
                y = int(landmarks.landmark[idx].y * h)
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


class DirectionClassifier:
    """Classifies head direction from pose angles."""

    def __init__(self, yaw_threshold: float, pitch_threshold: float):
        """Initialize direction classifier.

        Args:
            yaw_threshold: Threshold in degrees for left/right classification
            pitch_threshold: Threshold in degrees for up/down classification
        """
        self.yaw_threshold = yaw_threshold
        self.pitch_threshold = pitch_threshold

    def classify(self, pose: HeadPose) -> HeadDirection:
        """Classify head direction from pose angles.

        Args:
            pose: HeadPose with pitch, yaw, roll angles

        Returns:
            Classified head direction
        """
        # Check if looking straight ahead
        if abs(pose.yaw) < self.yaw_threshold and abs(pose.pitch) < self.pitch_threshold:
            return HeadDirection.FRONT

        # Check horizontal direction (yaw has priority)
        if pose.yaw > self.yaw_threshold:
            return HeadDirection.RIGHT  # Person looking to their right
        elif pose.yaw < -self.yaw_threshold:
            return HeadDirection.LEFT  # Person looking to their left

        # Check vertical direction
        if pose.pitch > self.pitch_threshold:
            return HeadDirection.DOWN
        elif pose.pitch < -self.pitch_threshold:
            return HeadDirection.UP

        # Default to front for ambiguous cases
        return HeadDirection.FRONT

import math

import cv2
import mediapipe as mp
import numpy as np


class FaceDirectionDetector:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True
        )

        self.nose_tip = 1
        self.chin = 152
        self.left_eye_corner = 33
        self.right_eye_corner = 263
        self.left_mouth_corner = 61
        self.right_mouth_corner = 291

    def rotation_matrix_to_euler_angles(self, rotation_matrix):
        x = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        y = math.atan2(-rotation_matrix[2, 0], math.sqrt(rotation_matrix[0, 0] ** 2 +
                                                         rotation_matrix[1, 0] ** 2))
        z = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        return np.array([x, y, z]) * 180. / math.pi

    def get_head_pose(self, landmarks, img_shape):
        """Calculate head pose angles from facial landmarks"""
        h, w = img_shape[:2]

        # Convert normalized coordinates to pixel coordinates
        face_2d = []

        # Key points for pose estimation
        key_points = [
            self.nose_tip,  # Nose tip
            self.chin,  # Chin
            self.left_eye_corner,  # Left eye corner
            self.right_eye_corner,  # Right eye corner
            self.left_mouth_corner,  # Left mouth corner
            self.right_mouth_corner  # Right mouth corner
        ]

        for idx in key_points:
            if idx < len(landmarks.landmark):
                x = int(landmarks.landmark[idx].x * w)
                y = int(landmarks.landmark[idx].y * h)
                face_2d.append([x, y])

        if len(face_2d) < 6:
            return None, None, None

        face_2d = np.array(face_2d, dtype=np.float64)

        # Camera matrix (approximate)
        focal_length = w
        cam_matrix = np.array([
            [focal_length, 0, w / 2],
            [0, focal_length, h / 2],
            [0, 0, 1]
        ])
        # Distortion coefficients
        dist_matrix = np.zeros((4, 1), dtype=np.float64)

        model_points = np.array([
            [0.0, 0.0, 0.0],  # Nose tip
            [0.0, -330.0, -65.0],  # Chin
            [-225.0, 170.0, -135.0],  # Left eye corner
            [225.0, 170.0, -135.0],  # Right eye corner
            [-150.0, -150.0, -125.0],  # Left mouth corner
            [150.0, -150.0, -125.0]  # Right mouth corner
        ])

        success, rotation_vector, translation_vector = cv2.solvePnP(
            model_points, face_2d, cam_matrix, dist_matrix
        )

        if not success:
            return None, None, None

        # Convert rotation vector to rotation matrix
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

        # Calculate Euler angles
        angles = self.rotation_matrix_to_euler_angles(rotation_matrix)

        pitch = angles[0]
        if pitch > 90:
            pitch -= 180
        elif pitch < -90:
            pitch += 180

        yaw = angles[1]
        roll = angles[2]

        return pitch, yaw, roll

    def classify_direction(self, pitch, yaw, roll):

        yaw_threshold = 15
        pitch_threshold = 15

        if abs(yaw) < yaw_threshold and abs(pitch) < pitch_threshold:
            return "front"
        elif yaw > yaw_threshold:
            return "right"  # Person looking to their right (our left)
        elif yaw < -yaw_threshold:
            return "left"  # Person looking to their left (our right)
        elif pitch > pitch_threshold:
            return "down"
        elif pitch < -pitch_threshold:
            return "up"
        else:
            return "front"  # Default to front for ambiguous cases

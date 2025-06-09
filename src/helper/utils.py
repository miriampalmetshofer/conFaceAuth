import cv2
import numpy as np
import mediapipe as mp

from src.helper.enums import HeadDirection



def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def is_eye_open(landmarks, img_w, img_h, eye_indices):
    # Convert normalized coordinates to pixel coordinates
    coords = [(int(landmarks[i].x * img_w), int(landmarks[i].y * img_h)) for i in eye_indices]

    # EAR calculation
    vertical1 = euclidean(coords[1], coords[5])
    vertical2 = euclidean(coords[2], coords[4])
    horizontal = euclidean(coords[0], coords[3])

    ear = (vertical1 + vertical2) / (2.0 * horizontal)
    return ear > 0.25


def are_both_eyes_closed(face_landmarks, img_w, img_h):
    # Indices for one eye (example: left eye)
    left_eye_idx = [33, 160, 158, 133, 153, 144]  # horizontal: 33-133, vertical: 160-144, 158-153
    right_eye_idx = [362, 385, 387, 263, 373, 380]  # horizontal: 362-263, vertical: 385-380, 387-373

    left_open = is_eye_open(face_landmarks, img_w, img_h, left_eye_idx)
    right_open = is_eye_open(face_landmarks, img_w, img_h, right_eye_idx)

    if left_open or right_open:
        return False
    else:
        return True


def get_head_position(image):
    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(static_image_mode=False,
                               max_num_faces=1,
                               min_detection_confidence=0.5,
                               min_tracking_confidence=0.5) as face_mesh:

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        if not results.multi_face_landmarks:
            print("No face landmarks detected.")
            return None

        img_h, img_w, _ = image.shape

        # Landmark indices for eyes, nose, and mouth corners
        key_indices = [33, 263, 1, 61, 291, 199]
        face_landmarks = results.multi_face_landmarks[0]

        if are_both_eyes_closed(face_landmarks.landmark, img_w, img_h):
            print("Both eyes are closed")
            return None

        face_2d = []
        face_3d = []

        for idx in key_indices:
            lm = face_landmarks.landmark[idx]
            x, y = int(lm.x * img_w), int(lm.y * img_h)

            face_2d.append([x, y])
            face_3d.append([x, y, lm.z])

        face_2d = np.array(face_2d, dtype=np.float64)
        face_3d = np.array(face_3d, dtype=np.float64)

        focal_length = 1 * img_w
        cam_matrix = np.array([[focal_length, 0, img_w / 2],
                               [0, focal_length, img_h / 2],
                               [0, 0, 1]])
        dist_matrix = np.zeros((4, 1))

        success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

        rmat, _ = cv2.Rodrigues(rot_vec)
        angles, *_ = cv2.RQDecomp3x3(rmat)

        x, y, z = angles[0] * 360, angles[1] * 360, angles[2] * 360

        print(f"Head rotation angles: x={x}, y={y}, z={z}")

        if y < -4:
            text = HeadDirection.LEFT
        elif y > 4:
            text = HeadDirection.RIGHT
        elif x < -5:
            text = HeadDirection.DOWN
        elif x > 5:
            text = HeadDirection.UP
        else:
            text = HeadDirection.FRONT

        return text

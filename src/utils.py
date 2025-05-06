from collections import deque

from keras_facenet import FaceNet
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mtcnn import MTCNN
import os

detector = MTCNN()
embedder = FaceNet()

trust_score = 0.5
trust_score_min = 0.0
trust_score_max = 1.0
trust_score_decay = 0.05


def load_image(image_path: str) -> np.ndarray:
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found at path: {image_path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def detect_face(image: np.ndarray) -> np.ndarray:
    result = detector.detect_faces(image)[0]
    x, y, width, height = result['box']
    x, y = max(0, x), max(0, y)
    face = image[y:y + height, x:x + width]
    face = cv2.resize(face, (160, 200))
    return face


def get_face_embedding(face_rgb: np.ndarray):
    embedding = embedder.embeddings([face_rgb])[0]
    return embedding


def get_enrollment_embeddings(enrollment_folder) -> list:
    enrollment_files = os.listdir(enrollment_folder)
    enrollment_embeddings = []
    for file in enrollment_files:
        image_path = os.path.join(enrollment_folder, file)
        image = cv2.imread(image_path)
        face_embedding = get_face_embedding(image)
        enrollment_embeddings.append(face_embedding)

    return enrollment_embeddings


def test_with_image_folder(folder_path, enrollment_embeddings, enrollment_images, threshold):
    test_files = os.listdir(folder_path)

    for file in test_files:
        image_path = os.path.join(folder_path, file)
        image = load_image(image_path)
        image = detect_face(image)
        test_embedding = get_face_embedding(image)
        plt.imshow(image)
        plt.show()

        distances = []

        # Compare test embedding with each enrollment embedding
        for idx, enrollment_embedding in enumerate(enrollment_embeddings):
            distance = np.linalg.norm(test_embedding - enrollment_embedding)
            distances.append(distance)

            # Load enrollment image for display
            enrollment_image = cv2.imread(enrollment_images[idx])
            enrollment_image_rgb = cv2.cvtColor(enrollment_image, cv2.COLOR_BGR2RGB)

            print(f"Distance with {enrollment_images[idx]}: {distance:.4f}")

        # Find the closest match
        closest_match_idx = np.argmin(distances)
        closest_match_distance = distances[closest_match_idx]
        # display the closest match
        closest_match_image = cv2.imread(enrollment_images[closest_match_idx])
        closest_match_image_rgb = cv2.cvtColor(closest_match_image, cv2.COLOR_BGR2RGB)
        plt.imshow(closest_match_image_rgb)

        if closest_match_distance < threshold:
            print(f"✅ Match found with {enrollment_images[closest_match_idx]} at distance {closest_match_distance:.4f}")
        else:
            print(f"❌ No match with {enrollment_images[closest_match_idx]} at distance {closest_match_distance:.4f}")

        plt.show()


def cut_faces_in_enrollment():
    enrollment_folder = 'data/enrollment_v2/raw/'
    enrollment_files = os.listdir(enrollment_folder)
    counter = 0

    for file in enrollment_files:
        counter += 1
        image_path = os.path.join(enrollment_folder, file)
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        try:
            face = detect_face(image_rgb)
            cv2.imwrite(f'data/enrollment_v2/face_{counter}.png', cv2.cvtColor(face, cv2.COLOR_RGB2BGR))
        except Exception as e:
            print(f"Error embedding face at file {file}: {e}")


def process_video_and_get_embedding():
    video_path = '../data/images/IMG_7109.mp4'
    cap = cv2.VideoCapture(video_path)

    embeddings = []
    frame_count = 0
    skip_frames = 1

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % skip_frames == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face = detect_face(rgb_frame)

            # save face image
            cv2.imwrite(f'data/enrollment/face_{frame_count}.jpg', cv2.cvtColor(face, cv2.COLOR_RGB2BGR))

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

    return embeddings


def get_closest_match_distance(test_embedding, enrollment_embeddings):
    distances = get_distances_between_enrollment_and_test(test_embedding, enrollment_embeddings)
    closest_match_idx = np.argmin(distances)
    closest_match_distance = distances[closest_match_idx]

    return closest_match_distance


def get_average_of_closest_10_percent(test_embedding, enrollment_embeddings):
    distances = get_distances_between_enrollment_and_test(test_embedding, enrollment_embeddings)
    closest_10_percent_idx = int(len(distances) * 0.1)
    closest_10_percent_distances = sorted(distances)[:closest_10_percent_idx]
    average_distance = np.mean(closest_10_percent_distances)

    return average_distance


def get_distances_between_enrollment_and_test(test_embedding, enrollment_embeddings):
    distances = []

    for idx, enrollment_embedding in enumerate(enrollment_embeddings):
        distance = np.linalg.norm(test_embedding - enrollment_embedding)
        distances.append(distance)
    distances.sort()

    return distances


def calculate_trust_score(window, threshold):
    if len(window) == 0:
        return trust_score  # No change if no history is available

    avg_distance = np.mean(window)

    if avg_distance < threshold:
        return min(trust_score + trust_score_decay, trust_score_max)
    else:
        return max(trust_score - trust_score_decay, trust_score_min)


def calculate_adaptive_threshold(enrollment_embeddings):
    mean_embedding = np.mean(enrollment_embeddings, axis=0)

    distances = np.linalg.norm(enrollment_embeddings - mean_embedding, axis=1)

    mean_distance = np.mean(distances)
    std_distance = np.std(distances)
    threshold = mean_distance + 2 * std_distance

    print(f"Adaptive threshold set at: {threshold:.4f}")
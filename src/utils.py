from collections import deque
from typing import Any

from keras_facenet import FaceNet
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mtcnn import MTCNN
import os

from numpy import floating

detector = MTCNN()
embedder = FaceNet()

initial_trust_score = 0.7
trust_score_min = 0.0
trust_score_max = 1.0
trust_score_decay = 0.05

SKIP_FRAMES = 30
WINDOW_SIZE = 5
THRESHOLD_DISTANCE = 0.7

alpha = 1


def load_image(image_path: str) -> np.ndarray:
    """
    Load an image from the specified path and convert it to RGB format.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found at path: {image_path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def detect_face(image: np.ndarray) -> np.ndarray:
    """
    Detect a face in the given image and return the cropped face image.
    """
    result = detector.detect_faces(image)[0]
    x, y, width, height = result['box']
    x, y = max(0, x), max(0, y)
    face = image[y:y + height, x:x + width]
    face = cv2.resize(face, (160, 200))
    return face


def get_face_embedding(face_rgb: np.ndarray) -> np.ndarray:
    """
    Get the face embedding for a given RGB face image.
    """
    embedding = embedder.embeddings([face_rgb])[0]
    return embedding


def get_enrollment_embeddings(enrollment_folder: str) -> list:
    """
    Get face embeddings for all images in the given enrollment folder.
    """
    enrollment_files = os.listdir(enrollment_folder)
    enrollment_embeddings = []
    for file in enrollment_files:
        image_path = os.path.join(enrollment_folder, file)
        image = cv2.imread(image_path)
        face_embedding = get_face_embedding(image)
        enrollment_embeddings.append(face_embedding)

    return enrollment_embeddings


def test_with_image_folder(folder_path: str, enrollment_embeddings: list, enrollment_images: list,
                           threshold: float) -> None:
    """
    Test and compare test images from a folder with enrollment embeddings.
    """
    test_files = os.listdir(folder_path)

    for file in test_files:
        image_path = os.path.join(folder_path, file)
        image = load_image(image_path)
        image = detect_face(image)
        test_embedding = get_face_embedding(image)
        plt.imshow(image)
        plt.show()

        distances = []

        for idx, enrollment_embedding in enumerate(enrollment_embeddings):
            distance = np.linalg.norm(test_embedding - enrollment_embedding)
            distances.append(distance)

            enrollment_image = cv2.imread(enrollment_images[idx])
            enrollment_image_rgb = cv2.cvtColor(enrollment_image, cv2.COLOR_BGR2RGB)

            print(f"Distance with {enrollment_images[idx]}: {distance:.4f}")

        closest_match_idx = np.argmin(distances)
        closest_match_distance = distances[closest_match_idx]
        closest_match_image = cv2.imread(enrollment_images[closest_match_idx])
        closest_match_image_rgb = cv2.cvtColor(closest_match_image, cv2.COLOR_BGR2RGB)
        plt.imshow(closest_match_image_rgb)

        if closest_match_distance < threshold:
            print(f"✅ Match found with {enrollment_images[closest_match_idx]} at distance {closest_match_distance:.4f}")
        else:
            print(f"❌ No match with {enrollment_images[closest_match_idx]} at distance {closest_match_distance:.4f}")

        plt.show()


def cut_faces_in_enrollment() -> None:
    """
    Cut faces from images in the enrollment folder and save them.
    """
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


def process_video_and_get_embedding(video_path: str) -> list:
    """
    Process the video and extract embeddings from faces in frames.
    """
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

            cv2.imwrite(f'data/enrollment/face_{frame_count}.jpg', cv2.cvtColor(face, cv2.COLOR_RGB2BGR))

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

    return embeddings


def get_closest_match_distance(test_embedding: np.ndarray, enrollment_embeddings: list) -> float:
    """
    Calculate the closest match distance between a test embedding and enrollment embeddings.
    """
    distances = get_distances_between_enrollment_and_test(test_embedding, enrollment_embeddings)
    closest_match_idx = np.argmin(distances)
    closest_match_distance = distances[closest_match_idx]

    return closest_match_distance


def get_average_of_closest_10_percent(test_embedding: np.ndarray, enrollment_embeddings: list) -> floating[Any]:
    """
    Get the average distance of the closest 10% matches for a test embedding.
    """
    distances = get_distances_between_enrollment_and_test(test_embedding, enrollment_embeddings)
    closest_10_percent_idx = int(len(distances) * 0.1)
    closest_10_percent_distances = sorted(distances)[:closest_10_percent_idx]
    average_distance = np.mean(closest_10_percent_distances)

    return average_distance


def get_distances_between_enrollment_and_test(test_embedding: np.ndarray, enrollment_embeddings: list) -> list:
    """
    Get the distances between the test embedding and all enrollment embeddings.
    """
    distances = []

    for idx, enrollment_embedding in enumerate(enrollment_embeddings):
        distance = np.linalg.norm(test_embedding - enrollment_embedding)
        distances.append(distance)
    distances.sort()

    return distances


def get_weighted_average_distance_of_window(window: deque) -> float:
    """
    Calculate the trust score based on the average distance within a window.
    """
    if len(window) == 0:
        return 0.7  # No change if no history is available

        # Exponentially decay weights with a factor `alpha`
    weights = np.exp(np.linspace(-alpha, 0, len(window)))
    weighted_avg_distance = np.average(window, weights=weights)

    return weighted_avg_distance


def calculate_adaptive_threshold(enrollment_embeddings: np.ndarray) -> float:
    """
    Calculate an adaptive threshold based on enrollment embeddings.
    """
    mean_embedding = np.mean(enrollment_embeddings, axis=0)

    distances = np.linalg.norm(enrollment_embeddings - mean_embedding, axis=1)

    mean_distance = np.mean(distances)
    std_distance = np.std(distances)
    threshold = mean_distance + 2 * std_distance

    print(f"Adaptive threshold set at: {threshold:.4f}")

    return threshold


def get_face_coordinates_and_cropped_image_for_frame(frame: np.ndarray) -> tuple | None:
    """
    Get the face coordinates and cropped image for the frame.
    """
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = detector.detect_faces(rgb_frame)

    if len(results) == 0:
        return None

    result = results[0]
    x, y, w, h = result['box']
    x, y = max(0, x), max(0, y)
    face = rgb_frame[y:y + h, x:x + w]

    return x, y, w, h, face


def get_face_embedding_for_frame(face: np.ndarray) -> np.ndarray:
    """
    Get the face embedding for the given face.
    """
    embedding = embedder.embeddings([face])[0]

    return embedding


def calculate_distance_to_enrolment(embedding: np.ndarray, distance_window: deque,
                                    enrollment_embeddings: list) -> floating[Any]:
    """
    Calculate the distance to the enrollment embeddings and update the distance window.
    """
    distance = get_average_of_closest_10_percent(embedding, enrollment_embeddings)

    return distance


def annotate_frame(frame: np.ndarray, x: int, y: int, w: int, h: int, color: tuple) -> None:
    """
    Annotate the frame with face bounding box and trust score information.
    """
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)


def write_summary_frame(out, frame_count: int, unauthenticated_count: int, width: int, height: int) -> None:
    """
    Write a summary frame at the end of the video indicating the number of unauthenticated frames.
    """
    summary_frame = np.ones((height, width, 3), dtype=np.uint8) * 255
    text = f"Frames not authenticated: {unauthenticated_count} / {frame_count // SKIP_FRAMES}"
    cv2.putText(summary_frame, text, (50, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 3)
    out.write(summary_frame)

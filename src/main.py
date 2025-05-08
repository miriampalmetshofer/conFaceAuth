from keras_facenet import FaceNet
import cv2
import numpy as np
from mtcnn import MTCNN
from collections import deque

from src.enums import Color
from src.utils import get_enrollment_embeddings, get_average_of_closest_10_percent, calculate_trust_score, \
    get_face_coordinates_and_copped_image_for_frame, get_face_embedding_for_frame, calculate_distance_to_enrolment, \
    annotate_frame, write_summary_frame, THICKNESS, FONT_SIZE, THRESHOLD_DISTANCE, SKIP_FRAMES, WINDOW_SIZE

# Constants
ENROLLMENT_FOLDER = '../data/enrollment_v2/processed/'
VIDEO_PATH = '../data/images/test_1.mp4'
OUTPUT_PATH = '../data/images/output/output-video-4.mp4'

detector = MTCNN()
embedder = FaceNet()

enrollment_embeddings = get_enrollment_embeddings(ENROLLMENT_FOLDER)


def test_with_video(video_path, output_path=OUTPUT_PATH) -> None:
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    frame_count = 0
    unauthenticated_count = 0
    distance_window = deque(maxlen=WINDOW_SIZE)
    distance = 0
    trust_score = 0
    color = Color.RED.value

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        try:
            if frame_count % SKIP_FRAMES == 0:
                x, y, w, h, face = get_face_coordinates_and_copped_image_for_frame(frame)
                embedding = get_face_embedding_for_frame(face)
                distance = calculate_distance_to_enrolment(embedding, distance_window)
                trust_score = calculate_trust_score(distance_window, THRESHOLD_DISTANCE)

                if trust_score >= 0.6:
                    color = Color.GREEN.value
                else:
                    unauthenticated_count += 1
                    color = Color.RED.value
                annotate_frame(frame, x, y, w, h, color)

        except Exception as e:
            print(f"Error embedding face at frame {frame_count}: {e}")

        cv2.putText(frame, f"Distance: {distance:.4f}, Trust: {trust_score:.2f}", (100, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, color, THICKNESS)
        out.write(frame)
        frame_count += 1

    write_summary_frame(out, frame_count, unauthenticated_count, width, height)

    cap.release()
    out.release()
    cv2.destroyAllWindows()


test_with_video(VIDEO_PATH)

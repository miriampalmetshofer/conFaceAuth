from keras_facenet import FaceNet
import cv2
from mtcnn import MTCNN
from collections import deque
from src.enums import Color
from src.utils import get_enrollment_embeddings, get_weighted_average_distance_of_window, \
    get_face_embedding_for_frame, calculate_distance_to_enrolment, \
    annotate_frame, write_summary_frame, SKIP_FRAMES, WINDOW_SIZE, \
    get_face_coordinates_and_cropped_image_for_frame

ENROLLMENT_FOLDER = '../data/enrollment_v2/processed/'
VIDEO_PATH = '../data/images/no_face_test_3.mp4'
OUTPUT_PATH = '../data/images/output/output-video-5.mp4'

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
    weighted_average_distance = 0.7
    color = Color.RED.value

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        try:
            if frame_count % SKIP_FRAMES == 0:
                face_data = get_face_coordinates_and_cropped_image_for_frame(frame)

                if face_data is None:  # If no face detected, handle this case
                    cv2.putText(frame, "No face detected", (300, 300), cv2.FONT_HERSHEY_SIMPLEX, 3,
                                Color.RED.value, 3)
                    distance = 1
                    distance_window.append(distance)
                else:
                    x, y, w, h, face = face_data
                    embedding = get_face_embedding_for_frame(face)
                    distance = calculate_distance_to_enrolment(embedding, distance_window, enrollment_embeddings)
                    distance_window.append(distance)
                    annotate_frame(frame, x, y, w, h, color)

                print(distance_window)
                weighted_average_distance = get_weighted_average_distance_of_window(distance_window)
                print(weighted_average_distance)

                if weighted_average_distance <= 0.8:
                    color = Color.GREEN.value
                else:
                    unauthenticated_count += 1
                    color = Color.RED.value


        except Exception as e:
            print(f"Error embedding face at frame {frame_count}: {e}")

        cv2.putText(frame, f"Distance: {distance:.4f}, Trust: {weighted_average_distance:.5f}", (100, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 3, color, 3)
        out.write(frame)
        frame_count += 1

    write_summary_frame(out, frame_count, unauthenticated_count, width, height)
    cap.release()
    out.release()
    cv2.destroyAllWindows()


test_with_video(VIDEO_PATH)

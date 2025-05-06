# %%
from enum import Enum

from keras_facenet import FaceNet
import cv2
import numpy as np
from mtcnn import MTCNN
from collections import deque
from src.utils import get_enrollment_embeddings, get_average_of_closest_10_percent, calculate_trust_score

enrollment_folder = '../data/enrollment_v2/processed/'

detector = MTCNN()
embedder = FaceNet()

# %%
enrollment_embeddings = get_enrollment_embeddings(enrollment_folder)


# %%
def test_with_video(video_path, output_path='output_video.mp4'):
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    frame_count = 0
    skip_frames = 30
    number_of_frames_unauthenticated = 0
    window_size = 5
    threshold_distance = 0.7
    distance_window = deque(maxlen=window_size)
    distance = 0
    trust_score = 0
    Color = Enum('Color', [('RED', (0, 0, 255)), ('GREEN', (0, 255, 0))])
    color = Color.GREEN.value

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        try:
            if frame_count % skip_frames == 0:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = detector.detect_faces(rgb_frame)
                if len(results) == 0:
                    number_of_frames_unauthenticated += 1
                    raise Exception("No faces detected")
                result = results[0]
                x, y, w, h = result['box']
                x, y = max(0, x), max(0, y)
                face = rgb_frame[y:y + h, x:x + w]
                embedding = embedder.embeddings([face])[0]

                distance = get_average_of_closest_10_percent(embedding, enrollment_embeddings)

                distance_window.append(distance)

                trust_score = calculate_trust_score(distance_window, threshold_distance)

                if trust_score >= 0.7:
                    color = Color.GREEN.value
                else:
                    number_of_frames_unauthenticated += 1
                    color = Color.RED.value

                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        except Exception as e:
            print(f"Error embedding face at frame {frame_count}: {e}")

        cv2.putText(frame, f"Distance: {distance:.4f}, Trust: {trust_score:.2f}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 3, color, 3)
        out.write(frame)
        frame_count += 1

    # Write summary frame
    summary_frame = np.ones((height, width, 3), dtype=np.uint8) * 255
    text = f"Frames not authenticated: {number_of_frames_unauthenticated} / {frame_count / skip_frames}"
    cv2.putText(summary_frame, text, (50, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
    out.write(summary_frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()


test_with_video('../data/images/test_1.mp4', '../data/images/output/output-video-3.mp4')

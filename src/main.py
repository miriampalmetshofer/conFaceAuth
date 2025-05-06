# %%
from keras_facenet import FaceNet
import cv2
import numpy as np
from mtcnn import MTCNN
from src.utils import get_enrollment_embeddings, get_average_of_closest_10_percent

enrollment_folder = 'data/enrollment_v2/processed/'

detector = MTCNN()
embedder = FaceNet()

# %%
enrollment_embeddings = get_enrollment_embeddings(enrollment_folder)

# %%
# compute threshold based on enrollment data
mean_embedding = np.mean(enrollment_embeddings, axis=0)

distances = np.linalg.norm(enrollment_embeddings - mean_embedding, axis=1)

mean_distance = np.mean(distances)
std_distance = np.std(distances)
threshold = mean_distance + 2 * std_distance

print(f"Adaptive threshold set at: {threshold:.4f}")

# %%
def test_with_video(video_path, output_path='output_video.mp4'):
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    frame_count = 0
    skip_frames = 30
    number_of_frames_unauthenticated = 0

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

                    if distance < threshold:
                        color = (0, 255, 0)
                    else:
                        number_of_frames_unauthenticated += 1
                        color = (0, 0, 255)

                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(frame, f"Distance: {distance:.4f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)

        except Exception as e:
            print(f"Error embedding face at frame {frame_count}: {e}")

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

test_with_video('data/images/motion-illumination-change-test-2.mp4', 'data/images/output/output-video-3.mp4')




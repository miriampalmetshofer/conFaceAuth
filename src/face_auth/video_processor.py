import cv2

from src.helper.enums import Color
from src.helper.utils import draw_detection_box, write_summary_frame

class VideoProcessor:
    def __init__(self, face_detector, embedding_manager, authenticator, threshold):
        self.face_detector = face_detector
        self.embedding_manager = embedding_manager
        self.authenticator = authenticator


    def process_video(self, video_path, output_path, skip_frames):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        frame_count = 0
        unauthenticated_count = 0
        distance = 0
        trust_score = 0
        color = Color.GREEN.value

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            try:
                if frame_count % skip_frames == 0:
                    result = self.face_detector.detect_and_crop(frame)

                    if result is None:  # If no face detected, handle this case
                        cv2.putText(frame, "No face detected", (300, 300), cv2.FONT_HERSHEY_SIMPLEX, 3,
                                    Color.RED.value, 3)
                        distance = 1
                    else:
                        face, box_coordinates = result
                        draw_detection_box(frame, box_coordinates)
                        embedding = self.embedding_manager.get_embedding(face)
                        distance = self.authenticator.compute_distance_between_embedding_and_enrolment(embedding)

                    self.authenticator.append_distance_to_window(distance)
                    trust_score = self.authenticator.trust_score

                    is_authenticated = self.authenticator.is_authenticated()

                    if is_authenticated:
                        color = Color.GREEN.value
                    else:
                        unauthenticated_count += 1
                        color = Color.RED.value

            except Exception as e:
                print(f"Error embedding face at frame {frame_count}: {e}")
                raise e

            cv2.putText(frame, f"Distance: {distance:.4f}, Trust: {trust_score:.5f}", (100, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 3, color, 3)
            out.write(frame)
            frame_count += 1

        write_summary_frame(out, frame_count / skip_frames, unauthenticated_count, width, height)
        cap.release()
        out.release()
        cv2.destroyAllWindows()


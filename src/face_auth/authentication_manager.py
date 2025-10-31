import cv2
import pandas as pd
import os

from face_auth.authenticator import Authenticator
from face_auth.embedder import EmbeddingManager
from face_auth.face_detector import FaceDetector
from face_auth.video_utils import get_video_rotation, rotate_frame
from helper.enums import Color


class AuthenticationManager:
    def __init__(self, face_detector: FaceDetector, embedding_manager: EmbeddingManager,
                 authenticator: Authenticator, config):
        self.face_detector = face_detector
        self.embedding_manager = embedding_manager
        self.authenticator = authenticator
        self.config = config


    def flatten_config_for_csv(self, config: dict, video_path) -> dict:
        """Extracts the needed fields from config."""
        flattened = {
            "video_path": video_path,
            "skip_frames": config.get("skip_frames", ""),
            "window_size": config.get("window_size", ""),
            "threshold": config.get("threshold", ""),
            "embedder": config.get("embedder", ""),
            "detector": config.get("detector", ""),
            "similarity_computation": config.get("similarity_computation", ""),
            "enrollment_frames_per_direction": config.get("enrollment_frames_per_direction", ""),
            "no_face_penalty": config.get("no_face_penalty", ""),
            "alpha": config.get("alpha", "")
        }
        return flattened

    def write_results_to_csv(self, results, results_csv_path, video_path) -> None:
        df = pd.DataFrame(results)
        flat_config = self.flatten_config_for_csv(self.config, video_path)

        for key, value in flat_config.items():
            df[key] = value

        file_exists = os.path.isfile(results_csv_path)
        print(f"{'Appending' if file_exists else 'Creating'} results to {results_csv_path}")
        df.to_csv(results_csv_path, mode='a', header=not file_exists, index=False)

    def append_frame_result(self, results, frame_count, predicted_state, distance):
        results.append({
            'frame': frame_count,
            'predicted_state': predicted_state,
            'distance': distance,
            'risk_score': self.authenticator.risk_score
        })

    def _check_video_requirements(self, cap: cv2.VideoCapture) -> None:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps != 30:
            print(f"Info: Video FPS is {fps}.")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Total frames in video: {total_frames}")


    def process_video(self, video_path, skip_frames, results_csv_path):
        # Detect video rotation from metadata
        rotation_angle = get_video_rotation(video_path)

        cap = cv2.VideoCapture(video_path)

        self._check_video_requirements(cap)

        frame_count = 1
        results = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Apply rotation based on metadata
            frame = rotate_frame(frame, rotation_angle)

            try:
                if frame_count == 1 or frame_count % skip_frames == 0:
                    result = self.face_detector.detect_and_crop(frame)

                    if result is None:
                        print(f"No face detected at frame {frame_count}.")
                        distance = self.config.get("no_face_penalty")
                        self.authenticator.append_distance_to_window_and_update_risk_score(distance)
                        predicted_state = "No Face"
                    else:
                        face, _ = result
                        embedding = self.embedding_manager.get_embedding(face)
                        distance = self.authenticator.compute_distance_between_embedding_and_enrollment(embedding)
                        self.authenticator.append_distance_to_window_and_update_risk_score(distance)
                        predicted_state = 'Unlocked' if self.authenticator.is_authenticated() else 'Locked'

                    print(f"Frame {frame_count}: Predicted State={predicted_state}, Distance={distance:.4f}, Risk Score={self.authenticator.risk_score:.4f}")

                    self.append_frame_result(results, frame_count, predicted_state, distance)

            except Exception as e:
                print(f"Error processing frame {frame_count}: {e}")
                raise e

            frame_count += 1

        cap.release()
        cv2.destroyAllWindows()
        self.write_results_to_csv(results, results_csv_path, video_path)


    def process_live_stream(self, skip_frames=30):
        cap = cv2.VideoCapture(0)  # MacBook webcam

        if not cap.isOpened():
            print("Cannot open webcam.")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        frame_count = 0
        risk_score = 0
        color = Color.GREEN.value
        threshold = self.config.get('threshold')

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Cannot read frame.")
                break

            try:
                if frame_count % skip_frames == 0:

                    result = self.face_detector.detect_and_crop(frame)

                    if result is None:
                        cv2.putText(frame, "No face detected", (70, 70), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    Color.RED.value, 2)
                        distance = 1
                    else:
                        face, box_coordinates = result
                        self.draw_detection_box(frame, box_coordinates)
                        embedding = self.embedding_manager.get_embedding(face)
                        distance = self.authenticator.compute_distance_between_embedding_and_enrollment(embedding)

                    self.authenticator.append_distance_to_window_and_update_risk_score(distance)
                    risk_score = self.authenticator.risk_score
                    is_authenticated = self.authenticator.is_authenticated()

                    color = Color.GREEN.value if is_authenticated else Color.RED.value

            except Exception as e:
                print(f"Error at frame {frame_count}: {e}")
                continue

            cv2.putText(frame, f"{risk_score:.4f} (Distance) < {threshold} (Threshold)", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

            cv2.imshow('Live Face Authentication', frame)

            frame_count += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def draw_detection_box(self, frame, points):
        x, y, w, h = points
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)

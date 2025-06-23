import cv2
import pandas as pd
import json
import os

from src.face_auth import Authenticator, EmbeddingManager, FaceDetector
from src.helper.enums import Color


class VideoProcessor:
    def __init__(self, face_detector: FaceDetector, embedding_manager: EmbeddingManager,
                 authenticator: Authenticator, config):
        self.face_detector = face_detector
        self.embedding_manager = embedding_manager
        self.authenticator = authenticator
        self.config = config

    def load_ground_truth(self, annotations_csv_path, video_filename):
        """Load ground truth labels from the annotations CSV."""
        print(f"Annotations CSV path: {annotations_csv_path}")
        df = pd.read_csv(annotations_csv_path)
        target_basename = os.path.basename(video_filename)
        row = df[df['video'].str.contains(target_basename, na=False)]
        if row.empty:
            raise ValueError(f"No annotation found for {target_basename} in {annotations_csv_path}.")

        labels_json = row.iloc[0]['videoLabels']
        return json.loads(labels_json)

    def label_frame_from_ground_truth(self, ground_truth, frame_number):
        """Return 'Unlocked' or 'Lock' based on frame number."""
        for segment in ground_truth:
            label = segment['timelinelabels'][0]
            for range_dict in segment['ranges']:
                if range_dict['start'] <= frame_number < range_dict['end']:
                    return label
        raise ValueError(f"No label for {frame_number} in {ground_truth}")

    def flatten_config_for_csv(self, config: dict, video_path) -> dict:
        """Extracts the needed fields from config."""
        flattened = {
            "video_path": video_path,
            "annotations_file": config.get("annotations_file", ""),
            "skip_frames": config.get("skip_frames", ""),
            "window_size": config.get("window_size", ""),
            "threshold": config.get("threshold", ""),
            "embedder": config.get("embedder", ""),
            "detector": config.get("detector", ""),
            "similarity_computation": config.get("similarity_computation", ""),
            "enrollment_frames_per_direction": config.get("enrollment_frames_per_direction", ""),
            "enrollment_folder": config.get("base_path", "") + "/enrollments"  # assuming based on naming convention
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

    def append_frame_result(self, results, frame_count, predicted_label, true_label, match, distance):
        results.append({
            'frame': frame_count,
            'predicted_label': predicted_label,
            'true_label': true_label,
            'match': match,
            'distance': distance,
            'risk_score': self.authenticator.risk_score
        })


    def process_video(self, video_path, skip_frames, annotations_csv_path, results_csv_path):
        cap = cv2.VideoCapture(video_path)
        ground_truth = self.load_ground_truth(annotations_csv_path, video_path)

        frame_count = 1
        results = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if 'mobile' in video_path.lower(): # mobile videos are 90 degrees rotated
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

            try:
                if frame_count % skip_frames == 0:
                    result = self.face_detector.detect_and_crop(frame)

                    if result is None:
                        print(f"No face detected at frame {frame_count}.")
                        cv2.imwrite(f"no_face/no_face_frame_{frame_count}.jpg", frame)
                        distance = self.config.get("no_face_penalty")
                        self.authenticator.append_distance_to_window_and_update_risk_score(distance)
                        predicted_label = "No Face"
                    else:
                        face, _ = result
                        embedding = self.embedding_manager.get_embedding(face)
                        distance = self.authenticator.compute_distance_between_embedding_and_enrollment(embedding)
                        self.authenticator.append_distance_to_window_and_update_risk_score(distance)
                        predicted_label = 'Unlocked' if self.authenticator.is_authenticated() else 'Locked'

                    true_label = self.label_frame_from_ground_truth(ground_truth, frame_count)
                    match = predicted_label == true_label
                    print(f"Frame {frame_count}: Predicted={predicted_label}, Ground Truth={true_label}, Match={match}")

                    self.append_frame_result(results, frame_count, predicted_label, true_label, match, distance)

            except Exception as e:
                print(f"Error embedding face at frame {frame_count}: {e}")
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
        distance = 0
        risk_score = 0
        color = Color.GREEN.value

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            try:
                if frame_count % skip_frames == 0:
                    result = self.face_detector.detect_and_crop(frame)

                    if result is None:
                        cv2.putText(frame, "No face detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
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

            cv2.putText(frame, f"Distance: {distance:.4f}", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            cv2.putText(frame, f"Trust: {risk_score:.4f}", (30, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

            cv2.imshow('Live Face Authentication', frame)

            frame_count += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def draw_detection_box(self, frame, points):
        x, y, w, h = points
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)

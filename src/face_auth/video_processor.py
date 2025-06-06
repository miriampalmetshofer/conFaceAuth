import cv2
import pandas as pd
import json
import os

from src.face_auth import Authenticator, EmbeddingManager, FaceDetector
from src.helper.enums import Color
from src.helper.utils import draw_detection_box

class VideoProcessor:
    def __init__(self, face_detector: FaceDetector, embedding_manager: EmbeddingManager, authenticator: Authenticator, threshold):
        self.face_detector = face_detector
        self.embedding_manager = embedding_manager
        self.authenticator = authenticator

    def load_ground_truth(self, annotations_csv_path, video_filename):
        print(f"Annotations CSV path: {annotations_csv_path}")
        df = pd.read_csv(annotations_csv_path)
        base_filename = os.path.basename(video_filename)
        row = df[df['video'] == base_filename]
        if row.empty:
            raise ValueError(f"No annotation found for {base_filename} in CSV.")
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

    def process_video(self, video_path, output_path, skip_frames, annotations_csv_path):
        cap = cv2.VideoCapture(video_path)
        ground_truth = self.load_ground_truth(annotations_csv_path, video_path)

        frame_count = 1
        match_count = 0
        total_compared = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            try:
                if frame_count % skip_frames == 0:
                    result = self.face_detector.detect_and_crop(frame)

                    if result is None:
                        distance = 1
                    else:
                        face, _ = result
                        embedding = self.embedding_manager.get_embedding(face)
                        distance = self.authenticator.compute_distance_between_embedding_and_enrolment(embedding)

                    self.authenticator.append_distance_to_window_and_update_trust_score(distance)
                    is_authenticated = self.authenticator.is_authenticated()

                    predicted_label = 'Unlocked' if is_authenticated else 'Lock'
                    true_label = self.label_frame_from_ground_truth(ground_truth, frame_count)

                    if true_label is not None:
                        match = predicted_label == true_label
                        print(f"Frame {frame_count}: Predicted={predicted_label}, Ground Truth={true_label}, Match={match}")
                        total_compared += 1
                        if match:
                            match_count += 1

            except Exception as e:
                print(f"Error embedding face at frame {frame_count}: {e}")
                raise e

            frame_count += 1

        cap.release()
        cv2.destroyAllWindows()


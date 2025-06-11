from collections import defaultdict
from pathlib import Path
import cv2
import numpy as np

from src.face_auth.face_direction_detector import FaceDirectionDetector


class EnrolmentVideoProcessor:
    def __init__(self, video_path):
        self.video_path = video_path
        self.frames_by_direction = defaultdict(list)

    def process_video(self, frame_interval=5):
        """
        Process video and store frames in memory categorized by head direction.
        """
        detector = FaceDirectionDetector()
        cap = cv2.VideoCapture(str(self.video_path))

        if not cap.isOpened():
            print(f"Error: Could not open video {self.video_path}")
            return

        frame_count = 0
        print(f"Processing video: {self.video_path}")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % frame_interval != 0:
                continue

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = detector.face_mesh.process(rgb_frame)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    pitch, yaw, roll = detector.get_head_pose(face_landmarks, frame.shape)
                    if pitch is not None and yaw is not None:
                        direction = detector.classify_direction(pitch, yaw, roll)
                        self.frames_by_direction[direction].append(frame)

                        print(f"Frame {frame_count}: {direction} (pitch={pitch:.1f}°, yaw={yaw:.1f}°, roll={roll:.1f}°)")

        cap.release()
        print("\nProcessing complete!")
        for direction, frames in self.frames_by_direction.items():
            print(f"  {direction}: {len(frames)} frames")


    def get_enrolment_frames(self, frames_per_direction=3):
        """
        Sample 5 frames per direction from the middle of the array.
        Returns a dict: {direction: [frames]}
        """
        sampled_frames = {}
        for direction, frames in self.frames_by_direction.items():
            count = len(frames)
            if count == 0:
                sampled_frames[direction] = []
                continue

            mean = count // 2
            stddev = count / 4  # heuristic: ~99.7% within range for large sets
            indices = np.clip(
                np.random.normal(loc=mean, scale=stddev, size=frames_per_direction).astype(int),
                0, count - 1
            )
            sampled_frames[direction] = [frames[i] for i in indices]

        print("\nSampled frames by direction:")
        for direction, frames in sampled_frames.items():
            print(f"  {direction}: {len(frames)} frames")

        return sampled_frames

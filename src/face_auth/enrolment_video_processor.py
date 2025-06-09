import cv2
import os
from collections import defaultdict

from src.helper.enums import HeadDirection
from src.helper.utils import get_head_position

class EnrolmentVideoProcessor:
    def __init__(self, video_path, output_folder, frames_per_direction=3):
        self.video_path = video_path
        self.output_folder = output_folder
        os.makedirs(output_folder, exist_ok=True)
        self.frames_per_direction = frames_per_direction
        self.collected = defaultdict(list)

    def is_blurry(self, image, threshold=12):
        return cv2.Laplacian(image, cv2.CV_64F).var() < threshold

    def process(self):
        cap = cv2.VideoCapture(self.video_path)
        frame_idx = 0

        while cap.isOpened():
            if all(len(self.collected[dir]) >= self.frames_per_direction for dir in HeadDirection):
                print("Collected enough frames for all directions. Stopping early.")
                break
            frame_idx += 1

            print(f"Processing frame {frame_idx}...")
            ret, frame = cap.read()
            if not ret:
                print("End of video or cannot read the frame.")
                break

            if self.is_blurry(frame):
                print(f"Frame {frame_idx} is blurry. Skipping.")
                continue

            direction = get_head_position(frame)

            if direction is None:
                print(f"Frame {frame_idx} does not match any direction. Skipping.")
                continue

            if len(self.collected[direction]) > self.frames_per_direction:
                print(f"Collected enough frames for {direction}. Skipping frame {frame_idx}.")
                continue

            img_name = f"{direction.value}_{frame_idx}.jpg"
            out_path = os.path.join(self.output_folder, img_name)
            cv2.imwrite(out_path, frame)
            print(f"Collected frame for {direction.name}: {out_path}")
            self.collected[direction].append(out_path)

        cap.release()

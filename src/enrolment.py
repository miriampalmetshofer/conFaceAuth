from pathlib import Path
import cv2
from src.face_auth import EnrolmentVideoProcessor, ConfigManager

config = ConfigManager("config.json").get('enrolment')

enrollment = EnrolmentVideoProcessor(
    video_path=config.get("enrolment_video_path"),
)
enrollment.process_video()
frames = enrollment.get_enrolment_frames(frames_per_direction=config.get("enrolment_frames_per_direction"))

for direction, frames_list in frames.items():
    enrolment_folder = Path(config.get("enrolment_folder"))
    direction_dir = enrolment_folder / direction
    direction_dir.mkdir(exist_ok=True)
    for i, frame in enumerate(frames_list):
        frame_path = direction_dir / f"frame_{i:03d}.jpg"
        cv2.imwrite(str(frame_path), frame)
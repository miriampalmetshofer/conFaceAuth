from pathlib import Path
import cv2

from src.face_auth import EnrolmentVideoProcessor, ConfigManager

config = ConfigManager("config.json").get('enrolment')

enrollment = EnrolmentVideoProcessor(
    video_path=config.get("enrolment_video_path"),
)
enrollment.process_video()
frames = enrollment.get_enrolment_frames(frames_per_direction=config.get("enrolment_frames_per_direction"))

if not frames:
    raise ValueError("No frames were derived from enrolment. Please check the video and ensure it contains detectable faces.")

for direction, frames_list in frames.items():
    enrolment_folder = Path(config.get("enrolment_folder"))
    for i, frame in enumerate(frames_list):
        frame_path = enrolment_folder / f"{direction}_{i:03d}.jpg"
        if not 'desktop' in config.get("enrolment_video_path").lower():
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        cv2.imwrite(str(frame_path), frame)

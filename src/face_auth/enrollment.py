from src.face_auth.enrollment_video_processor import EnrollmentVideoProcessor



class EnrollmentManager:
    def __init__(self, enrollment_folder: str, enrollment_video: str) -> None:
        self.enrollment_video = enrollment_video
        self.enrollment_folder = enrollment_folder

    def enroll(self, frames_per_direction: int) -> None:
        enrollment_video_processor = EnrollmentVideoProcessor(
            video_path=self.enrollment_video
        )
        frames_sorted_by_direction = enrollment_video_processor.get_frames_sorted_by_direction_from_video()
        samples_frames = enrollment_video_processor.get_enrollment_frames_per_direction(frames_sorted_by_direction, frames_per_direction)
        if not samples_frames:
            raise ValueError(
                "No frames were derived from enrollment. Please check the video and ensure it contains detectable faces.")

        enrollment_video_processor.save_enrollment_frames_to_folder(samples_frames, self.enrollment_folder)


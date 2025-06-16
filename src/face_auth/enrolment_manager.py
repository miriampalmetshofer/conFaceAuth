from src.face_auth.enrolment_video_processor import EnrolmentVideoProcessor



class EnrollmentManager:
    def __init__(self, enrollment_folder: str, enrolment_video: str) -> None:
        self.enrolment_video = enrolment_video
        self.enrollment_folder = enrollment_folder

    def enroll(self, frames_per_direction: int) -> None:
        enrolment_video_processor = EnrolmentVideoProcessor(
            video_path=self.enrolment_video
        )
        frames_sorted_by_direction = enrolment_video_processor.get_frames_sorted_by_direction_from_video()
        samples_frames = enrolment_video_processor.get_enrolment_frames(frames_sorted_by_direction, frames_per_direction)
        if not samples_frames:
            raise ValueError(
                "No frames were derived from enrolment. Please check the video and ensure it contains detectable faces.")

        enrolment_video_processor.save_enrolment_frames_to_folder(samples_frames, self.enrollment_folder)


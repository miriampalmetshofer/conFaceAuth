from src.face_auth import EnrolmentVideoProcessor


enrollment = EnrolmentVideoProcessor(
    video_path="../data/images/enrolment_test.mp4",
    output_folder="../data/enrolment"
)

enrollment.process()

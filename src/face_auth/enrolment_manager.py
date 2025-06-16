import os
import cv2
from src.face_auth.embedder import EmbeddingManager
from src.face_auth.face_detector import FaceDetector


class EnrollmentManager:
    def __init__(self, enrollment_folder: str):
        self.enrollment_folder = enrollment_folder
        self.face_detector = FaceDetector()
        self.embedding_manager = EmbeddingManager()
        self.embeddings = []

        self._get_embeddings_from_enrolment_images()

    def _get_embeddings_from_enrolment_images(self) -> None:
        if not os.path.exists(self.enrollment_folder) or not os.listdir(self.enrollment_folder):
            raise FileNotFoundError(f"No images found in the enrollment folder: {self.enrollment_folder}. Please ensure the folder exists and contains images.")

        for filename in os.listdir(self.enrollment_folder):
            image_path = os.path.join(self.enrollment_folder, filename)
            image = cv2.imread(image_path)
            result = self.face_detector.detect_and_crop(image)
            if result is None:
                print(f"No face detected in {filename}. Skipping.")
            face, _ = result
            embedding = self.embedding_manager.get_embedding(face)
            self.embeddings.append(embedding)

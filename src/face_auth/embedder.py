import os

import cv2
from keras_facenet import FaceNet

class EmbeddingManager:
    def __init__(self, embedder_name="FaceNet"):
        if embedder_name == "FaceNet":
            self.model = FaceNet()
        else:
            raise ValueError(f"Unsupported embedder: {embedder_name}")

        self.embeddings = []


    def get_embedding(self, face_rgb):
        return self.model.embeddings([face_rgb])[0]

    def initialize_embeddings_from_enrollment_images(self, enrollment_folder: str, face_detector) -> None:

        if not os.path.exists(enrollment_folder) or not os.listdir(enrollment_folder):
            raise FileNotFoundError(
                f"No images found in the enrollment folder: {enrollment_folder}. Please ensure the folder exists and contains images.")

        print(f"Found {len(os.listdir(enrollment_folder))} images in the enrollment folder.")
        for filename in os.listdir(enrollment_folder):
            image_path = os.path.join(enrollment_folder, filename)
            image = cv2.imread(image_path)
            result = face_detector.detect_and_crop(image)
            if result is None:
                print(f"No face detected in {filename}. Skipping.")
            face, _ = result
            embedding = self.get_embedding(face)
            if embedding is not None:
                self.embeddings.append(embedding)
            else:
                print(f"Failed to get enrollment embedding for {filename}. Skipping.")

        print(f"Done computing enrollment embeddings: {len(self.embeddings)}. Embeddings stored in EmbeddingsManager.")

from keras_facenet import FaceNet
import numpy as np


class EmbeddingManager:
    def __init__(self, embedder_name="FaceNet"):
        if embedder_name == "FaceNet":
            self.model = FaceNet()
        else:
            raise ValueError(f"Unsupported embedder: {embedder_name}")

    def get_embedding(self, face_rgb):
        return self.model.embeddings([face_rgb])[0]

    def get_average_distance(self, test_embedding, enrollment_embeddings, top_k=0.1):
        distances = [np.linalg.norm(test_embedding - emb) for emb in enrollment_embeddings]
        k = max(1, int(len(distances) * top_k))
        return np.mean(distances[:k])

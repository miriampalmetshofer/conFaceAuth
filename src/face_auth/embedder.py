from keras_facenet import FaceNet


class Embedder:
    """Handles embedding generation for face images."""

    def __init__(self, embedder_name="FaceNet"):
        if embedder_name == "FaceNet":
            self.model = FaceNet()
        else:
            raise ValueError(f"Unsupported embedder: {embedder_name}")

    def get_embedding(self, face_rgb):
        """Generate embedding for a single face image."""
        return self.model.embeddings([face_rgb])[0]
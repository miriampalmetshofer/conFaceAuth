"""InsightFace (ArcFace) embedder backend."""
import numpy as np
import cv2


class InsightFaceBackend:
    """Face embedder using InsightFace models (ArcFace, CosFace, etc.)."""

    def __init__(self, model_name: str = 'buffalo_l', det_size: tuple = (640, 640)):
        """Initialize InsightFace model.

        Args:
            model_name: Name of the InsightFace model pack to use
                       Options: 'buffalo_l' (default), 'buffalo_s', 'buffalo_sc',
                               'antelopev2', etc.
            det_size: Detection size for the model (width, height)
                     Larger = more accurate but slower
                     Default: (640, 640)
        """
        try:
            from insightface.app import FaceAnalysis
        except ImportError:
            raise ImportError(
                "insightface is not installed. "
                "Install it with: pip install insightface onnxruntime"
            )

        self.model_name = model_name
        self.det_size = det_size

        # Initialize FaceAnalysis with the specified model
        self._app = FaceAnalysis(
            name=model_name,
            providers=['CPUExecutionProvider']  # Use CPU by default, can be configured for GPU
        )
        self._app.prepare(ctx_id=0, det_size=det_size)

    def get_embedding(self, face_rgb: np.ndarray) -> np.ndarray:
        """Generate embedding vector using InsightFace.

        Args:
            face_rgb: Preprocessed face image in RGB format

        Returns:
            512-dimensional embedding vector (for most models)
        """
        # InsightFace expects BGR format
        face_bgr = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2BGR)

        # Get faces from image
        faces = self._app.get(face_bgr)

        if not faces or len(faces) == 0:
            # If no face detected, return zero vector
            # This maintains consistency with the existing flow
            return np.zeros(512, dtype=np.float32)

        # Return embedding of the first (largest) detected face
        # InsightFace returns normalized embeddings
        return faces[0].embedding
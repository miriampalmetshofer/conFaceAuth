"""InsightFace (ArcFace) embedder backend."""
from typing import Any, Dict, Tuple

import numpy as np
import cv2
import logging

from face_auth.core.embedder.backend.embedder_backend import EmbedderBackend
from face_auth.core.embedder.models import EmbeddingResult
from face_auth.core.detection.models import BoundingBox


class InsightFaceBackend(EmbedderBackend):
    """Face embedder using InsightFace ArcFace model with integrated detection."""

    def __init__(
        self,
        model_name: str,
        det_size: Tuple[int, int],
        min_detection_confidence: float,
    ):
        """Initialize InsightFace model.

        Args:
            model_name: Name of the InsightFace model pack to use
                       Options: 'buffalo_l' (ResNet50, default, best accuracy),
                               'buffalo_s' (smaller/faster),
                               'buffalo_sc' (smallest/fastest)
            det_size: Detection size for the model (width, height)
                     Larger = more accurate but slower
            min_detection_confidence: Minimum detection confidence threshold (0-1)
                                     Faces with confidence below this are treated as "no face"
        """
        logging.getLogger('onnxruntime').setLevel(logging.ERROR)

        try:
            from insightface.app import FaceAnalysis
        except ImportError:
            raise ImportError(
                "insightface is not installed. "
                "Install it with: pip install insightface onnxruntime"
            )

        self.model_name = model_name
        self.det_size = det_size
        self.min_detection_confidence = min_detection_confidence

        # Initialize FaceAnalysis with detection + recognition
        self._app = FaceAnalysis(
            name=model_name,
            providers=['CPUExecutionProvider'],
            min_detection_confidence=self.min_detection_confidence,
        )
        self._app.prepare(ctx_id=-1, det_size=det_size)

    def embed(
        self,
        frame_rgb: np.ndarray,
    ) -> EmbeddingResult:
        """Generate embedding vector using InsightFace.

        Args:
            frame_rgb: Full frame image in RGB format

        Returns:
            EmbeddingResult with 512-dimensional L2-normalized embedding or None if no face detected

        Note: InsightFace performs detection, alignment, and embedding in one pipeline.
              This is optimal as the model was trained with this specific alignment.
        """
        # InsightFace expects BGR format
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        faces = self._app.get(frame_bgr)

        if not faces or len(faces) == 0:
            return EmbeddingResult.no_face()

        face = self._select_largest_face(faces)
        embedding = face.embedding

        normalized_embedding = self._l2_normalize_embeddings(embedding)

        return EmbeddingResult.success(
            normalized_embedding,
            self._to_bounding_box(face.bbox),
        )

    def _to_bounding_box(self, bbox) -> BoundingBox:
        """Convert an InsightFace (x1, y1, x2, y2) box into a BoundingBox."""
        x1, y1, x2, y2 = bbox.astype(int)
        return BoundingBox(x=int(x1), y=int(y1), width=int(x2 - x1), height=int(y2 - y1))

    def _select_largest_face(self, faces):
        return max(faces, key=self._calculate_bbox_area)

    def _calculate_bbox_area(self, face):
        bbox = face.bbox
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        return width * height

    def _l2_normalize_embeddings(self, embedding):
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding

    def get_config(self) -> Dict[str, Any]:
        """Return a JSON-serializable description of this embedder configuration."""
        return {
            "backend": "insightface",
            "config": {
                "model_name": self.model_name,
                "det_size": list(self.det_size),
                "min_detection_confidence": self.min_detection_confidence,
            },
        }

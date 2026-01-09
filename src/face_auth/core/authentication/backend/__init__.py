"""Embedder backend implementations."""
from face_auth.core.authentication.backend.impl.facenet_backend import FaceNetBackend
from face_auth.core.authentication.backend.impl.insightface_backend import InsightFaceBackend

__all__ = ['FaceNetBackend', 'InsightFaceBackend']
"""Embedder backend implementations."""
from face_auth.authentication.embedder.backend.impl.insightface_backend import InsightFaceBackend
from face_auth.authentication.embedder.backend.impl.facenet_backend import FaceNetBackend

__all__ = ['FaceNetBackend', 'InsightFaceBackend']


"""Frame processing for face authentication."""
import numpy as np
import cv2
import time
from pathlib import Path
from typing import Any, Dict, Optional, Protocol

from face_auth.authentication.core.models import (
    AuthenticationResult,
    AuthenticationStatus,
    FrameAuthenticationInput,
)
from face_auth.authentication.core.backend.authenticator_backend import AuthenticatorBackend
from face_auth.authentication.embedder.embedder import Embedder
from face_auth.authentication.embedder.models import EmbeddingResult


class EmbeddingCache(Protocol):
    """Interface for source-frame embedding cache implementations."""

    def get_or_compute(
        self,
        video_path: Path,
        source_frame_index: int,
        frame_rgb: np.ndarray,
        source_metadata: Optional[Dict[str, Any]] = None,
    ) -> EmbeddingResult:
        """Return a cached embedding or compute it for a source frame."""
        ...


class FrameAuthenticator:
    """Processes individual frames for face authentication."""

    def __init__(
        self,
        embedder: Embedder,
        authenticator: AuthenticatorBackend,
        fps: int,
        use_wall_clock_time: bool = False,
        embedding_cache: Optional[EmbeddingCache] = None
    ):
        """Initialize frame processor.

        Args:
            embedder: Embedding generator instance
            authenticator: Authenticator backend instance
            fps: Frames per second of the video
            use_wall_clock_time: If True, use actual elapsed wall-clock time instead of frame-index-based timestamps
        """
        self._embedder = embedder
        self._authenticator = authenticator
        self._fps = fps
        self._use_wall_clock_time = use_wall_clock_time
        self._start_time: Optional[float] = None
        self._embedding_cache = embedding_cache

    def authenticate(self, frame: FrameAuthenticationInput) -> AuthenticationResult:
        """Process frame and return authentication result.

        Args:
            frame: Frame image and metadata in the composed timeline

        Returns:
            Authentication result for the frame
        """
        frame_rgb = cv2.cvtColor(frame.image_bgr, cv2.COLOR_BGR2RGB)

        if self._use_wall_clock_time:
            if self._start_time is None:
                self._start_time = time.time()
            timestamp_ms = (time.time() - self._start_time) * 1000
        else:
            timestamp_ms = (frame.frame_index / self._fps) * 1000

        if (
            self._embedding_cache is not None
            and frame.source is not None
        ):
            embedding_result = self._embedding_cache.get_or_compute(
                video_path=frame.source.video_path,
                source_frame_index=frame.source.frame_index,
                frame_rgb=frame_rgb,
                source_metadata=frame.source.metadata,
            )
        else:
            embedding_result = self._embedder.get_embedding(frame_rgb)

        if not embedding_result.face_detected:
            return self._create_result_for_no_face(timestamp_ms)

        return self._create_result_for_detected_face(embedding_result.embedding, timestamp_ms)

    def _create_result_for_no_face(self, timestamp_ms: float) -> AuthenticationResult:
        """Create authentication result when no face was detected.

        Args:
            timestamp_ms: Video timestamp in milliseconds

        Returns:
            AuthenticationResult with authentication status
        """
        self._authenticator.update_with_no_face(timestamp_ms)

        return AuthenticationResult(
            status=self._get_authentication_status(),
            similarity=None,
            trust=self._authenticator.get_score(),
            face_detected=False,
            bounding_box=None
        )

    def _create_result_for_detected_face(self, embedding: np.ndarray, timestamp_ms: float) -> AuthenticationResult:
        """Create authentication result when face was detected.

        Args:
            embedding: Face embedding vector
            timestamp_ms: Video timestamp in milliseconds

        Returns:
            AuthenticationResult with authentication status and trust score
        """
        self._authenticator.update_with_embedding(embedding, timestamp_ms)

        return AuthenticationResult(
            status=self._get_authentication_status(),
            similarity=self._authenticator.get_last_similarity(),
            trust=self._authenticator.get_score(),
            face_detected=True,
            bounding_box=None
        )

    def _get_authentication_status(self) -> AuthenticationStatus:
        """Get current authentication state from authenticator.

        Returns:
            Current authentication state
        """
        return (
            AuthenticationStatus.UNLOCKED
            if self._authenticator.is_authenticated()
            else AuthenticationStatus.LOCKED
        )

    @property
    def authenticator(self) -> AuthenticatorBackend:
        """Get the authenticator backend instance."""
        return self._authenticator

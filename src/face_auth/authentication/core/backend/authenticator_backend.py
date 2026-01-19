"""Abstract base class for authenticator backends."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
import numpy as np


@dataclass
class AuthenticationState:
    """Base class for internal state that can be persisted/restored."""
    pass


class AuthenticatorBackend(ABC):
    """Abstract interface for authentication backends."""

    @abstractmethod
    def update_with_embedding(self, embedding: np.ndarray, timestamp: datetime) -> None:
        """Update internal state with face embedding.

        Args:
            embedding: Face embedding vector
            timestamp: Time when measurement was taken
        """
        pass

    @abstractmethod
    def update_with_no_face(self, timestamp: datetime) -> None:
        """Update internal state when no face was detected.

        Args:
            timestamp: Time when measurement was taken
        """
        pass

    @abstractmethod
    def is_authenticated(self) -> bool:
        """Check if currently authenticated based on internal state."""
        pass

    @abstractmethod
    def get_score(self) -> float:
        """Get current authentication score."""
        pass

    @abstractmethod
    def get_last_distance(self) -> float:
        """Get the last computed distance value."""
        pass

    @abstractmethod
    def get_state(self) -> AuthenticationState:
        """Get current state for caching."""
        pass

    @abstractmethod
    def restore_state(self, state: AuthenticationState) -> None:
        """Restore from cached state.

        Args:
            state: Previously saved state
        """
        pass

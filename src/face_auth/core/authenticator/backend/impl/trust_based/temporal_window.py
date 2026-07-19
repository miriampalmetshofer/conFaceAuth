"""Temporal sliding window for storing sequential data."""
from collections import deque
from typing import Generic, TypeVar, Sequence

T = TypeVar('T')


class TemporalWindow(Generic[T]):
    """Fixed-size sliding window for temporal data."""

    def __init__(self, window_size: int):
        """Initialize temporal window.

        Args:
            window_size: Maximum number of items to store

        Raises:
            ValueError: If window_size is not positive
        """
        if window_size <= 0:
            raise ValueError(f"Window size must be positive, got {window_size}")
        self._window: deque[T] = deque(maxlen=window_size)

    def append(self, value: T) -> None:
        """Add value to window.

        When window is full, oldest value is automatically removed.

        Args:
            value: Value to add to window
        """
        self._window.append(value)

    def get_values(self) -> list[T]:
        """Get all values in window from oldest to newest.

        Returns:
            List of values currently in window
        """
        return list(self._window)

    def is_empty(self) -> bool:
        """Check if window contains no values.

        Returns:
            True if window is empty, False otherwise
        """
        return len(self._window) == 0

    @property
    def size(self) -> int:
        """Get current number of values in window.

        Returns:
            Number of values currently stored (may be less than max window_size)
        """
        return len(self._window)

"""trust score computation from temporal similarity data."""
from typing import Sequence
import numpy as np


class TrustScorer:
    """Computes trsut score from temporal window using exponential weighting."""

    def __init__(self, alpha: float):
        """Initialize trust scorer.

        Args:
            alpha: Exponential decay parameter. Higher values give more weight
                   to recent observations.
        """
        self.alpha = alpha

    def compute_trust_score(self, similarity_window: Sequence[float]) -> float:
        """Compute exponentially-weighted average of similarities.



        Args:
            similarity_window: Sequence of similarity values (oldest to newest)

        Returns:
            Weighted trust score

        Raises:
            ValueError: If similarity_window is empty
        """


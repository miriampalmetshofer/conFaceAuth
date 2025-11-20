"""Risk score computation from temporal distance data."""
from typing import Sequence
import numpy as np


class RiskScorer:
    """Computes risk score from temporal window using exponential weighting."""

    def __init__(self, alpha: float):
        """Initialize risk scorer.

        Args:
            alpha: Exponential decay parameter. Higher values give more weight
                   to recent observations.
        """
        self.alpha = alpha

    def compute_risk_score(self, distance_window: Sequence[float]) -> float:
        """Compute exponentially-weighted average of distances.

        More recent distances receive higher weight based on alpha parameter.

        Args:
            distance_window: Sequence of distance values (oldest to newest)

        Returns:
            Weighted risk score

        Raises:
            ValueError: If distance_window is empty
        """
        if not distance_window:
            raise ValueError("Cannot compute risk score from empty window")

        weights = np.exp(np.linspace(-self.alpha, 0, len(distance_window)))
        return float(np.average(distance_window, weights=weights))

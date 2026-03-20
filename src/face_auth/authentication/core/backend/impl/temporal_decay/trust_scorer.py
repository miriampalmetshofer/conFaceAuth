"""Trust score calculation for temporal decay authentication."""
from face_auth.authentication.core.backend.impl.temporal_decay.time_functions import (
    weight_function,
    decay_function
)


class TrustScorer:
    """Computes trust scores using time-weighted observations."""

    def __init__(self, k_weight: float, k_decay: float):
        """Initialize trust scorer.

        Args:
            k_weight: Weight function parameter (higher = more weight to old trust score)
            k_decay: Decay function parameter (higher = slower decay)
        """
        self.k_weight = k_weight
        self.k_decay = k_decay

    def compute_with_face(
        self,
        previous_trust: float,
        new_similarity: float,
        delta_t: float
    ) -> float:
        """Calculate new trust score when face is detected.

        Xt = (Xt-1 · fwei) + (Zt · (1 - fwei))

        Args:
            previous_trust: Previous trust score (Xt-1)
            new_similarity: New observation similarity value (Zt)
            delta_t: Elapsed time since last observation (milliseconds)

        Returns:
            Updated trust score
        """
        weight = weight_function(delta_t, self.k_weight)
        return (previous_trust * weight) + (new_similarity * (1 - weight))

    def compute_with_no_face(
        self,
        previous_trust: float,
        delta_t: float
    ) -> float:
        """Calculate new trust score when no face detected.

        Xt = Xt-1 · fdec

        Args:
            previous_trust: Previous trust score (Xt-1)
            delta_t: Elapsed time since last observation (milliseconds)

        Returns:
            Decayed trust score
        """
        decay = decay_function(delta_t, self.k_decay)
        return previous_trust * decay

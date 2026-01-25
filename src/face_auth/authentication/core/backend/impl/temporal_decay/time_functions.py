"""Time-based exponential functions for weighting and decay."""
import numpy as np


def weight_function(delta_t: float, k: float) -> float:
    """Calculate weight for new observation based on elapsed time.

    fwei(Δt) = e^(-Δt/k)
    Higher k means old confidence has more impact vs new observation.

    Args:
        delta_t: Elapsed time between consecutive observations (milliseconds)
        k: Strictness parameter controlling function slope

    Returns:
        Weight value between 0 and 1
    """
    return float(np.exp(-delta_t / k))


def decay_function(delta_t: float, k: float) -> float:
    """Calculate time decay multiplier when no face detected.

    fdec(Δt) = e^(-Δt/k)

    Args:
        delta_t: Elapsed time since last observation (milliseconds)
        k: Decay rate parameter

    Returns:
        Decay multiplier between 0 and 1
    """
    return float(np.exp(-delta_t / k))

"""Shared core models."""
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class Frame:
    """RGB image observed at a specific timestamp."""
    image: np.ndarray
    timestamp_ms: float
    source_name: Optional[str] = None

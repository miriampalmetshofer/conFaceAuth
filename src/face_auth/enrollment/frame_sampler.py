"""Frame sampling strategies for enrollment."""
import numpy as np

from face_auth.config.logging_config import get_logger

logger = get_logger(__name__)


class NormalDistributionSampler:
    """Samples frames using normal distribution around sequence center."""

    def __init__(
        self,
        mean_fraction: float,
        stddev_fraction: float,
        random_seed: int
    ):
        """Initialize normal distribution sampler.

        Args:
            mean_fraction: Position of distribution mean as fraction of sequence length (0.5 = center)
            stddev_fraction: Standard deviation as fraction of sequence length
            random_seed: Random seed for reproducible sampling
        """
        self.mean_fraction = mean_fraction
        self.stddev_fraction = stddev_fraction
        self.random_seed = random_seed

    def sample(
        self,
        frames: list[np.ndarray],
        n_samples: int
    ) -> list[np.ndarray]:
        """Sample frames from list using normal distribution.

        Samples are drawn from a normal distribution centered around the middle
        of the sequence, favoring frames near the center.

        Args:
            frames: List of frames to sample from
            n_samples: Number of frames to sample

        Returns:
            List of sampled frames
        """
        count = len(frames)

        if count == 0:
            logger.warning("Cannot sample from empty frame list")
            return []

        if count < n_samples:
            logger.warning(
                f"Requested {n_samples} samples but only {count} frames available. "
                f"Returning all {count} frames."
            )
            return frames

        # Calculate distribution parameters
        mean = count * self.mean_fraction
        stddev = count * self.stddev_fraction

        # Set random seed for reproducibility
        np.random.seed(self.random_seed)

        # Sample indices from normal distribution, clipped to valid range
        indices = np.clip(
            np.random.normal(loc=mean, scale=stddev, size=n_samples).astype(int),
            0,
            count - 1
        )

        sampled_frames = [frames[i] for i in indices]
        logger.debug(f"Sampled {len(sampled_frames)} frames from {count} total frames")

        return sampled_frames

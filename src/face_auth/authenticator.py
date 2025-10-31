from collections import deque
import numpy as np
from face_auth.logging_config import get_logger

logger = get_logger(__name__)


class Authenticator:
    def __init__(self, enrollment_embeddings, similarity_computation: str, window_size: int, threshold: float, alpha: float) -> None:
        self.enrollment_embeddings = enrollment_embeddings
        self.distance_window = deque(maxlen=window_size)
        self.threshold = threshold
        self.similarity_computation = similarity_computation
        self.risk_score = None
        self.alpha = alpha

    def is_authenticated(self) -> bool:
        logger.debug(f"risk_score: {self.risk_score}, threshold: {self.threshold}")
        return self.risk_score <= self.threshold

    def compute_distance_between_embedding_and_enrollment(self, embedding):
        if isinstance(self.similarity_computation, float) and 0 < self.similarity_computation <= 1.0:
            distance = self._get_average_of_closest_percent(embedding, self.similarity_computation)
        else:
            raise ValueError(f"Unsupported similarity config: {self.similarity_computation}")
        return distance

    def append_distance_to_window_and_update_risk_score(self, distance) -> None:
        self.distance_window.append(distance)
        self._update_risk_score()

    def _update_risk_score(self):
        """Use exponentially decaying weights to compute the weighted average of the distances in the window."""
        alpha = self.alpha
        weights = np.exp(np.linspace(-alpha, 0, len(self.distance_window)))
        self.risk_score = np.average(self.distance_window, weights=weights)
        logger.debug(f"Updated risk_score: {self.risk_score}")

    def _get_average_of_closest_percent(self, embedding: np.ndarray, percent: float):
        """ Compute the average distance of the closest n percent of distances."""
        distances = self._compute_distance_to_enrollment_images(embedding)
        logger.debug(f"Distances to enrollment embeddings: {distances}")
        num_to_select = max(1, int(len(distances) * percent))  # at least 1
        closest_distances = sorted(distances)[:num_to_select]
        average_distance = np.mean(closest_distances)
        return average_distance

    def _compute_distance_to_enrollment_images(self, embedding) -> list[float]:
        """ Compute the distance between the given embedding and all enrollment embeddings using Euclidean distance."""
        distances = []

        for idx, enrollment_embedding in enumerate(self.enrollment_embeddings):
            distance = np.linalg.norm(embedding - enrollment_embedding)
            distances.append(distance)
        distances.sort()

        return distances

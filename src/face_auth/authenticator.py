from collections import deque
import numpy as np


class Authenticator:
    def __init__(self, enrollment_embeddings, similarity_computation: str, window_size: int, threshold: float):
        self.enrollment_embeddings = enrollment_embeddings
        self.distance_window = deque(maxlen=window_size)
        self.threshold = threshold
        self.similarity_computation = similarity_computation
        self.trust_score = None

    def compute_distance_between_embedding_and_enrolment(self, embedding):
        if isinstance(self.similarity_computation, float) and 0 < self.similarity_computation <= 1.0:
            distance = self._get_average_of_closest_percent(embedding, self.similarity_computation)
        else:
            raise ValueError(f"Unsupported similarity config: {self.similarity_computation}")
        return distance


    def is_authenticated(self) -> bool:
        return self.trust_score <= self.threshold


    def append_distance_to_window_and_update_trust_score(self, distance) -> None:
        self.distance_window.append(distance)
        self._update_trust_score()


    def _update_trust_score(self):
        alpha = 1
        if not self.distance_window:
            return self.threshold # start from threshold
        weights = np.exp(np.linspace(-alpha, 0, len(self.distance_window)))
        self.trust_score = np.average(self.distance_window, weights=weights)


    def _compute_distance_to_enrollment_images(self, embedding) -> list[float]:
        distances = []

        for idx, enrollment_embedding in enumerate(self.enrollment_embeddings):
            distance = np.linalg.norm(embedding - enrollment_embedding)
            distances.append(distance)
        distances.sort()

        return distances

    def _get_average_of_closest_percent(self, embedding: np.ndarray, percent: float):
        distances = self._compute_distance_to_enrollment_images(embedding)
        num_to_select = max(1, int(len(distances) * percent))  # at least 1
        closest_distances = sorted(distances)[:num_to_select]
        average_distance = np.mean(closest_distances)
        return average_distance
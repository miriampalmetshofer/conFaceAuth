"""Similarity computation between face embeddings."""
import numpy as np


class SimilarityCalculator:
    """Computes similarity between face embeddings using Euclidean distance."""

    def compute_distance(
        self,
        embedding: np.ndarray,
        reference_embedding: np.ndarray
    ) -> float:
        """Compute Euclidean distance between two embeddings.

        Args:
            embedding: Query embedding vector
            reference_embedding: Reference embedding vector to compare against

        Returns:
            Euclidean distance (lower = more similar)
        """
        return float(np.linalg.norm(embedding - reference_embedding))

    def compute_distances_to_all(
        self,
        embedding: np.ndarray,
        reference_embeddings: list[np.ndarray]
    ) -> list[float]:
        """Compute distances from embedding to all reference embeddings.

        Args:
            embedding: Query embedding vector
            reference_embeddings: List of reference embeddings

        Returns:
            List of distances in same order as reference_embeddings
        """
        return [
            self.compute_distance(embedding, ref_embedding)
            for ref_embedding in reference_embeddings
        ]

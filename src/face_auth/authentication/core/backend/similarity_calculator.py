"""Similarity computation between face embeddings."""
import numpy as np


class SimilarityCalculator:
    """Computes similarity between face embeddings using cosine similarity."""

    def compute_similarity(
        self,
        embedding: np.ndarray,
        reference_embedding: np.ndarray
    ) -> float:
        """Compute cosine similarity between two embeddings.

        Args:
            embedding: Query embedding vector
            reference_embedding: Reference embedding vector to compare against

        Returns:
            Cosine similarity in range [-1, 1] (higher = more similar)
        """
        return float(np.dot(embedding, reference_embedding))

    def compute_similarities_to_all(
        self,
        embedding: np.ndarray,
        reference_embeddings: list[np.ndarray]
    ) -> list[float]:
        """Compute similarities from embedding to all reference embeddings.

        Args:
            embedding: Query embedding vector
            reference_embeddings: List of reference embeddings

        Returns:
            List of similarities in same order as reference_embeddings
        """
        return [
            self.compute_similarity(embedding, ref_embedding)
            for ref_embedding in reference_embeddings
        ]

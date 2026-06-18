"""Persistent source-frame embedding cache."""
import hashlib
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Optional, Protocol

import numpy as np

from face_auth.authentication.embedder.models import EmbeddingResult
from face_auth.config.logging_config import get_logger

logger = get_logger(__name__)


class Embedder(Protocol):
    """Interface for embedder implementations used by the cache."""

    def get_embedding(self, frame_rgb: np.ndarray) -> EmbeddingResult:
        """Generate an embedding result for an RGB frame."""
        ...


class EmbeddingCache:
    """Caches expensive face embeddings by source video frame."""

    SCHEMA_VERSION = 1
    LOCK_STALE_SECONDS = 300

    def __init__(
        self,
        cache_dir: Path,
        embedder: Embedder,
        embedder_model: str,
        embedder_config: Dict[str, Any],
    ):
        """Initialize the embedding cache.

        Args:
            cache_dir: Directory where cache files are stored
            embedder: Embedder used for cache misses
            embedder_model: Configured embedder model name
            embedder_config: Configured embedder model parameters
        """
        self.cache_dir = cache_dir
        self.embedder = embedder
        self.embedder_model = embedder_model
        self.embedder_config = self._normalize_embedder_config(
            embedder_model,
            embedder_config or {},
        )
        self._reported_miss_paths: set[Path] = set()
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_or_compute(
        self,
        video_path: Path,
        source_frame_index: int,
        frame_rgb: np.ndarray,
        source_metadata: Optional[Dict[str, Any]] = None,
    ) -> EmbeddingResult:
        """Return cached embedding for a source frame or compute and persist it."""
        manifest = self._build_manifest(video_path, source_metadata)
        cache_path = self._cache_path(video_path, manifest)

        with self._file_lock(cache_path):
            records = self._load_records(cache_path)
            cached = records.get(source_frame_index)
            if cached is not None:
                logger.debug(
                    f"Embedding cache hit for {video_path.name} frame {source_frame_index}"
                )
                return cached

        if cache_path not in self._reported_miss_paths:
            logger.info(
                f"Embedding cache miss for {video_path.name}; "
                "computing missing source frames"
            )
            self._reported_miss_paths.add(cache_path)

        computed = self.embedder.get_embedding(frame_rgb)

        with self._file_lock(cache_path):
            records = self._load_records(cache_path)
            # Another worker may have written the same frame while this worker computed it.
            cached = records.get(source_frame_index)
            if cached is not None:
                return cached
            records[source_frame_index] = computed
            self._save_records(cache_path, manifest, records)

        return computed

    def _cache_path(
        self,
        video_path: Path,
        manifest: Dict[str, Any],
    ) -> Path:
        cache_key = self._fingerprint(manifest)
        return self.cache_dir / f"{video_path.stem}_{cache_key}.npz"

    def _build_manifest(
        self,
        video_path: Path,
        source_metadata: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        stat = video_path.stat()
        return {
            "cache_schema_version": self.SCHEMA_VERSION,
            "cache_type": "source_frame_embeddings",
            "video_path": str(video_path.resolve()),
            "video_size_bytes": stat.st_size,
            "video_mtime_ns": stat.st_mtime_ns,
            "source_metadata": source_metadata or {},
            "embedder": {
                "model": self.embedder_model,
                "config": self.embedder_config,
            },
        }

    def _fingerprint(self, payload: Dict[str, Any]) -> str:
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]

    def _load_records(self, cache_path: Path) -> Dict[int, EmbeddingResult]:
        if not cache_path.exists():
            return {}

        try:
            with np.load(cache_path, allow_pickle=False) as data:
                source_frame_indices = data["source_frame_indices"].astype(np.int64)
                face_detected = data["face_detected"].astype(bool)
                embeddings = data["embeddings"]
                records = {}

                for row_index, source_frame_index in enumerate(source_frame_indices):
                    if face_detected[row_index]:
                        records[int(source_frame_index)] = EmbeddingResult.success(
                            embeddings[row_index].copy()
                        )
                    else:
                        records[int(source_frame_index)] = EmbeddingResult.no_face()

                return records
        except Exception as exc:
            logger.warning(f"Ignoring unreadable embedding cache {cache_path}: {exc}")
            return {}

    def _save_records(
        self,
        cache_path: Path,
        manifest: Dict[str, Any],
        records: Dict[int, EmbeddingResult],
    ) -> None:
        frame_indices = sorted(records)
        detected_embeddings = [
            records[index].embedding
            for index in frame_indices
            if records[index].face_detected and records[index].embedding is not None
        ]
        embedding_dim = len(detected_embeddings[0]) if detected_embeddings else 0

        source_frame_indices = np.array(frame_indices, dtype=np.int64)
        face_detected = np.array(
            [records[index].face_detected for index in frame_indices],
            dtype=bool,
        )
        embeddings = np.zeros((len(frame_indices), embedding_dim), dtype=np.float32)

        if embedding_dim > 0:
            for row_index, source_frame_index in enumerate(frame_indices):
                result = records[source_frame_index]
                if result.face_detected and result.embedding is not None:
                    embeddings[row_index] = result.embedding.astype(np.float32)

        temp_path = cache_path.with_suffix(f".{os.getpid()}.tmp.npz")
        np.savez_compressed(
            temp_path,
            manifest_json=np.array(json.dumps(manifest, sort_keys=True)),
            source_frame_indices=source_frame_indices,
            face_detected=face_detected,
            embeddings=embeddings,
        )
        os.replace(temp_path, cache_path)

    @contextmanager
    def _file_lock(self, cache_path: Path):
        lock_path = cache_path.with_suffix(cache_path.suffix + ".lock")
        lock_fd = None
        try:
            while True:
                try:
                    lock_fd = os.open(
                        lock_path,
                        os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                    )
                    break
                except FileExistsError:
                    if self._is_stale_lock(lock_path):
                        logger.warning(f"Removing stale embedding cache lock {lock_path}")
                        try:
                            os.unlink(lock_path)
                        except FileNotFoundError:
                            pass
                        continue
                    time.sleep(0.05)
            yield
        finally:
            if lock_fd is not None:
                os.close(lock_fd)
                try:
                    os.unlink(lock_path)
                except FileNotFoundError:
                    pass

    def _is_stale_lock(self, lock_path: Path) -> bool:
        try:
            return (time.time() - lock_path.stat().st_mtime) > self.LOCK_STALE_SECONDS
        except FileNotFoundError:
            return False

    def _normalize_embedder_config(
        self,
        embedder_model: str,
        embedder_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        if embedder_model == "insightface":
            return {
                "model_name": embedder_config.get("model_name", "buffalo_sc"),
                "det_size": embedder_config.get("det_size", [640, 640]),
                "min_detection_confidence": embedder_config.get(
                    "min_detection_confidence",
                    0.5,
                ),
            }
        if embedder_model == "facenet":
            return {
                "detector": embedder_config.get("detector", "mediapipe"),
                "target_size": embedder_config.get("target_size", [160, 160]),
            }
        return dict(embedder_config)

"""Face detection and recognition for CAPTCHA using InsightFace."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import warnings

import cv2
import numpy as np

warnings.filterwarnings(
    "ignore",
    message="`estimate` is deprecated*",
    category=FutureWarning,
    module="insightface.utils.face_align",
)

try:
    from insightface.app import FaceAnalysis
except ImportError:
    FaceAnalysis = None

from config import AppConfig
from utils import iter_image_files


class RecognitionError(RuntimeError):
    """Raised when face recognition cannot be completed."""


@dataclass(frozen=True)
class KnownFace:
    """A reference face embedding loaded from known_faces."""

    filename: str
    embedding: np.ndarray


@dataclass(frozen=True)
class RecognitionResult:
    """The result of comparing a captured face to known faces."""

    authorized: bool
    confidence: float
    best_match_filename: str | None


class FaceRecognizer:
    """Load known faces and verify captured images against them."""

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        known_face_paths = iter_image_files(config.known_faces_dir)
        if not known_face_paths:
            raise RecognitionError(f"No known face images found in {config.known_faces_dir}.")

        if FaceAnalysis is None:
            raise RecognitionError(
                "InsightFace is not installed. Install dependencies from requirements.txt."
            )

        self._app = FaceAnalysis(
            name=config.insightface_model_name,
            providers=[config.insightface_provider],
        )
        self._app.prepare(ctx_id=0, det_size=config.detection_size)
        self._known_faces = self._load_known_faces(known_face_paths)

    def verify(self, image_path: Path) -> RecognitionResult:
        """Compare a captured image against all loaded known faces."""
        if not self._known_faces:
            raise RecognitionError("No known face images were loaded.")

        captured_embedding = self._extract_single_embedding(image_path)
        scores = [
            (self._cosine_similarity(captured_embedding, known_face.embedding), known_face)
            for known_face in self._known_faces
        ]
        best_score, best_face = max(scores, key=lambda item: item[0])
        confidence = self._similarity_to_confidence(best_score)

        return RecognitionResult(
            authorized=best_score >= self._config.recognition_threshold,
            confidence=confidence,
            best_match_filename=best_face.filename,
        )

    def _load_known_faces(self, image_paths: list[Path]) -> list[KnownFace]:
        """Load and embed every supported image inside known_faces."""
        cached_faces = self._load_cached_known_faces(image_paths)
        if cached_faces is not None:
            return cached_faces

        known_faces: list[KnownFace] = []

        for image_path in image_paths:
            try:
                embedding = self._extract_single_embedding(image_path)
            except RecognitionError as exc:
                raise RecognitionError(f"{image_path.name}: {exc}") from exc

            known_faces.append(KnownFace(filename=image_path.name, embedding=embedding))

        self._save_known_face_cache(image_paths, known_faces)
        return known_faces

    def _load_cached_known_faces(self, image_paths: list[Path]) -> list[KnownFace] | None:
        """Return cached known faces when source images are unchanged."""
        cache_path = self._config.known_face_cache_path
        if not cache_path.exists():
            return None

        try:
            with np.load(cache_path, allow_pickle=False) as cache:
                cache_key = str(cache["cache_key"][0])
                cached_signatures = cache["signatures"].astype(str).tolist()
                filenames = cache["filenames"].astype(str).tolist()
                embeddings = cache["embeddings"].astype(np.float32)
        except (OSError, KeyError, ValueError):
            return None

        current_signatures = [self._image_signature(path) for path in image_paths]
        if (
            cache_key != self._cache_key()
            or cached_signatures != current_signatures
            or len(filenames) != len(embeddings)
        ):
            return None

        return [
            KnownFace(filename=filename, embedding=embedding)
            for filename, embedding in zip(filenames, embeddings, strict=True)
        ]

    def _save_known_face_cache(
        self,
        image_paths: list[Path],
        known_faces: list[KnownFace],
    ) -> None:
        """Persist known face embeddings for faster future startup."""
        self._config.data_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            self._config.known_face_cache_path,
            cache_key=np.array([self._cache_key()], dtype=str),
            signatures=np.array(
                [self._image_signature(path) for path in image_paths],
                dtype=str,
            ),
            filenames=np.array([face.filename for face in known_faces], dtype=str),
            embeddings=np.vstack([face.embedding for face in known_faces]),
        )

    @staticmethod
    def _image_signature(image_path: Path) -> str:
        """Return a cache signature for an image file."""
        stat = image_path.stat()
        return f"{image_path.name}:{stat.st_size}:{stat.st_mtime_ns}"

    def _cache_key(self) -> str:
        """Return the recognizer settings that affect generated embeddings."""
        detection_width, detection_height = self._config.detection_size
        return (
            f"{self._config.insightface_model_name}:"
            f"{detection_width}x{detection_height}:"
            f"{self._config.recognition_image_max_dimension}"
        )

    def _extract_single_embedding(self, image_path: Path) -> np.ndarray:
        """Return the embedding for the single largest face in an image."""
        image = cv2.imread(str(image_path))
        if image is None:
            raise RecognitionError(f"Could not read image: {image_path}")

        image = self._resize_for_recognition(image)
        faces = self._app.get(image)
        if not faces:
            raise RecognitionError(f"No face detected in image: {image_path}")

        face = max(faces, key=lambda item: self._face_area(item.bbox))
        embedding = getattr(face, "normed_embedding", None)
        if embedding is None:
            raise RecognitionError(f"Could not generate face embedding: {image_path}")

        return np.asarray(embedding, dtype=np.float32)

    def _resize_for_recognition(self, image: np.ndarray) -> np.ndarray:
        """Downscale large images before face analysis to reduce CPU work."""
        max_dimension = self._config.recognition_image_max_dimension
        height, width = image.shape[:2]
        largest_dimension = max(height, width)
        if largest_dimension <= max_dimension:
            return image

        scale = max_dimension / largest_dimension
        resized_width = max(1, int(width * scale))
        resized_height = max(1, int(height * scale))
        return cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_AREA)

    @staticmethod
    def _face_area(bbox: np.ndarray) -> float:
        """Return the area of an InsightFace bounding box."""
        left, top, right, bottom = bbox
        return max(0.0, float(right - left)) * max(0.0, float(bottom - top))

    @staticmethod
    def _cosine_similarity(first: np.ndarray, second: np.ndarray) -> float:
        """Return cosine similarity for two normalized embeddings."""
        return float(np.dot(first, second))

    @staticmethod
    def _similarity_to_confidence(similarity: float) -> float:
        """Convert cosine similarity from -1..1 into a 0..1 confidence score."""
        return max(0.0, min(1.0, (similarity + 1.0) / 2.0))

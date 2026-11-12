"""Application configuration for CAPTCHA."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path


def get_base_dir() -> Path:
    """Return the application base directory for source or packaged runs."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent

    return Path(__file__).resolve().parent


@dataclass(frozen=True)
class AppConfig:
    """Runtime settings for the current CAPTCHA milestone."""

    base_dir: Path = get_base_dir()
    camera_index: int = 0
    camera_warmup_frames: int = 30
    camera_warmup_delay_seconds: float = 0.03
    startup_delay_seconds: float = 2.0
    recognition_threshold: float = 0.45
    insightface_model_name: str = "buffalo_s"
    insightface_provider: str = "CPUExecutionProvider"
    detection_size: tuple[int, int] = (320, 320)
    recognition_image_max_dimension: int = 640

    @property
    def captures_dir(self) -> Path:
        """Directory where captured unlock photos are stored."""
        return self.base_dir / "captures"

    @property
    def known_faces_dir(self) -> Path:
        """Directory containing authorized reference face images."""
        return self.base_dir / "known_faces"

    @property
    def logs_dir(self) -> Path:
        """Directory reserved for application logs."""
        return self.base_dir / "logs"

    @property
    def data_dir(self) -> Path:
        """Directory reserved for generated application data."""
        return self.base_dir / "data"

    @property
    def known_face_cache_path(self) -> Path:
        """File used to cache known face embeddings."""
        return self.data_dir / "known_faces_cache.npz"

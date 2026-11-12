"""Small shared helpers for CAPTCHA."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def ensure_directory(path: Path) -> Path:
    """Create a directory if needed and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def timestamped_image_path(directory: Path, suffix: str = ".jpg") -> Path:
    """Build a timestamped image path inside a directory."""
    filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + suffix
    return directory / filename


def iter_image_files(directory: Path) -> list[Path]:
    """Return supported image files in stable filename order."""
    if not directory.exists():
        return []

    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def format_confidence(score: float) -> str:
    """Format a 0.0 to 1.0 score as a percentage."""
    bounded_score = max(0.0, min(1.0, score))
    return f"{bounded_score * 100:.1f}%"

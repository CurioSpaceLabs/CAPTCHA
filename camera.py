"""Webcam capture support for CAPTCHA."""

from __future__ import annotations

import time
from pathlib import Path

import cv2

from config import AppConfig
from utils import ensure_directory, timestamped_image_path


class CameraError(RuntimeError):
    """Raised when the webcam cannot capture an image."""


class Camera:
    """Capture still images from a configured webcam."""

    def __init__(self, config: AppConfig) -> None:
        self._config = config

    def capture_photo(self) -> Path:
        """Capture one photo from the webcam and return the saved image path."""
        output_dir = ensure_directory(self._config.captures_dir)
        output_path = timestamped_image_path(output_dir)

        camera = cv2.VideoCapture(self._config.camera_index)
        try:
            if not camera.isOpened():
                raise CameraError("Could not open webcam.")

            self._warm_up(camera)
            success, frame = camera.read()
            if not success:
                raise CameraError("Could not capture image.")

            if not cv2.imwrite(str(output_path), frame):
                raise CameraError(f"Could not save image to {output_path}.")

            return output_path
        finally:
            camera.release()

    def _warm_up(self, camera: cv2.VideoCapture) -> None:
        """Allow the webcam to settle exposure and white balance."""
        for _ in range(self._config.camera_warmup_frames):
            camera.read()
            time.sleep(self._config.camera_warmup_delay_seconds)

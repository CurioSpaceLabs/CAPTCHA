import sys
import time

from camera import Camera, CameraError
from config import AppConfig
from recognizer import FaceRecognizer, RecognitionError
from utils import ensure_directory, format_confidence
from warning import ask_for_retake


def prepare_directories(config: AppConfig) -> None:
    """Ensure the configured application folders exist."""
    ensure_directory(config.captures_dir)
    ensure_directory(config.known_faces_dir)
    ensure_directory(config.logs_dir)
    ensure_directory(config.data_dir)


def print_result(authorized: bool, confidence: float, match_filename: str | None) -> None:
    """Print a readable authorization result."""
    print("Access granted" if authorized else "Access denied")
    print(f"Confidence: {format_confidence(confidence)}")

    label = "Matched" if authorized else "Best match"
    print(f"{label}: {match_filename or 'None'}")


def main() -> int:
    """Run the current CAPTCHA face recognition milestone."""
    config = AppConfig()
    prepare_directories(config)
    time.sleep(config.startup_delay_seconds)

    try:
        camera = Camera(config)
        captured_path = camera.capture_photo()
        recognizer = FaceRecognizer(config)

        while True:
            result = recognizer.verify(captured_path)
            print_result(result.authorized, result.confidence, result.best_match_filename)

            if result.authorized:
                return 0

            should_retake = ask_for_retake(
                format_confidence(result.confidence),
                result.best_match_filename,
            )
            if not should_retake:
                return 1

            captured_path = camera.capture_photo()
    except (CameraError, RecognitionError) as exc:
        print(f"Error: {exc}")
        return 1
    except Exception as exc:
        print(f"Unexpected error: {exc}")
        return 1

if __name__ == "__main__":
    sys.exit(main())

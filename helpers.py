from datetime import datetime
from pathlib import Path

import cv2


def save_image_cv(cv_image, output_path: Path):
    """Saves an OpenCV image to the specified path."""
    try:
        cv2.imwrite(str(output_path), cv_image)
        print(f"Image saved to: {output_path}")
    except Exception as e:
        print(f"Error saving image to {output_path}: {e}")


def generate_output_filename(input_filename: str, suffix="_detected"):
    """Generates an output filename with a timestamp and suffix."""
    now = datetime.now()
    timestamp_str = now.strftime("%Y%m%d_%H%M%S")
    base, ext = Path(input_filename).stem, Path(input_filename).suffix
    return f"{base}{suffix}_{timestamp_str}{ext}"

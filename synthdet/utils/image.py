"""Image I/O helpers and augmentation variant grouping."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Image I/O
# ---------------------------------------------------------------------------

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def load_image(path: Path, mode: str = "bgr") -> np.ndarray:
    """Load an image from disk.

    Args:
        path: Path to the image file.
        mode: "bgr" (OpenCV default), "rgb", or "gray".

    Returns:
        Image as numpy array.
    """
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {path}")
    if mode == "rgb":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif mode == "gray":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif mode != "bgr":
        raise ValueError(f"Unknown mode: {mode!r}. Use 'bgr', 'rgb', or 'gray'.")
    return img


def get_image_dimensions(path: Path) -> tuple[int, int]:
    """Get (width, height) of an image without fully loading it (header-only)."""
    with Image.open(path) as img:
        return img.size  # (width, height)


def compute_image_stats(image: np.ndarray) -> dict[str, float]:
    """Compute basic image statistics (brightness, contrast, per-channel means).

    Args:
        image: BGR or grayscale numpy array.

    Returns:
        Dict with brightness, contrast, and per-channel mean values.
    """
    if image.ndim == 2:
        return {
            "brightness": float(np.mean(image)),
            "contrast": float(np.std(image)),
        }
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    stats: dict[str, float] = {
        "brightness": float(np.mean(gray)),
        "contrast": float(np.std(gray)),
    }
    for i, channel in enumerate(["blue", "green", "red"]):
        stats[f"{channel}_mean"] = float(np.mean(image[:, :, i]))
    return stats


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------


def find_image_files(directory: Path) -> list[Path]:
    """Find all image files in a directory (non-recursive), sorted by name."""
    if not directory.is_dir():
        return []
    return sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


# ---------------------------------------------------------------------------
# Perceptual hashing
# ---------------------------------------------------------------------------


def compute_perceptual_hash(image: np.ndarray, hash_size: int = 8) -> str:
    """Compute average perceptual hash of an image.

    Resizes to hash_size x hash_size, computes mean, produces binary hash.
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    resized = cv2.resize(gray, (hash_size, hash_size), interpolation=cv2.INTER_AREA)
    mean_val = np.mean(resized)
    bits = (resized > mean_val).flatten()
    # Pack bits into hex string
    hash_int = 0
    for bit in bits:
        hash_int = (hash_int << 1) | int(bit)
    return format(hash_int, f"0{hash_size * hash_size // 4}x")


# ---------------------------------------------------------------------------
# Augmentation variant grouping
# ---------------------------------------------------------------------------

# Roboflow naming: {source_name}.rf.{hash}.{ext}
_ROBOFLOW_PATTERN = re.compile(r"^(.+)\.rf\.[a-zA-Z0-9]+(\.\w+)$")


def group_augmentation_variants(image_paths: list[Path]) -> dict[str, list[Path]]:
    """Group Roboflow-augmented images by their source image name.

    Roboflow generates variants with naming: {source}.rf.{hash}.{ext}
    Images sharing the same {source} prefix are augmentation variants.

    Returns:
        Dict mapping source name to list of variant paths.
    """
    groups: dict[str, list[Path]] = {}
    for path in image_paths:
        match = _ROBOFLOW_PATTERN.match(path.name)
        if match:
            source_name = match.group(1)
        else:
            source_name = path.stem
        groups.setdefault(source_name, []).append(path)
    return groups

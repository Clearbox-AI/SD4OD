"""Write YOLO-format datasets (images + labels + data.yaml)."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import cv2
import yaml

from synthdet.types import BBox, ImageRecord

logger = logging.getLogger(__name__)


def write_yolo_label(label_path: Path, bboxes: list[BBox]) -> None:
    """Write a single YOLO label file.

    Args:
        label_path: Path to the output .txt file.
        bboxes: Bounding boxes to write. Empty list creates an empty file
                (negative example).
    """
    label_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [bbox.to_yolo_line() for bbox in bboxes]
    label_path.write_text("\n".join(lines) + "\n" if lines else "")


def write_yolo_split(
    records: list[ImageRecord],
    output_dir: Path,
    split: str,
    output_format: str = "jpg",
    output_quality: int = 95,
) -> None:
    """Write a dataset split (images + labels) from ImageRecords.

    Args:
        records: ImageRecord objects to write.
        output_dir: Root output directory.
        split: Split name ("train", "valid", "test").
        output_format: Image format ("jpg" or "png").
        output_quality: JPEG quality (1-100).
    """
    images_dir = output_dir / split / "images"
    labels_dir = output_dir / split / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    for i, record in enumerate(records):
        # Determine filename
        stem = record.image_path.stem
        ext = f".{output_format}"
        img_name = f"{stem}{ext}"
        img_path = images_dir / img_name
        label_path = labels_dir / f"{stem}.txt"

        # Handle name collisions
        if img_path.exists():
            stem = f"{stem}_{i:04d}"
            img_name = f"{stem}{ext}"
            img_path = images_dir / img_name
            label_path = labels_dir / f"{stem}.txt"

        # Write image
        if record.image is not None:
            if output_format == "jpg":
                cv2.imwrite(
                    str(img_path),
                    record.image,
                    [cv2.IMWRITE_JPEG_QUALITY, output_quality],
                )
            else:
                cv2.imwrite(str(img_path), record.image)
        elif record.image_path.is_file():
            shutil.copy2(record.image_path, img_path)
        else:
            logger.warning("No image data for %s, skipping", record.image_path)
            continue

        # Write label
        write_yolo_label(label_path, record.bboxes)

        # Update record path to point to new location
        record.image_path = img_path

    logger.info("Wrote %d images to %s/%s", len(records), output_dir, split)


def write_data_yaml(
    output_dir: Path,
    class_names: list[str],
    splits: list[str],
) -> Path:
    """Write a YOLO data.yaml file.

    Args:
        output_dir: Root output directory.
        class_names: List of class names.
        splits: List of split names present.

    Returns:
        Path to the written data.yaml file.
    """
    data: dict = {"nc": len(class_names), "names": class_names}

    for split in splits:
        key = "val" if split == "valid" else split
        data[key] = f"{split}/images"

    yaml_path = output_dir / "data.yaml"
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(yaml_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    logger.info("Wrote data.yaml to %s", yaml_path)
    return yaml_path


def write_yolo_dataset(
    records_by_split: dict[str, list[ImageRecord]],
    output_dir: Path,
    class_names: list[str],
    output_format: str = "jpg",
    output_quality: int = 95,
) -> Path:
    """Write a complete YOLO dataset (all splits + data.yaml).

    Args:
        records_by_split: Dict mapping split name to ImageRecords.
        output_dir: Root output directory.
        class_names: List of class names.
        output_format: Image format.
        output_quality: JPEG quality.

    Returns:
        Path to the written data.yaml file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for split, records in records_by_split.items():
        write_yolo_split(records, output_dir, split, output_format, output_quality)

    yaml_path = write_data_yaml(output_dir, class_names, list(records_by_split.keys()))
    return yaml_path

"""YOLO dataset parser — loads data.yaml and builds a Dataset object."""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

from synthdet.types import AnnotationSource, Dataset, ImageRecord
from synthdet.utils.bbox import parse_yolo_label_file
from synthdet.utils.image import IMAGE_EXTENSIONS

logger = logging.getLogger(__name__)


def parse_data_yaml(yaml_path: Path) -> dict:
    """Parse a YOLO data.yaml file and resolve relative paths.

    Roboflow data.yaml uses paths relative to the yaml file's parent directory
    (e.g. ``../train/images`` from ``data/data.yaml`` → ``data/../train/images``).
    """
    yaml_path = yaml_path.resolve()
    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    base_dir = yaml_path.parent

    # Resolve split paths — try relative to yaml parent first, then
    # handle Roboflow's '../train/images' convention where splits are
    # actually siblings of the yaml file, not of its parent.
    for key in ("train", "val", "test"):
        if key in data and data[key]:
            split_path = Path(data[key])
            if split_path.is_absolute():
                continue
            resolved = (base_dir / split_path).resolve()
            if resolved.is_dir():
                data[key] = str(resolved)
            else:
                # Fallback: strip leading '../' and resolve relative to yaml parent
                stripped = str(split_path)
                while stripped.startswith("../"):
                    stripped = stripped[3:]
                fallback = (base_dir / stripped).resolve()
                if fallback.is_dir():
                    logger.info("Resolved %s path via fallback: %s", key, fallback)
                    data[key] = str(fallback)
                else:
                    data[key] = str(resolved)

    return data


def load_split(
    images_dir: str | Path,
    source: AnnotationSource = AnnotationSource.human,
) -> tuple[list[ImageRecord], list[str]]:
    """Load a single dataset split (train/valid/test).

    Expects sibling ``labels/`` directory next to ``images/``.
    Returns (records, warnings).
    """
    images_dir = Path(images_dir)
    labels_dir = images_dir.parent / "labels"

    records: list[ImageRecord] = []
    warnings: list[str] = []

    if not images_dir.is_dir():
        warnings.append(f"Images directory not found: {images_dir}")
        return records, warnings

    image_files = sorted(
        p for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    for img_path in image_files:
        label_path = labels_dir / f"{img_path.stem}.txt"
        if label_path.is_file():
            try:
                bboxes = parse_yolo_label_file(label_path, source=source)
            except (ValueError, OSError) as e:
                warnings.append(f"Error parsing {label_path}: {e}")
                bboxes = []
        else:
            bboxes = []
            warnings.append(f"No label file for {img_path.name}")

        records.append(ImageRecord(
            image_path=img_path,
            bboxes=bboxes,
            metadata={"source": source.value, "label_path": str(label_path)},
        ))

    return records, warnings


def load_yolo_dataset(yaml_path: str | Path) -> Dataset:
    """Load a complete YOLO dataset from a data.yaml file.

    This is the main entry point for dataset loading.

    Args:
        yaml_path: Path to data.yaml.

    Returns:
        A fully populated Dataset object.
    """
    yaml_path = Path(yaml_path)
    data = parse_data_yaml(yaml_path)

    class_names = data.get("names", [])
    if isinstance(class_names, dict):
        # Handle {0: 'Scratch'} format
        max_id = max(class_names.keys())
        class_names = [class_names.get(i, f"class_{i}") for i in range(max_id + 1)]

    all_warnings: list[str] = []

    # Load splits
    train, w = load_split(data.get("train", ""), source=AnnotationSource.human)
    all_warnings.extend(w)

    val_key = "val" if "val" in data else "test"
    valid, w = load_split(data.get("val", data.get("valid", "")), source=AnnotationSource.human)
    all_warnings.extend(w)

    test, w = load_split(data.get("test", ""), source=AnnotationSource.human)
    all_warnings.extend(w)

    if all_warnings:
        for warning in all_warnings:
            logger.warning(warning)

    dataset = Dataset(
        root=yaml_path.parent.resolve(),
        class_names=class_names,
        train=train,
        valid=valid,
        test=test,
    )

    logger.info(
        "Loaded dataset: %d train, %d valid, %d test images (%d classes)",
        len(train), len(valid), len(test), len(class_names),
    )
    return dataset

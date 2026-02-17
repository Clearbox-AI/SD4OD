"""Shared test fixtures for SynthDet."""

from __future__ import annotations

from pathlib import Path

import pytest

from synthdet.types import AnnotationSource, BBox, Dataset, ImageRecord

FIXTURES_DIR = Path(__file__).parent / "fixtures"
MINI_DATASET_DIR = FIXTURES_DIR / "mini_dataset"
REAL_DATASET_YAML = Path(__file__).parent.parent / "data" / "data.yaml"


@pytest.fixture
def sample_bbox() -> BBox:
    """A single BBox for unit tests."""
    return BBox(
        class_id=0,
        x_center=0.5,
        y_center=0.5,
        width=0.1,
        height=0.1,
        source=AnnotationSource.human,
    )


@pytest.fixture
def sample_bboxes() -> list[BBox]:
    """A list of varied BBoxes covering different sizes and regions."""
    return [
        # Tiny bbox, top-left
        BBox(class_id=0, x_center=0.1, y_center=0.1, width=0.02, height=0.02,
             source=AnnotationSource.human),
        # Small bbox, middle-center
        BBox(class_id=0, x_center=0.5, y_center=0.5, width=0.1, height=0.1,
             source=AnnotationSource.human),
        # Medium bbox, bottom-right
        BBox(class_id=0, x_center=0.85, y_center=0.85, width=0.2, height=0.2,
             source=AnnotationSource.compositor),
        # Large bbox, top-right
        BBox(class_id=1, x_center=0.85, y_center=0.1, width=0.3, height=0.3,
             source=AnnotationSource.inpainting),
    ]


@pytest.fixture
def mini_dataset_path() -> Path:
    """Path to the mini dataset data.yaml."""
    return MINI_DATASET_DIR / "data.yaml"


@pytest.fixture
def mini_dataset(mini_dataset_path) -> Dataset:
    """Load the mini test dataset."""
    from synthdet.analysis.loader import load_yolo_dataset
    return load_yolo_dataset(mini_dataset_path)


@pytest.fixture
def real_dataset_path() -> Path:
    """Path to the real dataset data.yaml (may not exist in CI)."""
    return REAL_DATASET_YAML


def has_real_dataset() -> bool:
    return REAL_DATASET_YAML.is_file()

"""Tests for synthdet.generate.generative_compositor.

All tests are mock-based — no real API calls, GPU, or disk I/O required.
"""

from __future__ import annotations

import random
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from synthdet.generate.errors import InpaintingAPIError
from synthdet.generate.generative_compositor import (
    DEFAULT_DEFECT_PROMPTS,
    GenerativeDefectCompositor,
    _save_incremental,
)
from synthdet.types import (
    AnnotationSource,
    BBox,
    BBoxSizeBucket,
    GenerationTask,
    ImageRecord,
    SpatialRegion,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_provider(cost=0.04, patch_size=(64, 64)):
    """Create a mock provider that returns a synthetic defect patch + mask."""
    provider = MagicMock()
    provider.cost_per_image = cost

    def fake_generate(prompt, size=(512, 512), seed=None):
        h, w = size[1], size[0]
        patch_bgr = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
        mask = np.full((h, w), 255, dtype=np.uint8)
        return patch_bgr, mask

    provider.generate_defect_patch.side_effect = fake_generate
    return provider


def _make_task(
    task_id="task_001",
    num_images=2,
    target_classes=None,
    target_size_buckets=None,
    target_regions=None,
):
    """Create a GenerationTask for testing."""
    return GenerationTask(
        task_id=task_id,
        priority=0.8,
        num_images=num_images,
        target_classes=target_classes if target_classes is not None else [0],
        target_size_buckets=target_size_buckets or [BBoxSizeBucket.medium],
        target_regions=target_regions or [SpatialRegion.middle_center],
        suggested_prompts=["test prompt"],
        rationale="test task",
        method="compositor",
    )


def _make_backgrounds(n=3, shape=(640, 860, 3)):
    """Create fake background images."""
    return [np.random.randint(0, 255, shape, dtype=np.uint8) for _ in range(n)]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestGenerativeDefectCompositor:
    def test_basic_generation(self):
        """Generate images with a mock provider."""
        provider = _make_mock_provider()
        compositor = GenerativeDefectCompositor(
            provider,
            max_defects_per_image=1,
            requests_per_minute=1000.0,
            max_cost_usd=100.0,
        )

        task = _make_task(num_images=3)
        backgrounds = _make_backgrounds()

        records = compositor.generate(
            task,
            backgrounds=backgrounds,
            img_size=(860, 640),
            class_names=["scratch"],
            seed=42,
        )

        assert len(records) == 3
        for rec in records:
            assert isinstance(rec, ImageRecord)
            assert len(rec.bboxes) > 0
            assert rec.image is not None
            assert rec.image.shape == (640, 860, 3)
            assert rec.metadata["source"] == "generative_compositor"

    def test_bbox_annotations_correct(self):
        """Generated bboxes should have compositor source and valid coordinates."""
        provider = _make_mock_provider()
        compositor = GenerativeDefectCompositor(
            provider,
            max_defects_per_image=1,
            requests_per_minute=1000.0,
            max_cost_usd=100.0,
        )

        task = _make_task(num_images=1)
        records = compositor.generate(
            task,
            backgrounds=_make_backgrounds(1),
            img_size=(860, 640),
            class_names=["scratch"],
            seed=42,
        )

        assert len(records) == 1
        for bbox in records[0].bboxes:
            assert bbox.source == AnnotationSource.compositor
            assert bbox.class_id == 0
            assert 0.0 <= bbox.x_center <= 1.0
            assert 0.0 <= bbox.y_center <= 1.0
            assert 0.0 < bbox.width <= 1.0
            assert 0.0 < bbox.height <= 1.0

    def test_negative_examples(self):
        """Task with empty target_classes should produce negative examples."""
        provider = _make_mock_provider()
        compositor = GenerativeDefectCompositor(
            provider,
            requests_per_minute=1000.0,
            max_cost_usd=100.0,
        )

        task = _make_task(num_images=2, target_classes=[])
        records = compositor.generate(
            task,
            backgrounds=_make_backgrounds(),
            img_size=(860, 640),
            seed=42,
        )

        assert len(records) == 2
        for rec in records:
            assert rec.bboxes == []
            assert rec.is_negative
            assert rec.metadata.get("is_negative") is True

    def test_empty_backgrounds_returns_empty(self):
        """No backgrounds → no records."""
        provider = _make_mock_provider()
        compositor = GenerativeDefectCompositor(provider, requests_per_minute=1000.0)

        task = _make_task(num_images=5)
        records = compositor.generate(
            task, backgrounds=[], img_size=(860, 640)
        )
        assert records == []

    def test_cost_limit_stops_generation(self):
        """Generation should stop when cost limit is reached."""
        provider = _make_mock_provider(cost=5.0)
        compositor = GenerativeDefectCompositor(
            provider,
            max_defects_per_image=1,
            requests_per_minute=1000.0,
            max_cost_usd=10.0,
        )

        task = _make_task(num_images=100)
        records = compositor.generate(
            task,
            backgrounds=_make_backgrounds(),
            img_size=(860, 640),
            class_names=["scratch"],
            seed=42,
        )

        # Should stop well before 100 images due to $5/image cost with $10 limit
        assert len(records) < 100
        assert compositor.cumulative_cost <= 15.0  # at most 3 calls

    def test_cumulative_cost_tracked(self):
        """Cumulative cost should increase with each API call."""
        provider = _make_mock_provider(cost=0.04)
        compositor = GenerativeDefectCompositor(
            provider,
            max_defects_per_image=1,
            requests_per_minute=1000.0,
            max_cost_usd=100.0,
        )

        assert compositor.cumulative_cost == 0.0

        task = _make_task(num_images=2)
        compositor.generate(
            task,
            backgrounds=_make_backgrounds(),
            img_size=(860, 640),
            class_names=["scratch"],
            seed=42,
        )

        assert compositor.cumulative_cost > 0.0

    def test_provider_failure_skips_defect(self):
        """When provider raises InpaintingAPIError, the defect is skipped."""
        provider = MagicMock()
        provider.cost_per_image = 0.04
        provider.generate_defect_patch.side_effect = InpaintingAPIError(
            "test", "test error", retryable=False
        )

        compositor = GenerativeDefectCompositor(
            provider,
            max_defects_per_image=1,
            max_retries=1,
            requests_per_minute=1000.0,
            max_cost_usd=100.0,
        )

        task = _make_task(num_images=3)
        records = compositor.generate(
            task,
            backgrounds=_make_backgrounds(),
            img_size=(860, 640),
            class_names=["scratch"],
            seed=42,
        )

        # Records with no bboxes are not added
        assert len(records) == 0

    def test_retryable_error_retries(self):
        """Retryable errors should be retried up to max_retries."""
        provider = MagicMock()
        provider.cost_per_image = 0.04

        # Fail twice with retryable, then succeed
        call_count = 0

        def side_effect(prompt, size=(512, 512), seed=None):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise InpaintingAPIError("test", "rate limit", retryable=True)
            h, w = size[1], size[0]
            return np.zeros((h, w, 3), dtype=np.uint8), np.full((h, w), 255, dtype=np.uint8)

        provider.generate_defect_patch.side_effect = side_effect

        compositor = GenerativeDefectCompositor(
            provider,
            max_defects_per_image=1,
            max_retries=3,
            retry_delay_seconds=0.0,
            requests_per_minute=1000.0,
            max_cost_usd=100.0,
        )

        task = _make_task(num_images=1)
        records = compositor.generate(
            task,
            backgrounds=_make_backgrounds(1),
            img_size=(860, 640),
            class_names=["scratch"],
            seed=42,
        )

        assert call_count == 3  # 2 retries + 1 success
        assert len(records) == 1

    def test_background_resized(self):
        """Backgrounds should be resized to match img_size."""
        provider = _make_mock_provider()
        compositor = GenerativeDefectCompositor(
            provider,
            max_defects_per_image=1,
            requests_per_minute=1000.0,
            max_cost_usd=100.0,
        )

        # Create backgrounds with different size
        backgrounds = [np.zeros((480, 640, 3), dtype=np.uint8)]

        task = _make_task(num_images=1)
        records = compositor.generate(
            task,
            backgrounds=backgrounds,
            img_size=(860, 640),
            class_names=["scratch"],
            seed=42,
        )

        if records:
            assert records[0].image.shape == (640, 860, 3)


class TestPromptPicking:
    def test_known_class_prompt(self):
        """Known class name should pick from class_prompts."""
        provider = _make_mock_provider()
        compositor = GenerativeDefectCompositor(provider, requests_per_minute=1000.0)

        random.seed(42)
        prompt = compositor._pick_prompt("minor scratch")
        assert prompt in DEFAULT_DEFECT_PROMPTS["minor scratch"]

    def test_case_insensitive_match(self):
        """Prompt lookup should be case-insensitive."""
        provider = _make_mock_provider()
        compositor = GenerativeDefectCompositor(provider, requests_per_minute=1000.0)

        random.seed(42)
        prompt = compositor._pick_prompt("MINOR SCRATCH")
        assert prompt in DEFAULT_DEFECT_PROMPTS["minor scratch"]

    def test_unknown_class_fallback(self):
        """Unknown class name should generate a fallback prompt."""
        provider = _make_mock_provider()
        compositor = GenerativeDefectCompositor(provider, requests_per_minute=1000.0)

        prompt = compositor._pick_prompt("unknown_defect")
        assert "unknown_defect" in prompt
        assert "realistic" in prompt.lower()

    def test_custom_class_prompts(self):
        """Custom class_prompts should override defaults."""
        provider = _make_mock_provider()
        custom = {"scratch": ["Custom scratch prompt"]}
        compositor = GenerativeDefectCompositor(
            provider, class_prompts=custom, requests_per_minute=1000.0
        )

        prompt = compositor._pick_prompt("scratch")
        assert prompt == "Custom scratch prompt"


class TestDefaultDefectPrompts:
    def test_all_classes_present(self):
        """All expected defect classes should have prompts."""
        expected_classes = [
            "broken", "hard scratch", "minor crack",
            "minor dent", "minor scratch", "sticker marks",
        ]
        for cls in expected_classes:
            assert cls in DEFAULT_DEFECT_PROMPTS
            assert len(DEFAULT_DEFECT_PROMPTS[cls]) >= 2


class TestSaveIncremental:
    def test_no_output_dir_is_noop(self):
        """_save_incremental with output_dir=None should do nothing."""
        record = ImageRecord(
            image_path=Path("test.jpg"),
            bboxes=[],
            image=np.zeros((10, 10, 3), dtype=np.uint8),
        )
        # Should not raise
        _save_incremental(record, None, "jpg", 95)

    def test_saves_image_and_labels(self, tmp_path):
        """_save_incremental should write image and label files."""
        bbox = BBox(
            class_id=0, x_center=0.5, y_center=0.5,
            width=0.1, height=0.1, source=AnnotationSource.compositor,
        )
        record = ImageRecord(
            image_path=Path("synth_test_0000.jpg"),
            bboxes=[bbox],
            image=np.zeros((100, 100, 3), dtype=np.uint8),
        )

        _save_incremental(record, tmp_path, "jpg", 95)

        img_path = tmp_path / "incremental" / "images" / "synth_test_0000.jpg"
        label_path = tmp_path / "incremental" / "labels" / "synth_test_0000.txt"
        assert img_path.exists()
        assert label_path.exists()

        label_content = label_path.read_text()
        assert label_content.startswith("0 ")

    def test_negative_example_empty_label(self, tmp_path):
        """Negative examples should produce empty label files."""
        record = ImageRecord(
            image_path=Path("synth_neg_0000.jpg"),
            bboxes=[],
            image=np.zeros((100, 100, 3), dtype=np.uint8),
        )

        _save_incremental(record, tmp_path, "jpg", 95)

        label_path = tmp_path / "incremental" / "labels" / "synth_neg_0000.txt"
        assert label_path.exists()
        assert label_path.read_text() == ""

    def test_no_image_data_skips_image_write(self, tmp_path):
        """If record.image is None, no image file should be written."""
        record = ImageRecord(
            image_path=Path("synth_noimg_0000.jpg"),
            bboxes=[],
            image=None,
        )

        _save_incremental(record, tmp_path, "jpg", 95)

        img_path = tmp_path / "incremental" / "images" / "synth_noimg_0000.jpg"
        assert not img_path.exists()

    def test_png_format(self, tmp_path):
        """_save_incremental should support PNG format."""
        record = ImageRecord(
            image_path=Path("synth_png_0000.png"),
            bboxes=[],
            image=np.zeros((100, 100, 3), dtype=np.uint8),
        )

        _save_incremental(record, tmp_path, "png", 95)

        img_path = tmp_path / "incremental" / "images" / "synth_png_0000.png"
        assert img_path.exists()

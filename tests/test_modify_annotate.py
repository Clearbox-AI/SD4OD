"""Tests for synthdet.generate.modify_annotate.

All tests are mock-based — no real API calls, GPU, or model loading required.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from synthdet.config import ModifyAnnotateConfig
from synthdet.generate.errors import InpaintingAPIError
from synthdet.generate.modify_annotate import (
    DEFAULT_DAMAGE_PROMPTS,
    ModifyAndAnnotateGenerator,
    _create_provider,
    _save_record_incremental,
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


def _make_config(**overrides) -> ModifyAnnotateConfig:
    """Create a ModifyAnnotateConfig with test defaults."""
    defaults = {
        "provider": "imagen",
        "annotator": "grounding_dino",
        "requests_per_minute": 1000.0,
        "max_cost_usd": 100.0,
        "max_retries": 3,
        "retry_delay_seconds": 0.0,
        "sam_refine": False,
        "clip_verify": False,
        "project": "test-project",
    }
    defaults.update(overrides)
    return ModifyAnnotateConfig(**defaults)


def _make_mock_provider(cost=0.02):
    """Create a mock modifier provider."""
    provider = MagicMock()
    provider.cost_per_image = cost

    def fake_modify(image, prompt, seed=None, num_images=1):
        return [np.random.randint(0, 255, image.shape, dtype=np.uint8)]

    provider.modify.side_effect = fake_modify
    return provider


def _make_mock_annotator(bboxes=None):
    """Create a mock annotator that returns fixed bboxes."""
    annotator = MagicMock()
    if bboxes is None:
        bboxes = [
            BBox(
                class_id=0, x_center=0.5, y_center=0.5,
                width=0.1, height=0.1,
                source=AnnotationSource.grounding_dino,
                confidence=0.8,
            )
        ]
    annotator.annotate.return_value = bboxes
    return annotator


def _make_task(
    task_id="task_001",
    num_images=2,
    target_classes=None,
):
    """Create a GenerationTask for testing."""
    return GenerationTask(
        task_id=task_id,
        priority=0.8,
        num_images=num_images,
        target_classes=target_classes if target_classes is not None else [0],
        target_size_buckets=[BBoxSizeBucket.medium],
        target_regions=[SpatialRegion.middle_center],
        suggested_prompts=["test damage prompt"],
        rationale="test task",
        method="inpainting",
    )


def _make_source_images(n=3):
    """Create fake source ImageRecords."""
    records = []
    for i in range(n):
        img = np.random.randint(0, 255, (640, 860, 3), dtype=np.uint8)
        records.append(ImageRecord(
            image_path=Path(f"source_{i:04d}.jpg"),
            bboxes=[BBox(class_id=0, x_center=0.5, y_center=0.5,
                         width=0.1, height=0.1, source=AnnotationSource.human)],
            image=img,
        ))
    return records


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestModifyAndAnnotateGenerator:
    def test_basic_generation(self):
        """Generate modified+annotated images with mock provider and annotator."""
        config = _make_config()
        provider = _make_mock_provider()
        generator = ModifyAndAnnotateGenerator(provider, config)
        generator._annotator = _make_mock_annotator()

        task = _make_task(num_images=3)
        records = generator.generate(
            task,
            source_images=_make_source_images(),
            class_names=["scratch"],
            seed=42,
        )

        assert len(records) == 3
        for rec in records:
            assert isinstance(rec, ImageRecord)
            assert rec.image is not None
            assert rec.metadata["source"] == "modify_annotate"

    def test_bboxes_tagged_with_modify_annotate_source(self):
        """All bboxes should be re-tagged with modify_annotate source."""
        config = _make_config()
        provider = _make_mock_provider()
        generator = ModifyAndAnnotateGenerator(provider, config)
        generator._annotator = _make_mock_annotator()

        task = _make_task(num_images=1)
        records = generator.generate(
            task,
            source_images=_make_source_images(),
            class_names=["scratch"],
            seed=42,
        )

        assert len(records) == 1
        for bbox in records[0].bboxes:
            assert bbox.source == AnnotationSource.modify_annotate

    def test_empty_source_images_returns_empty(self):
        """No source images → no records."""
        config = _make_config()
        provider = _make_mock_provider()
        generator = ModifyAndAnnotateGenerator(provider, config)

        task = _make_task(num_images=5)
        records = generator.generate(
            task, source_images=[], class_names=["scratch"]
        )
        assert records == []

    def test_cost_limit_stops_generation(self):
        """Generation should stop when cost limit is reached."""
        config = _make_config(max_cost_usd=0.05)
        provider = _make_mock_provider(cost=0.03)
        generator = ModifyAndAnnotateGenerator(provider, config)
        generator._annotator = _make_mock_annotator()

        task = _make_task(num_images=100)
        records = generator.generate(
            task,
            source_images=_make_source_images(),
            class_names=["scratch"],
            seed=42,
        )

        assert len(records) < 100
        assert generator.cumulative_cost <= 0.10

    def test_cumulative_cost_tracked(self):
        """Cumulative cost should increase with each API call."""
        config = _make_config()
        provider = _make_mock_provider(cost=0.02)
        generator = ModifyAndAnnotateGenerator(provider, config)
        generator._annotator = _make_mock_annotator()

        assert generator.cumulative_cost == 0.0

        task = _make_task(num_images=2)
        generator.generate(
            task,
            source_images=_make_source_images(),
            class_names=["scratch"],
        )

        assert generator.cumulative_cost == pytest.approx(0.04, abs=0.001)

    def test_provider_failure_skips_image(self):
        """When provider raises non-retryable error, image is skipped."""
        config = _make_config(max_retries=1)
        provider = MagicMock()
        provider.cost_per_image = 0.02
        provider.modify.side_effect = InpaintingAPIError(
            "test", "test error", retryable=False
        )
        generator = ModifyAndAnnotateGenerator(provider, config)

        task = _make_task(num_images=3)
        records = generator.generate(
            task,
            source_images=_make_source_images(),
            class_names=["scratch"],
        )

        assert len(records) == 0

    def test_retryable_error_retries(self):
        """Retryable errors should be retried up to max_retries."""
        config = _make_config(max_retries=3, retry_delay_seconds=0.0)
        provider = MagicMock()
        provider.cost_per_image = 0.02

        call_count = 0

        def side_effect(image, prompt, seed=None, num_images=1):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise InpaintingAPIError("test", "rate limit", retryable=True)
            return [np.zeros(image.shape, dtype=np.uint8)]

        provider.modify.side_effect = side_effect
        generator = ModifyAndAnnotateGenerator(provider, config)
        generator._annotator = _make_mock_annotator()

        task = _make_task(num_images=1)
        records = generator.generate(
            task,
            source_images=_make_source_images(1),
            class_names=["scratch"],
        )

        assert call_count == 3
        assert len(records) == 1

    def test_annotator_failure_returns_empty_bboxes(self):
        """When annotator raises, the image record has no bboxes."""
        config = _make_config()
        provider = _make_mock_provider()
        generator = ModifyAndAnnotateGenerator(provider, config)

        annotator = MagicMock()
        annotator.annotate.side_effect = RuntimeError("model crashed")
        generator._annotator = annotator

        task = _make_task(num_images=1)
        records = generator.generate(
            task,
            source_images=_make_source_images(1),
            class_names=["scratch"],
        )

        assert len(records) == 1
        assert records[0].bboxes == []

    def test_modify_single_convenience(self):
        """modify_single() should call provider and return result."""
        config = _make_config()
        provider = _make_mock_provider()
        generator = ModifyAndAnnotateGenerator(provider, config)

        img = np.zeros((64, 64, 3), dtype=np.uint8)
        result = generator.modify_single(img, "add scratch")

        assert result is not None
        assert result.shape == img.shape

    def test_annotate_single_convenience(self):
        """annotate_single() should call annotator and return bboxes."""
        config = _make_config()
        provider = _make_mock_provider()
        generator = ModifyAndAnnotateGenerator(provider, config)
        generator._annotator = _make_mock_annotator()

        img = np.zeros((64, 64, 3), dtype=np.uint8)
        bboxes = generator.annotate_single(img, ["scratch"])

        assert len(bboxes) == 1
        assert bboxes[0].class_id == 0

    def test_metadata_includes_source_image(self):
        """Record metadata should track which source image was used."""
        config = _make_config()
        provider = _make_mock_provider()
        generator = ModifyAndAnnotateGenerator(provider, config)
        generator._annotator = _make_mock_annotator()

        task = _make_task(num_images=1)
        records = generator.generate(
            task,
            source_images=_make_source_images(1),
            class_names=["scratch"],
            seed=42,
        )

        assert "source_image" in records[0].metadata
        assert "prompt" in records[0].metadata


class TestLazyAnnotatorLoading:
    def test_grounding_dino_loaded(self):
        """Annotator='grounding_dino' should lazy-load GroundingDINOAnnotator."""
        config = _make_config(annotator="grounding_dino")
        provider = _make_mock_provider()
        generator = ModifyAndAnnotateGenerator(provider, config)

        # Patch the import
        with patch(
            "synthdet.generate.modify_annotate.GroundingDINOAnnotator",
            create=True,
        ) as mock_cls:
            mock_cls.return_value = MagicMock()
            from synthdet.annotate.grounding_dino import GroundingDINOAnnotator

            with patch(
                "synthdet.annotate.grounding_dino.GroundingDINOAnnotator",
                mock_cls,
            ):
                annotator = generator._get_annotator()
                assert annotator is not None

    def test_owlvit_loaded(self):
        """Annotator='owlvit' should lazy-load OWLViTAnnotator."""
        config = _make_config(annotator="owlvit")
        provider = _make_mock_provider()
        generator = ModifyAndAnnotateGenerator(provider, config)

        with patch(
            "synthdet.annotate.owlvit.OWLViTAnnotator",
        ) as mock_cls:
            mock_cls.return_value = MagicMock()
            annotator = generator._get_annotator()
            assert annotator is not None

    def test_unknown_annotator_raises(self):
        """Unknown annotator name should raise ValueError."""
        config = _make_config(annotator="nonexistent")
        provider = _make_mock_provider()
        generator = ModifyAndAnnotateGenerator(provider, config)

        with pytest.raises(ValueError, match="Unknown annotator"):
            generator._get_annotator()

    def test_annotator_cached(self):
        """Second call to _get_annotator should return the same instance."""
        config = _make_config()
        provider = _make_mock_provider()
        generator = ModifyAndAnnotateGenerator(provider, config)

        mock_annotator = MagicMock()
        generator._annotator = mock_annotator

        assert generator._get_annotator() is mock_annotator


class TestSAMAndCLIPIntegration:
    def test_sam_refine_when_enabled(self):
        """SAM refinement should be applied when config.sam_refine=True."""
        config = _make_config(sam_refine=True)
        provider = _make_mock_provider()
        generator = ModifyAndAnnotateGenerator(provider, config)
        generator._annotator = _make_mock_annotator()

        # Mock SAM refiner
        mock_sam = MagicMock()
        refined_bbox = BBox(
            class_id=0, x_center=0.5, y_center=0.5,
            width=0.08, height=0.08,
            source=AnnotationSource.sam_refined,
        )
        mock_sam.refine.return_value = [refined_bbox]
        generator._sam_refiner = mock_sam

        task = _make_task(num_images=1)
        records = generator.generate(
            task,
            source_images=_make_source_images(1),
            class_names=["scratch"],
        )

        mock_sam.refine.assert_called_once()

    def test_clip_verify_when_enabled(self):
        """CLIP verification should filter bboxes when config.clip_verify=True."""
        config = _make_config(clip_verify=True, min_clip_score=0.5)
        provider = _make_mock_provider()
        generator = ModifyAndAnnotateGenerator(provider, config)

        good_bbox = BBox(
            class_id=0, x_center=0.5, y_center=0.5,
            width=0.1, height=0.1,
            source=AnnotationSource.grounding_dino,
            confidence=0.9,
        )
        bad_bbox = BBox(
            class_id=0, x_center=0.2, y_center=0.2,
            width=0.05, height=0.05,
            source=AnnotationSource.grounding_dino,
            confidence=0.3,
        )
        generator._annotator = _make_mock_annotator([good_bbox, bad_bbox])

        # Mock CLIP verifier — returns (bbox, score) tuples
        mock_clip = MagicMock()
        mock_clip.verify.return_value = [(good_bbox, 0.8), (bad_bbox, 0.2)]
        generator._clip_verifier = mock_clip

        task = _make_task(num_images=1)
        records = generator.generate(
            task,
            source_images=_make_source_images(1),
            class_names=["scratch"],
        )

        # Only the good bbox (score 0.8 >= 0.5) should survive
        assert len(records) == 1
        assert len(records[0].bboxes) == 1

    def test_sam_not_called_when_disabled(self):
        """SAM should not be loaded/called when config.sam_refine=False."""
        config = _make_config(sam_refine=False)
        provider = _make_mock_provider()
        generator = ModifyAndAnnotateGenerator(provider, config)

        assert generator._get_sam_refiner() is None

    def test_clip_not_called_when_disabled(self):
        """CLIP should not be loaded/called when config.clip_verify=False."""
        config = _make_config(clip_verify=False)
        provider = _make_mock_provider()
        generator = ModifyAndAnnotateGenerator(provider, config)

        assert generator._get_clip_verifier(["scratch"]) is None


class TestPromptBuilding:
    def test_class_prompts_from_config(self):
        """Config class_prompts should take precedence."""
        config = _make_config(
            class_prompts={"scratch": ["Config scratch prompt"]}
        )
        provider = _make_mock_provider()
        generator = ModifyAndAnnotateGenerator(provider, config)

        prompt = generator._build_prompt("scratch", [])
        assert prompt == "Config scratch prompt"

    def test_case_insensitive_match(self):
        """Class name matching should be case-insensitive."""
        config = _make_config(
            class_prompts={"Scratch": ["Matched prompt"]}
        )
        provider = _make_mock_provider()
        generator = ModifyAndAnnotateGenerator(provider, config)

        prompt = generator._build_prompt("scratch", [])
        assert prompt == "Matched prompt"

    def test_fallback_to_default_prompts(self):
        """When no class match, use config.default_prompts."""
        config = _make_config(
            class_prompts={},
            default_prompts=["Default fallback prompt"],
        )
        provider = _make_mock_provider()
        generator = ModifyAndAnnotateGenerator(provider, config)

        prompt = generator._build_prompt("unknown_class", [])
        assert prompt == "Default fallback prompt"

    def test_fallback_to_builtin_defaults(self):
        """When config has no prompts, fall back to DEFAULT_DAMAGE_PROMPTS."""
        config = _make_config(class_prompts={}, default_prompts=[])
        provider = _make_mock_provider()
        generator = ModifyAndAnnotateGenerator(provider, config)

        prompt = generator._build_prompt("minor scratch", [])
        assert prompt in DEFAULT_DAMAGE_PROMPTS["minor scratch"]

    def test_fallback_to_task_prompts(self):
        """When nothing matches, use task suggested_prompts."""
        config = _make_config(class_prompts={}, default_prompts=[])
        provider = _make_mock_provider()
        generator = ModifyAndAnnotateGenerator(provider, config)

        prompt = generator._build_prompt("totally_unknown", ["Task prompt"])
        assert prompt == "Task prompt"

    def test_generic_fallback(self):
        """When all else fails, generate a generic prompt."""
        config = _make_config(class_prompts={}, default_prompts=[])
        provider = _make_mock_provider()
        generator = ModifyAndAnnotateGenerator(provider, config)

        prompt = generator._build_prompt("totally_unknown", [])
        assert "totally_unknown" in prompt


class TestDefaultDamagePrompts:
    def test_all_classes_present(self):
        expected = [
            "broken", "hard scratch", "minor crack",
            "minor dent", "minor scratch", "sticker marks",
        ]
        for cls in expected:
            assert cls in DEFAULT_DAMAGE_PROMPTS
            assert len(DEFAULT_DAMAGE_PROMPTS[cls]) >= 2


class TestSaveRecordIncremental:
    def test_no_output_dir_is_noop(self):
        config = _make_config()
        record = ImageRecord(
            image_path=Path("test.jpg"),
            bboxes=[],
            image=np.zeros((10, 10, 3), dtype=np.uint8),
        )
        _save_record_incremental(record, None, config)

    def test_saves_image_and_labels(self, tmp_path):
        config = _make_config()
        bbox = BBox(
            class_id=0, x_center=0.5, y_center=0.5,
            width=0.1, height=0.1, source=AnnotationSource.modify_annotate,
        )
        record = ImageRecord(
            image_path=Path("modanno_test_0000.jpg"),
            bboxes=[bbox],
            image=np.zeros((100, 100, 3), dtype=np.uint8),
        )

        _save_record_incremental(record, tmp_path, config)

        img_path = tmp_path / "incremental" / "images" / "modanno_test_0000.jpg"
        label_path = tmp_path / "incremental" / "labels" / "modanno_test_0000.txt"
        assert img_path.exists()
        assert label_path.exists()

    def test_empty_label_for_no_bboxes(self, tmp_path):
        config = _make_config()
        record = ImageRecord(
            image_path=Path("modanno_neg_0000.jpg"),
            bboxes=[],
            image=np.zeros((100, 100, 3), dtype=np.uint8),
        )

        _save_record_incremental(record, tmp_path, config)

        label_path = tmp_path / "incremental" / "labels" / "modanno_neg_0000.txt"
        assert label_path.read_text() == ""


class TestCreateProvider:
    def test_imagen_provider(self, monkeypatch):
        """_create_provider with 'imagen' should create ImagenModifierProvider."""
        monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
        config = _make_config(provider="imagen")
        provider = _create_provider(config)
        from synthdet.generate.providers.imagen_modifier import ImagenModifierProvider
        assert isinstance(provider, ImagenModifierProvider)

    def test_unknown_provider_raises(self):
        """_create_provider with unknown provider should raise."""
        config = _make_config(provider="nonexistent")
        with pytest.raises(ValueError, match="Unknown modifier provider"):
            _create_provider(config)

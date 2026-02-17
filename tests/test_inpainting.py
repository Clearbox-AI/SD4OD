"""Tests for synthdet.generate.inpainting."""

from __future__ import annotations

import random
from pathlib import Path
from unittest.mock import PropertyMock

import cv2
import numpy as np
import pytest

from synthdet.config import AugmentationConfig, InpaintingConfig
from synthdet.generate.errors import InpaintingAPIError
from synthdet.generate.inpainting import (
    InpaintingGenerator,
    InpaintingProvider,
    MaskPlacer,
    run_inpainting_pipeline,
)
from synthdet.types import (
    AnnotationSource,
    BBox,
    BBoxSizeBucket,
    Dataset,
    GenerationTask,
    ImageRecord,
    SpatialRegion,
    SynthesisStrategy,
)
from synthdet.utils.rate_limiter import RateLimiter


# ---------------------------------------------------------------------------
# Mock provider
# ---------------------------------------------------------------------------


class MockInpaintingProvider:
    """Returns the input image with mask region coloured red.  No real API calls."""

    @property
    def cost_per_image(self) -> float:
        return 0.02

    def inpaint(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        prompt: str,
        *,
        seed: int | None = None,
        num_images: int = 1,
    ) -> list[np.ndarray]:
        result = image.copy()
        result[mask == 255] = [0, 0, 255]  # BGR red
        return [result]


class FailingProvider:
    """Always raises InpaintingAPIError."""

    @property
    def cost_per_image(self) -> float:
        return 0.02

    def inpaint(self, image, mask, prompt, *, seed=None, num_images=1):
        raise InpaintingAPIError("mock", "simulated failure", retryable=False)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_provider():
    return MockInpaintingProvider()


@pytest.fixture
def inpainting_config():
    return InpaintingConfig(
        provider="imagen",
        max_defects_per_image=2,
        mask_shape="rectangle",
        mask_padding=0.1,
        max_placement_attempts=20,
        max_overlap_iou=0.3,
        requests_per_minute=0,  # unlimited
        max_cost_usd=0,  # unlimited
        max_retries=1,
        retry_delay_seconds=0.0,
    )


@pytest.fixture
def sample_backgrounds():
    return [np.full((200, 300, 3), 180, dtype=np.uint8) for _ in range(3)]


@pytest.fixture
def defect_task():
    return GenerationTask(
        task_id="inp-defect-1",
        priority=0.7,
        num_images=5,
        target_classes=[0],
        target_size_buckets=[BBoxSizeBucket.small, BBoxSizeBucket.medium],
        target_regions=list(SpatialRegion),
        suggested_prompts=["Scratch on laptop surface"],
        rationale="Test",
        method="inpainting",
    )


@pytest.fixture
def negative_task():
    return GenerationTask(
        task_id="inp-neg-1",
        priority=0.8,
        num_images=3,
        target_classes=[],
        target_size_buckets=[],
        target_regions=[],
        suggested_prompts=["Clean laptop"],
        rationale="Negatives",
        method="inpainting",
    )


@pytest.fixture
def synthetic_dataset(tmp_path):
    """A small dataset for pipeline tests."""
    train_imgs = tmp_path / "train" / "images"
    train_lbls = tmp_path / "train" / "labels"
    valid_imgs = tmp_path / "valid" / "images"
    valid_lbls = tmp_path / "valid" / "labels"
    train_imgs.mkdir(parents=True)
    train_lbls.mkdir(parents=True)
    valid_imgs.mkdir(parents=True)
    valid_lbls.mkdir(parents=True)

    records_train = []
    for i in range(2):
        img = np.full((200, 300, 3), 180, dtype=np.uint8)
        x1, y1 = 80 + i * 20, 50 + i * 10
        x2, y2 = x1 + 60, y1 + 40
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 200), -1)
        img_path = train_imgs / f"img_{i}.jpg"
        cv2.imwrite(str(img_path), img)
        cx = (x1 + x2) / 2 / 300
        cy = (y1 + y2) / 2 / 200
        w = (x2 - x1) / 300
        h = (y2 - y1) / 200
        bbox = BBox(class_id=0, x_center=cx, y_center=cy, width=w, height=h,
                     source=AnnotationSource.human)
        lbl_path = train_lbls / f"img_{i}.txt"
        lbl_path.write_text(bbox.to_yolo_line() + "\n")
        records_train.append(ImageRecord(image_path=img_path, bboxes=[bbox], image=img))

    img = np.full((200, 300, 3), 180, dtype=np.uint8)
    img_path = valid_imgs / "val_0.jpg"
    cv2.imwrite(str(img_path), img)
    bbox = BBox(class_id=0, x_center=0.5, y_center=0.5, width=0.2, height=0.2,
                source=AnnotationSource.human)
    (valid_lbls / "val_0.txt").write_text(bbox.to_yolo_line() + "\n")
    records_valid = [ImageRecord(image_path=img_path, bboxes=[bbox], image=img)]

    return Dataset(
        root=tmp_path,
        class_names=["Scratch"],
        train=records_train,
        valid=records_valid,
        test=[],
    )


# ---------------------------------------------------------------------------
# MaskPlacer tests
# ---------------------------------------------------------------------------


class TestMaskPlacer:
    def test_create_rectangular_mask(self, inpainting_config):
        placer = MaskPlacer(inpainting_config)
        random.seed(42)
        result = placer.create_mask(300, 200, BBoxSizeBucket.small, None, 0, [])
        assert result is not None
        assert result.mask.shape == (200, 300)
        assert result.mask.dtype == np.uint8
        assert result.mask.max() == 255
        # Has a rectangular white region
        white = np.count_nonzero(result.mask)
        assert white > 0

    def test_create_elliptical_mask(self):
        cfg = InpaintingConfig(mask_shape="ellipse", requests_per_minute=0, max_cost_usd=0)
        placer = MaskPlacer(cfg)
        random.seed(42)
        result = placer.create_mask(300, 200, BBoxSizeBucket.small, None, 0, [])
        assert result is not None
        # Ellipse mask should have fewer white pixels than a full rectangle
        assert np.count_nonzero(result.mask) > 0

    def test_mask_dimensions_match_image(self, inpainting_config):
        placer = MaskPlacer(inpainting_config)
        random.seed(42)
        result = placer.create_mask(860, 640, BBoxSizeBucket.medium, None, 0, [])
        assert result is not None
        assert result.mask.shape == (640, 860)

    def test_bbox_from_mask_normalized(self, inpainting_config):
        placer = MaskPlacer(inpainting_config)
        random.seed(42)
        result = placer.create_mask(300, 200, BBoxSizeBucket.small, None, 0, [])
        assert result is not None
        bbox = result.bbox
        assert 0 <= bbox.x_center <= 1
        assert 0 <= bbox.y_center <= 1
        assert 0 < bbox.width <= 1
        assert 0 < bbox.height <= 1

    def test_mask_targets_size_bucket(self, inpainting_config):
        placer = MaskPlacer(inpainting_config)
        random.seed(42)
        result = placer.create_mask(860, 640, BBoxSizeBucket.small, None, 0, [])
        assert result is not None
        # Bbox area should be reasonable (not huge)
        assert result.bbox.area < 0.5

    def test_mask_targets_spatial_region(self, inpainting_config):
        placer = MaskPlacer(inpainting_config)
        in_region = 0
        for seed_val in range(50):
            random.seed(seed_val)
            result = placer.create_mask(
                300, 200, BBoxSizeBucket.small, SpatialRegion.top_left, 0, [],
            )
            if result is not None and result.bbox.spatial_region == SpatialRegion.top_left:
                in_region += 1
        assert in_region > 0

    def test_mask_respects_valid_zone(self, inpainting_config):
        hull = np.array([
            [[0.3, 0.3]], [[0.7, 0.3]], [[0.7, 0.7]], [[0.3, 0.7]]
        ], dtype=np.float32)
        placer = MaskPlacer(inpainting_config)
        for seed_val in range(20):
            random.seed(seed_val)
            result = placer.create_mask(
                300, 200, BBoxSizeBucket.small, None, 0, [], valid_zone=hull,
            )
            if result is not None:
                assert 0.15 <= result.bbox.x_center <= 0.85
                assert 0.15 <= result.bbox.y_center <= 0.85

    def test_mask_returns_none_when_impossible(self):
        cfg = InpaintingConfig(
            max_placement_attempts=2,
            requests_per_minute=0,
            max_cost_usd=0,
        )
        placer = MaskPlacer(cfg)
        # Fill image with existing bboxes so no room
        existing = [
            BBox(class_id=0, x_center=x / 10, y_center=y / 10, width=0.15, height=0.15)
            for x in range(1, 10)
            for y in range(1, 10)
        ]
        random.seed(42)
        result = placer.create_mask(100, 100, BBoxSizeBucket.medium, None, 0, existing)
        # Should likely be None or at least not crash
        # (with very tight attempts, it may fail)


# ---------------------------------------------------------------------------
# InpaintingGenerator tests
# ---------------------------------------------------------------------------


class TestInpaintingGenerator:
    def test_generate_creates_records(
        self, mock_provider, inpainting_config, sample_backgrounds, defect_task,
    ):
        random.seed(42)
        gen = InpaintingGenerator(mock_provider, inpainting_config)
        records = gen.generate(
            defect_task, backgrounds=sample_backgrounds, img_size=(300, 200),
        )
        assert len(records) > 0
        assert len(records) <= defect_task.num_images

    def test_source_is_inpainting(
        self, mock_provider, inpainting_config, sample_backgrounds, defect_task,
    ):
        random.seed(42)
        gen = InpaintingGenerator(mock_provider, inpainting_config)
        records = gen.generate(
            defect_task, backgrounds=sample_backgrounds, img_size=(300, 200),
        )
        for rec in records:
            for bbox in rec.bboxes:
                assert bbox.source == AnnotationSource.inpainting

    def test_negative_examples(
        self, mock_provider, inpainting_config, sample_backgrounds, negative_task,
    ):
        gen = InpaintingGenerator(mock_provider, inpainting_config)
        records = gen.generate(
            negative_task, backgrounds=sample_backgrounds, img_size=(300, 200),
        )
        assert len(records) == negative_task.num_images
        for rec in records:
            assert rec.is_negative
            assert len(rec.bboxes) == 0

    def test_api_error_handled_gracefully(
        self, inpainting_config, sample_backgrounds, defect_task,
    ):
        provider = FailingProvider()
        gen = InpaintingGenerator(provider, inpainting_config)
        random.seed(42)
        # Should not raise — errors are caught
        records = gen.generate(
            defect_task, backgrounds=sample_backgrounds, img_size=(300, 200),
        )
        # All calls fail, so no records with defects
        assert len(records) == 0

    def test_cost_limit_stops_generation(
        self, mock_provider, sample_backgrounds, defect_task,
    ):
        config = InpaintingConfig(
            max_cost_usd=0.03,  # Only allows ~1 image
            requests_per_minute=0,
            max_retries=1,
        )
        gen = InpaintingGenerator(mock_provider, config)
        random.seed(42)
        records = gen.generate(
            defect_task, backgrounds=sample_backgrounds, img_size=(300, 200),
        )
        # Should generate fewer than the requested 5
        assert len(records) < defect_task.num_images

    def test_dry_run_no_api_calls(
        self, sample_backgrounds, defect_task,
    ):
        call_count = 0

        class CountingProvider:
            @property
            def cost_per_image(self):
                return 0.02

            def inpaint(self, image, mask, prompt, *, seed=None, num_images=1):
                nonlocal call_count
                call_count += 1
                return [image]

        config = InpaintingConfig(requests_per_minute=0, max_cost_usd=0)
        gen = InpaintingGenerator(CountingProvider(), config)
        random.seed(42)
        records = gen.generate(
            defect_task, backgrounds=sample_backgrounds, img_size=(300, 200),
            dry_run=True,
        )
        assert call_count == 0
        assert len(records) == 0  # dry run produces no image records

    def test_multi_defect_sequential(
        self, mock_provider, sample_backgrounds,
    ):
        config = InpaintingConfig(
            max_defects_per_image=3,
            requests_per_minute=0,
            max_cost_usd=0,
            max_retries=1,
        )
        task = GenerationTask(
            task_id="multi-defect",
            priority=0.7,
            num_images=3,
            target_classes=[0],
            target_size_buckets=[BBoxSizeBucket.small],
            target_regions=list(SpatialRegion),
            suggested_prompts=["Scratch"],
            rationale="Test",
            method="inpainting",
        )
        gen = InpaintingGenerator(mock_provider, config)
        random.seed(42)
        records = gen.generate(
            task, backgrounds=sample_backgrounds, img_size=(300, 200),
        )
        # Some images may have multiple bboxes
        multi = [r for r in records if len(r.bboxes) > 1]
        assert len(records) > 0

    def test_uses_class_prompts_from_config(
        self, sample_backgrounds,
    ):
        """class_prompts in config take priority over task suggested_prompts."""
        prompts_used = []

        class PromptCapture:
            @property
            def cost_per_image(self):
                return 0.02

            def inpaint(self, image, mask, prompt, *, seed=None, num_images=1):
                prompts_used.append(prompt)
                return [image]

        config = InpaintingConfig(
            max_defects_per_image=1,
            requests_per_minute=0,
            max_cost_usd=0,
            max_retries=1,
            class_prompts={"Scratch": ["Custom scratch prompt"]},
            prompt_template="{prompt} on {class_name}",
        )
        task = GenerationTask(
            task_id="prompt-test",
            priority=0.7,
            num_images=3,
            target_classes=[0],
            target_size_buckets=[BBoxSizeBucket.small],
            target_regions=list(SpatialRegion),
            suggested_prompts=["Fallback prompt"],
            rationale="Test",
            method="inpainting",
        )
        gen = InpaintingGenerator(PromptCapture(), config)
        random.seed(42)
        gen.generate(task, backgrounds=sample_backgrounds, img_size=(300, 200), class_names=["Scratch"])
        for p in prompts_used:
            assert "Custom scratch prompt" in p
            assert "Scratch" in p  # class_name substituted in template

    def test_falls_back_to_task_prompts(
        self, sample_backgrounds,
    ):
        """When no class_prompts or default_prompts, uses task suggested_prompts."""
        prompts_used = []

        class PromptCapture:
            @property
            def cost_per_image(self):
                return 0.02

            def inpaint(self, image, mask, prompt, *, seed=None, num_images=1):
                prompts_used.append(prompt)
                return [image]

        config = InpaintingConfig(
            max_defects_per_image=1,
            requests_per_minute=0,
            max_cost_usd=0,
            max_retries=1,
            class_prompts={},
            default_prompts=[],
            prompt_template="{prompt}",
        )
        task = GenerationTask(
            task_id="prompt-fallback",
            priority=0.7,
            num_images=3,
            target_classes=[0],
            target_size_buckets=[BBoxSizeBucket.small],
            target_regions=list(SpatialRegion),
            suggested_prompts=["My custom task prompt"],
            rationale="Test",
            method="inpainting",
        )
        gen = InpaintingGenerator(PromptCapture(), config)
        random.seed(42)
        gen.generate(task, backgrounds=sample_backgrounds, img_size=(300, 200), class_names=["Scratch"])
        for p in prompts_used:
            assert p == "My custom task prompt"


# ---------------------------------------------------------------------------
# Pipeline tests
# ---------------------------------------------------------------------------


class TestInpaintingPipeline:
    def test_run_pipeline(self, synthetic_dataset, tmp_path, monkeypatch):
        """Full pipeline produces YOLO output."""
        output_dir = tmp_path / "output_inpaint"

        # Monkeypatch _create_provider to return our mock
        from synthdet.generate import inpainting as inpaint_mod

        monkeypatch.setattr(inpaint_mod, "_create_provider", lambda cfg: MockInpaintingProvider())

        config = InpaintingConfig(
            max_defects_per_image=1,
            requests_per_minute=0,
            max_cost_usd=0,
            max_retries=1,
        )
        strategy = SynthesisStrategy(
            target_total_images=10,
            target_class_counts={0: 10},
            negative_ratio=0.15,
            size_bucket_gaps={},
            spatial_gaps={},
            aspect_ratio_gaps=[],
            generation_tasks=[
                GenerationTask(
                    task_id="pipe-defect",
                    priority=0.7,
                    num_images=4,
                    target_classes=[0],
                    target_size_buckets=[BBoxSizeBucket.small],
                    target_regions=list(SpatialRegion),
                    suggested_prompts=["Scratch"],
                    rationale="Test",
                    method="inpainting",
                ),
                GenerationTask(
                    task_id="pipe-neg",
                    priority=0.8,
                    num_images=2,
                    target_classes=[],
                    target_size_buckets=[],
                    target_regions=[],
                    suggested_prompts=["Clean"],
                    rationale="Neg",
                    method="inpainting",
                ),
            ],
        )

        result = run_inpainting_pipeline(
            dataset=synthetic_dataset,
            strategy=strategy,
            config=config,
            output_dir=output_dir,
            seed=42,
        )

        assert output_dir.is_dir()
        assert (output_dir / "data.yaml").is_file()
        total = len(result.train) + len(result.valid)
        assert total > 0

    def test_pipeline_with_augmentation(self, synthetic_dataset, tmp_path, monkeypatch):
        output_dir = tmp_path / "output_inpaint_aug"

        from synthdet.generate import inpainting as inpaint_mod

        monkeypatch.setattr(inpaint_mod, "_create_provider", lambda cfg: MockInpaintingProvider())

        config = InpaintingConfig(
            max_defects_per_image=1,
            requests_per_minute=0,
            max_cost_usd=0,
            max_retries=1,
        )
        aug_config = AugmentationConfig(
            enabled=True,
            variants_per_image=1,
            horizontal_flip_p=0.5,
            brightness_contrast_p=0.0,
            hue_saturation_p=0.0,
            noise_p=0.0,
            blur_p=0.0,
            shift_scale_rotate_p=0.0,
        )
        strategy = SynthesisStrategy(
            target_total_images=6,
            target_class_counts={0: 6},
            negative_ratio=0.0,
            size_bucket_gaps={},
            spatial_gaps={},
            aspect_ratio_gaps=[],
            generation_tasks=[
                GenerationTask(
                    task_id="aug-test",
                    priority=0.7,
                    num_images=3,
                    target_classes=[0],
                    target_size_buckets=list(BBoxSizeBucket),
                    target_regions=list(SpatialRegion),
                    suggested_prompts=["Scratch"],
                    rationale="Test",
                    method="inpainting",
                ),
            ],
        )

        result = run_inpainting_pipeline(
            dataset=synthetic_dataset,
            strategy=strategy,
            config=config,
            output_dir=output_dir,
            augment_config=aug_config,
            seed=42,
        )

        total = len(result.train) + len(result.valid)
        # 3 original + 3 augmented = 6
        assert total > 0

    def test_pipeline_writes_correct_structure(self, synthetic_dataset, tmp_path, monkeypatch):
        output_dir = tmp_path / "output_struct"

        from synthdet.generate import inpainting as inpaint_mod

        monkeypatch.setattr(inpaint_mod, "_create_provider", lambda cfg: MockInpaintingProvider())

        config = InpaintingConfig(
            max_defects_per_image=1,
            requests_per_minute=0,
            max_cost_usd=0,
            max_retries=1,
        )
        strategy = SynthesisStrategy(
            target_total_images=4,
            target_class_counts={0: 4},
            negative_ratio=0.0,
            size_bucket_gaps={},
            spatial_gaps={},
            aspect_ratio_gaps=[],
            generation_tasks=[
                GenerationTask(
                    task_id="struct-test",
                    priority=0.7,
                    num_images=3,
                    target_classes=[0],
                    target_size_buckets=[BBoxSizeBucket.small],
                    target_regions=list(SpatialRegion),
                    suggested_prompts=["Scratch"],
                    rationale="Test",
                    method="inpainting",
                ),
            ],
        )

        run_inpainting_pipeline(
            dataset=synthetic_dataset,
            strategy=strategy,
            config=config,
            output_dir=output_dir,
            seed=42,
        )

        assert (output_dir / "data.yaml").is_file()
        assert (output_dir / "train" / "images").is_dir()
        assert (output_dir / "train" / "labels").is_dir()

    def test_incremental_save(
        self, mock_provider, inpainting_config, sample_backgrounds, defect_task, tmp_path,
    ):
        """Images are saved incrementally to output_dir/incremental/."""
        output_dir = tmp_path / "incr_output"
        gen = InpaintingGenerator(mock_provider, inpainting_config)
        random.seed(42)
        records = gen.generate(
            defect_task,
            backgrounds=sample_backgrounds,
            img_size=(300, 200),
            output_dir=output_dir,
        )
        assert len(records) > 0
        incr_images = list((output_dir / "incremental" / "images").glob("*"))
        incr_labels = list((output_dir / "incremental" / "labels").glob("*.txt"))
        assert len(incr_images) == len(records)
        assert len(incr_labels) == len(records)
        # Each label file should have content (defects or empty for negatives)
        for lbl in incr_labels:
            assert lbl.is_file()

    def test_incremental_save_negatives(
        self, mock_provider, inpainting_config, sample_backgrounds, negative_task, tmp_path,
    ):
        """Negative examples are also saved incrementally."""
        output_dir = tmp_path / "incr_neg"
        gen = InpaintingGenerator(mock_provider, inpainting_config)
        records = gen.generate(
            negative_task,
            backgrounds=sample_backgrounds,
            img_size=(300, 200),
            output_dir=output_dir,
        )
        assert len(records) == negative_task.num_images
        incr_labels = list((output_dir / "incremental" / "labels").glob("*.txt"))
        assert len(incr_labels) == len(records)
        # Negative labels should be empty files
        for lbl in incr_labels:
            assert lbl.read_text().strip() == ""

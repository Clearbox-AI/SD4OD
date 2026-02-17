"""Tests for the defect compositor pipeline."""

from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np
import pytest

from synthdet.config import AugmentationConfig, CompositorConfig
from synthdet.generate.compositor import (
    BackgroundGenerator,
    DefectCompositor,
    DefectPatch,
    DefectPatchExtractor,
    _check_placement_valid,
    _create_feathered_mask,
    _determine_center,
    _scale_patch,
    compute_valid_zone,
    point_in_valid_zone,
    run_compositor_pipeline,
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
from synthdet.utils.bbox import bbox_iou


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_image() -> np.ndarray:
    """A 200x300 BGR image with a visible rectangle (simulating a defect)."""
    img = np.full((200, 300, 3), 180, dtype=np.uint8)  # Gray background
    cv2.rectangle(img, (100, 60), (180, 120), (0, 0, 200), -1)  # Red "defect"
    return img


@pytest.fixture
def synthetic_dataset(tmp_path) -> Dataset:
    """A small synthetic dataset with 3 images and annotations."""
    train_imgs = tmp_path / "train" / "images"
    train_lbls = tmp_path / "train" / "labels"
    valid_imgs = tmp_path / "valid" / "images"
    valid_lbls = tmp_path / "valid" / "labels"
    train_imgs.mkdir(parents=True)
    train_lbls.mkdir(parents=True)
    valid_imgs.mkdir(parents=True)
    valid_lbls.mkdir(parents=True)

    records_train = []
    records_valid = []

    # Create 2 train images with defects
    for i in range(2):
        img = np.full((200, 300, 3), 180, dtype=np.uint8)
        # Draw a "defect" rectangle
        x1, y1 = 80 + i * 20, 50 + i * 10
        x2, y2 = x1 + 60, y1 + 40
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 200), -1)

        img_path = train_imgs / f"img_{i}.jpg"
        cv2.imwrite(str(img_path), img)

        # YOLO bbox (normalized)
        cx = (x1 + x2) / 2 / 300
        cy = (y1 + y2) / 2 / 200
        w = (x2 - x1) / 300
        h = (y2 - y1) / 200
        bbox = BBox(class_id=0, x_center=cx, y_center=cy, width=w, height=h,
                     source=AnnotationSource.human)

        lbl_path = train_lbls / f"img_{i}.txt"
        lbl_path.write_text(bbox.to_yolo_line() + "\n")

        records_train.append(ImageRecord(image_path=img_path, bboxes=[bbox], image=img))

    # Create 1 valid image
    img = np.full((200, 300, 3), 180, dtype=np.uint8)
    cv2.rectangle(img, (120, 80), (200, 140), (0, 200, 0), -1)  # Green "defect"
    img_path = valid_imgs / "val_0.jpg"
    cv2.imwrite(str(img_path), img)
    bbox = BBox(class_id=0, x_center=160/300, y_center=110/200, width=80/300, height=60/200,
                source=AnnotationSource.human)
    lbl_path = valid_lbls / "val_0.txt"
    lbl_path.write_text(bbox.to_yolo_line() + "\n")
    records_valid.append(ImageRecord(image_path=img_path, bboxes=[bbox], image=img))

    return Dataset(
        root=tmp_path,
        class_names=["Scratch"],
        train=records_train,
        valid=records_valid,
        test=[],
    )


@pytest.fixture
def sample_patches(synthetic_image) -> list[DefectPatch]:
    """A few sample DefectPatch objects."""
    patch = synthetic_image[60:120, 100:180].copy()
    mask = _create_feathered_mask(60, 80)
    return [
        DefectPatch(
            image=patch,
            mask=mask,
            class_id=0,
            original_size=(80/300, 60/200),
            source_path=Path("test.jpg"),
        ),
    ]


@pytest.fixture
def sample_backgrounds() -> list[np.ndarray]:
    """Clean background images."""
    return [np.full((200, 300, 3), 180, dtype=np.uint8) for _ in range(3)]


@pytest.fixture
def simple_strategy() -> SynthesisStrategy:
    """A minimal synthesis strategy for testing."""
    return SynthesisStrategy(
        target_total_images=10,
        target_class_counts={0: 10},
        negative_ratio=0.15,
        size_bucket_gaps={},
        spatial_gaps={},
        aspect_ratio_gaps=[],
        generation_tasks=[
            GenerationTask(
                task_id="test-defect-1",
                priority=0.7,
                num_images=5,
                target_classes=[0],
                target_size_buckets=[BBoxSizeBucket.small, BBoxSizeBucket.medium],
                target_regions=list(SpatialRegion),
                suggested_prompts=["Test defect"],
                rationale="Test",
                method="compositor",
            ),
            GenerationTask(
                task_id="test-negative-1",
                priority=0.8,
                num_images=2,
                target_classes=[],
                target_size_buckets=[],
                target_regions=[],
                suggested_prompts=["Clean surface"],
                rationale="Negative examples",
                method="compositor",
            ),
        ],
    )


# ---------------------------------------------------------------------------
# DefectPatchExtractor tests
# ---------------------------------------------------------------------------


class TestDefectPatchExtractor:
    def test_extract_patches_from_dataset(self, synthetic_dataset):
        extractor = DefectPatchExtractor(margin=0.1, min_patch_pixels=4)
        patches = extractor.extract_patches(synthetic_dataset)
        # Should extract from both train images
        assert len(patches) == 2
        for p in patches:
            assert p.image.ndim == 3
            assert p.mask.ndim == 2
            assert p.mask.shape == p.image.shape[:2]
            assert p.class_id == 0

    def test_extract_single_patch_with_margin(self, synthetic_image):
        extractor = DefectPatchExtractor(margin=0.15, min_patch_pixels=4)
        bbox = BBox(class_id=0, x_center=140/300, y_center=90/200,
                    width=80/300, height=60/200, source=AnnotationSource.human)

        patch = extractor._extract_single(
            synthetic_image, bbox, 300, 200, Path("test.jpg")
        )
        assert patch is not None
        # With margin, crop should be larger than the bbox pixel size
        assert patch.image.shape[1] >= 80  # width
        assert patch.image.shape[0] >= 60  # height
        assert patch.mask.shape == patch.image.shape[:2]

    def test_skip_tiny_patch(self, synthetic_image):
        extractor = DefectPatchExtractor(margin=0.0, min_patch_pixels=1000)
        bbox = BBox(class_id=0, x_center=0.5, y_center=0.5,
                    width=0.01, height=0.01, source=AnnotationSource.human)
        patch = extractor._extract_single(
            synthetic_image, bbox, 300, 200, Path("test.jpg")
        )
        assert patch is None


# ---------------------------------------------------------------------------
# BackgroundGenerator tests
# ---------------------------------------------------------------------------


class TestBackgroundGenerator:
    def test_generate_backgrounds(self, synthetic_dataset):
        bg_gen = BackgroundGenerator(inpaint_radius=3, method="telea")
        backgrounds = bg_gen.generate_from_dataset(synthetic_dataset)
        assert len(backgrounds) == 2  # 2 unique train images
        for bg in backgrounds:
            assert bg.shape == (200, 300, 3)

    def test_load_from_directory(self, tmp_path):
        # Create some background images
        for i in range(3):
            img = np.full((100, 150, 3), 200, dtype=np.uint8)
            cv2.imwrite(str(tmp_path / f"bg_{i}.jpg"), img)

        bg_gen = BackgroundGenerator()
        backgrounds = bg_gen.load_from_directory(tmp_path)
        assert len(backgrounds) == 3


# ---------------------------------------------------------------------------
# DefectCompositor tests
# ---------------------------------------------------------------------------


class TestDefectCompositor:
    def test_composite_single_defect(self, sample_patches, sample_backgrounds):
        config = CompositorConfig(max_defects_per_image=1, rotation_jitter=0)
        compositor = DefectCompositor(config)

        task = GenerationTask(
            task_id="test-1",
            priority=0.7,
            num_images=3,
            target_classes=[0],
            target_size_buckets=[BBoxSizeBucket.small],
            target_regions=[SpatialRegion.middle_center],
            suggested_prompts=[],
            rationale="Test",
            method="compositor",
        )

        random.seed(42)
        np.random.seed(42)
        records = compositor.generate(
            task,
            patches=sample_patches,
            backgrounds=sample_backgrounds,
            img_size=(300, 200),
        )

        assert len(records) == 3
        for rec in records:
            assert rec.image is not None
            assert rec.image.shape == (200, 300, 3)
            assert len(rec.bboxes) > 0
            for bbox in rec.bboxes:
                assert bbox.source == AnnotationSource.compositor
                assert 0 <= bbox.x_center <= 1
                assert 0 <= bbox.y_center <= 1
                assert 0 < bbox.width <= 1
                assert 0 < bbox.height <= 1

    def test_composite_respects_target_region(self, sample_patches, sample_backgrounds):
        config = CompositorConfig(max_defects_per_image=1, rotation_jitter=0)
        compositor = DefectCompositor(config)
        target_region = SpatialRegion.top_left

        task = GenerationTask(
            task_id="region-test",
            priority=0.7,
            num_images=10,
            target_classes=[0],
            target_size_buckets=[BBoxSizeBucket.small],
            target_regions=[target_region],
            suggested_prompts=[],
            rationale="Test",
            method="compositor",
        )

        random.seed(42)
        np.random.seed(42)
        records = compositor.generate(
            task,
            patches=sample_patches,
            backgrounds=sample_backgrounds,
            img_size=(300, 200),
        )

        # At least some bboxes should be in the target region
        in_region = sum(
            1 for r in records for b in r.bboxes
            if b.spatial_region == target_region
        )
        assert in_region > 0

    def test_composite_respects_target_size(self, sample_patches, sample_backgrounds):
        config = CompositorConfig(
            max_defects_per_image=1,
            rotation_jitter=0,
            scale_jitter=(0.95, 1.05),  # tight jitter for testing
        )
        compositor = DefectCompositor(config)

        task = GenerationTask(
            task_id="size-test",
            priority=0.7,
            num_images=10,
            target_classes=[0],
            target_size_buckets=[BBoxSizeBucket.small],
            target_regions=list(SpatialRegion),
            suggested_prompts=[],
            rationale="Test",
            method="compositor",
        )

        random.seed(42)
        np.random.seed(42)
        records = compositor.generate(
            task,
            patches=sample_patches,
            backgrounds=sample_backgrounds,
            img_size=(300, 200),
        )

        # Check that generated bboxes are in or near the small bucket range
        for r in records:
            for b in r.bboxes:
                # Allow some variance due to scaling/rotation
                assert b.area < 0.5, f"Bbox area {b.area} is unreasonably large"

    def test_negative_example_generation(self, sample_patches, sample_backgrounds):
        config = CompositorConfig()
        compositor = DefectCompositor(config)

        task = GenerationTask(
            task_id="neg-test",
            priority=0.8,
            num_images=5,
            target_classes=[],  # Empty = negative
            target_size_buckets=[],
            target_regions=[],
            suggested_prompts=[],
            rationale="Negatives",
            method="compositor",
        )

        records = compositor.generate(
            task,
            patches=sample_patches,
            backgrounds=sample_backgrounds,
            img_size=(300, 200),
        )

        assert len(records) == 5
        for rec in records:
            assert rec.is_negative
            assert len(rec.bboxes) == 0
            assert rec.image is not None

    def test_no_overlapping_placements(self, sample_patches, sample_backgrounds):
        config = CompositorConfig(
            max_defects_per_image=4,
            max_overlap_iou=0.3,
            rotation_jitter=0,
        )
        compositor = DefectCompositor(config)

        task = GenerationTask(
            task_id="overlap-test",
            priority=0.7,
            num_images=10,
            target_classes=[0],
            target_size_buckets=[BBoxSizeBucket.small],
            target_regions=list(SpatialRegion),
            suggested_prompts=[],
            rationale="Test",
            method="compositor",
        )

        random.seed(42)
        np.random.seed(42)
        records = compositor.generate(
            task,
            patches=sample_patches,
            backgrounds=sample_backgrounds,
            img_size=(300, 200),
        )

        for rec in records:
            bboxes = rec.bboxes
            for i in range(len(bboxes)):
                for j in range(i + 1, len(bboxes)):
                    iou = bbox_iou(bboxes[i], bboxes[j])
                    assert iou <= config.max_overlap_iou + 0.05, (
                        f"IoU {iou:.2f} exceeds max {config.max_overlap_iou}"
                    )


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_create_feathered_mask(self):
        mask = _create_feathered_mask(60, 80)
        assert mask.shape == (60, 80)
        assert mask.dtype == np.uint8
        # Center should be bright, corners should be dark
        assert mask[30, 40] > mask[0, 0]

    def test_check_placement_valid_empty(self):
        bbox = BBox(class_id=0, x_center=0.5, y_center=0.5, width=0.1, height=0.1)
        assert _check_placement_valid(bbox, [], 0.3) is True

    def test_check_placement_valid_overlap(self):
        existing = BBox(class_id=0, x_center=0.5, y_center=0.5, width=0.1, height=0.1)
        new = BBox(class_id=0, x_center=0.52, y_center=0.52, width=0.1, height=0.1)
        # These overlap significantly
        assert _check_placement_valid(new, [existing], 0.3) is False

    def test_determine_center_region(self):
        random.seed(42)
        center = _determine_center(
            SpatialRegion.top_left, 20, 20, 300, 200, [], 10, 0.3
        )
        assert center is not None
        cx, cy = center
        # Should be in top-left third
        assert cx < 120  # 300/3 + patch margin
        assert cy < 90   # 200/3 + patch margin


# ---------------------------------------------------------------------------
# Valid zone tests
# ---------------------------------------------------------------------------


class TestValidZone:
    def test_compute_valid_zone_from_dataset(self, tmp_path):
        """Need at least 3 distinct annotation centers for a convex hull."""
        train_imgs = tmp_path / "train" / "images"
        train_lbls = tmp_path / "train" / "labels"
        train_imgs.mkdir(parents=True)
        train_lbls.mkdir(parents=True)

        records = []
        centers = [(0.3, 0.3), (0.7, 0.3), (0.5, 0.7)]
        for i, (cx, cy) in enumerate(centers):
            img = np.full((200, 300, 3), 180, dtype=np.uint8)
            img_path = train_imgs / f"img_{i}.jpg"
            cv2.imwrite(str(img_path), img)
            bbox = BBox(class_id=0, x_center=cx, y_center=cy,
                        width=0.1, height=0.1, source=AnnotationSource.human)
            records.append(ImageRecord(image_path=img_path, bboxes=[bbox], image=img))

        dataset = Dataset(
            root=tmp_path, class_names=["Scratch"],
            train=records, valid=[], test=[],
        )
        hull = compute_valid_zone(dataset, margin=0.05)
        assert hull is not None
        assert hull.ndim == 3
        assert hull.shape[1] == 1
        assert hull.shape[2] == 2
        assert hull.shape[0] >= 3

    def test_compute_valid_zone_too_few_points(self, tmp_path):
        """With fewer than 3 annotation centers, returns None."""
        img_path = tmp_path / "img.jpg"
        img = np.full((100, 150, 3), 180, dtype=np.uint8)
        cv2.imwrite(str(img_path), img)

        dataset = Dataset(
            root=tmp_path,
            class_names=["Scratch"],
            train=[ImageRecord(
                image_path=img_path,
                bboxes=[BBox(class_id=0, x_center=0.5, y_center=0.5,
                             width=0.1, height=0.1)],
            )],
            valid=[],
            test=[],
        )
        hull = compute_valid_zone(dataset)
        # Only 1 center point — can't form a hull
        assert hull is None

    def test_point_in_valid_zone_inside(self):
        # Square hull: (0.2,0.2), (0.8,0.2), (0.8,0.8), (0.2,0.8)
        hull = np.array([
            [[0.2, 0.2]], [[0.8, 0.2]], [[0.8, 0.8]], [[0.2, 0.8]]
        ], dtype=np.float32)
        assert point_in_valid_zone(0.5, 0.5, hull) is True
        assert point_in_valid_zone(0.3, 0.3, hull) is True

    def test_point_in_valid_zone_outside(self):
        hull = np.array([
            [[0.2, 0.2]], [[0.8, 0.2]], [[0.8, 0.8]], [[0.2, 0.8]]
        ], dtype=np.float32)
        assert point_in_valid_zone(0.05, 0.05, hull) is False
        assert point_in_valid_zone(0.95, 0.95, hull) is False

    def test_point_in_valid_zone_no_hull(self):
        """No hull means no restriction — everything is valid."""
        assert point_in_valid_zone(0.0, 0.0, None) is True
        assert point_in_valid_zone(0.99, 0.99, None) is True

    def test_determine_center_respects_valid_zone(self):
        """Placements should be rejected if outside the valid zone."""
        # Hull covering only the center area (0.3-0.7)
        hull = np.array([
            [[0.3, 0.3]], [[0.7, 0.3]], [[0.7, 0.7]], [[0.3, 0.7]]
        ], dtype=np.float32)

        random.seed(42)
        results = []
        for _ in range(20):
            center = _determine_center(
                None, 10, 10, 300, 200, [], 50, 0.3, valid_zone=hull
            )
            if center is not None:
                cx, cy = center
                norm_x, norm_y = cx / 300, cy / 200
                results.append((norm_x, norm_y))

        assert len(results) > 0, "Should find at least some valid placements"
        for nx, ny in results:
            # Allow small tolerance for patch half-size offsets
            assert 0.2 <= nx <= 0.8, f"x={nx} outside valid zone"
            assert 0.2 <= ny <= 0.8, f"y={ny} outside valid zone"

    def test_compositor_with_valid_zone(self, sample_patches, sample_backgrounds):
        """Compositor should respect valid zone when generating."""
        config = CompositorConfig(max_defects_per_image=1, rotation_jitter=0)
        compositor = DefectCompositor(config)

        # Tight valid zone in center
        hull = np.array([
            [[0.3, 0.3]], [[0.7, 0.3]], [[0.7, 0.7]], [[0.3, 0.7]]
        ], dtype=np.float32)

        task = GenerationTask(
            task_id="zone-test",
            priority=0.7,
            num_images=10,
            target_classes=[0],
            target_size_buckets=[BBoxSizeBucket.small],
            target_regions=list(SpatialRegion),
            suggested_prompts=[],
            rationale="Test",
            method="compositor",
        )

        random.seed(42)
        np.random.seed(42)
        records = compositor.generate(
            task,
            patches=sample_patches,
            backgrounds=sample_backgrounds,
            img_size=(300, 200),
            valid_zone=hull,
        )

        for rec in records:
            for bbox in rec.bboxes:
                # Center should be within or near the valid zone
                assert 0.15 <= bbox.x_center <= 0.85, (
                    f"x_center={bbox.x_center} outside valid zone"
                )
                assert 0.15 <= bbox.y_center <= 0.85, (
                    f"y_center={bbox.y_center} outside valid zone"
                )


# ---------------------------------------------------------------------------
# Integration: full pipeline
# ---------------------------------------------------------------------------


class TestPipeline:
    def test_run_compositor_pipeline(self, synthetic_dataset, tmp_path):
        output_dir = tmp_path / "output"
        config = CompositorConfig(
            max_defects_per_image=2,
            rotation_jitter=5,
            min_patch_pixels=4,
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
                    num_images=5,
                    target_classes=[0],
                    target_size_buckets=[BBoxSizeBucket.small],
                    target_regions=list(SpatialRegion),
                    suggested_prompts=[],
                    rationale="Test",
                    method="compositor",
                ),
                GenerationTask(
                    task_id="pipe-neg",
                    priority=0.8,
                    num_images=2,
                    target_classes=[],
                    target_size_buckets=[],
                    target_regions=[],
                    suggested_prompts=[],
                    rationale="Negatives",
                    method="compositor",
                ),
            ],
        )

        result = run_compositor_pipeline(
            dataset=synthetic_dataset,
            strategy=strategy,
            config=config,
            output_dir=output_dir,
            seed=42,
        )

        assert output_dir.is_dir()
        assert (output_dir / "data.yaml").is_file()
        assert (output_dir / "train" / "images").is_dir()
        assert (output_dir / "train" / "labels").is_dir()

        total = len(result.train) + len(result.valid)
        assert total == 7  # 5 defect + 2 negative

    def test_pipeline_with_augmentation(self, synthetic_dataset, tmp_path):
        output_dir = tmp_path / "output_aug"
        config = CompositorConfig(min_patch_pixels=4, rotation_jitter=0)
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
                    suggested_prompts=[],
                    rationale="Test",
                    method="compositor",
                ),
            ],
        )

        result = run_compositor_pipeline(
            dataset=synthetic_dataset,
            strategy=strategy,
            config=config,
            output_dir=output_dir,
            augment_config=aug_config,
            seed=42,
        )

        total = len(result.train) + len(result.valid)
        # 3 original + 3 augmented variants (1 per image)
        assert total == 6

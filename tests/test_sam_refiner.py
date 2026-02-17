"""Tests for synthdet.annotate.sam_refiner.

All tests are mock-based — no real model loading or GPU required.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

from synthdet.types import AnnotationSource, BBox

# ---------------------------------------------------------------------------
# Mock torch if not installed
# ---------------------------------------------------------------------------

_torch_available = True
try:
    import torch as _real_torch
except ImportError:
    _torch_available = False


@pytest.fixture()
def mock_torch():
    """Provide a mock torch module when torch is not installed."""
    if _torch_available:
        yield _real_torch
        return

    mock = MagicMock()
    mock.cuda.is_available.return_value = False
    mock.no_grad.return_value.__enter__ = MagicMock(return_value=None)
    mock.no_grad.return_value.__exit__ = MagicMock(return_value=False)

    saved = sys.modules.get("torch")
    sys.modules["torch"] = mock
    try:
        yield mock
    finally:
        if saved is None:
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = saved


def _make_mask(img_h, img_w, x_min, y_min, x_max, y_max):
    """Create a binary mask with a filled rectangle."""
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    mask[y_min:y_max, x_min:x_max] = 1
    return mask


class _FakeMaskTensor:
    """Fake tensor that supports [0, 0] indexing and .numpy()."""

    def __init__(self, mask_array: np.ndarray) -> None:
        self._mask = mask_array

    def __getitem__(self, idx):
        # Handles masks[0][0, 0] — returns self for any index
        return self

    def numpy(self):
        return self._mask.astype(np.float32)


def _setup_sam_refiner(mock_torch, mask_array, margin=0.05, iou_threshold=0.5):
    """Create a SAMRefiner with mocked model returning a specific mask."""
    from synthdet.annotate.sam_refiner import SAMRefiner

    refiner = SAMRefiner(iou_threshold=iou_threshold, margin=margin)

    # Mock SAM model outputs
    mock_pred_masks = MagicMock()
    mock_pred_masks.cpu.return_value = MagicMock()

    mock_outputs = MagicMock()
    mock_outputs.pred_masks = mock_pred_masks

    mock_model = MagicMock()
    mock_model.return_value = mock_outputs

    # Mock processor: __call__ returns dict with tensors that have .cpu()
    orig_sizes = MagicMock()
    orig_sizes.cpu.return_value = MagicMock()
    reshaped_sizes = MagicMock()
    reshaped_sizes.cpu.return_value = MagicMock()

    mock_proc = MagicMock()
    mock_proc.return_value = {
        "input_ids": MagicMock(),
        "original_sizes": orig_sizes,
        "reshaped_input_sizes": reshaped_sizes,
    }

    # post_process_masks returns [batch_of_masks]
    # Code accesses: masks[0][0, 0].numpy().astype(np.uint8)
    fake_tensor = _FakeMaskTensor(mask_array)
    mock_proc.image_processor.post_process_masks.return_value = [fake_tensor]

    refiner._model = mock_model
    refiner._processor = mock_proc
    refiner._resolved_device = "cpu"

    return refiner


class TestSAMRefiner:
    """Unit tests for SAMRefiner."""

    def test_refine_basic(self, mock_torch):
        """Basic refinement with a mask close to the original bbox."""
        # 100x100 image, original bbox covers center 40x40
        mask = _make_mask(100, 100, 28, 28, 72, 72)
        refiner = _setup_sam_refiner(mock_torch, mask, margin=0.0, iou_threshold=0.1)

        original = BBox(
            class_id=0, x_center=0.5, y_center=0.5,
            width=0.4, height=0.4,
            confidence=0.9, source=AnnotationSource.grounding_dino,
        )

        result = refiner.refine(
            np.zeros((100, 100, 3), dtype=np.uint8), [original]
        )
        assert len(result) == 1
        # With sufficient IoU, should get sam_refined source
        assert result[0].source == AnnotationSource.sam_refined

    def test_iou_threshold_keeps_original(self, mock_torch):
        """When IoU is below threshold, original bbox is kept."""
        # Mask far from original bbox → low IoU
        mask = _make_mask(100, 100, 0, 0, 10, 10)  # top-left corner
        refiner = _setup_sam_refiner(mock_torch, mask, margin=0.0, iou_threshold=0.9)

        original = BBox(
            class_id=0, x_center=0.5, y_center=0.5,
            width=0.4, height=0.4,
            confidence=0.9, source=AnnotationSource.grounding_dino,
        )

        result = refiner.refine(
            np.zeros((100, 100, 3), dtype=np.uint8), [original]
        )
        assert len(result) == 1
        assert result[0].source == AnnotationSource.grounding_dino  # kept original

    def test_annotation_source_sam_refined(self, mock_torch):
        """Accepted refined bboxes should have source=sam_refined."""
        mask = _make_mask(100, 100, 25, 25, 75, 75)
        refiner = _setup_sam_refiner(mock_torch, mask, margin=0.0, iou_threshold=0.1)

        original = BBox(
            class_id=0, x_center=0.5, y_center=0.5,
            width=0.5, height=0.5,
            confidence=0.8, source=AnnotationSource.grounding_dino,
        )

        result = refiner.refine(
            np.zeros((100, 100, 3), dtype=np.uint8), [original]
        )
        assert result[0].source == AnnotationSource.sam_refined

    def test_empty_input(self, mock_torch):
        """Empty bbox list should return empty list without loading model."""
        from synthdet.annotate.sam_refiner import SAMRefiner

        refiner = SAMRefiner()
        result = refiner.refine(np.zeros((64, 64, 3), dtype=np.uint8), [])
        assert result == []
        assert refiner._model is None  # Model shouldn't have been loaded

    def test_lazy_load(self):
        """Model should not be loaded at construction."""
        from synthdet.annotate.sam_refiner import SAMRefiner

        refiner = SAMRefiner()
        assert refiner._model is None
        assert refiner._processor is None

    def test_missing_torch_error(self):
        """Clear error when torch is not installed."""
        from synthdet.annotate.sam_refiner import SAMRefiner

        refiner = SAMRefiner()

        saved = sys.modules.get("torch")
        sys.modules["torch"] = None  # type: ignore[assignment]
        try:
            bbox = BBox(class_id=0, x_center=0.5, y_center=0.5, width=0.1, height=0.1)
            with pytest.raises(ImportError, match="PyTorch is required"):
                refiner.refine(np.zeros((64, 64, 3), dtype=np.uint8), [bbox])
        finally:
            if saved is None:
                sys.modules.pop("torch", None)
            else:
                sys.modules["torch"] = saved

    def test_preserves_class_id(self, mock_torch):
        """Refined bbox should preserve the original class_id."""
        mask = _make_mask(100, 100, 25, 25, 75, 75)
        refiner = _setup_sam_refiner(mock_torch, mask, margin=0.0, iou_threshold=0.1)

        original = BBox(
            class_id=2, x_center=0.5, y_center=0.5,
            width=0.5, height=0.5,
            confidence=0.85, source=AnnotationSource.grounding_dino,
        )

        result = refiner.refine(
            np.zeros((100, 100, 3), dtype=np.uint8), [original]
        )
        assert result[0].class_id == 2

    def test_preserves_confidence(self, mock_torch):
        """Refined bbox should preserve the original confidence."""
        mask = _make_mask(100, 100, 25, 25, 75, 75)
        refiner = _setup_sam_refiner(mock_torch, mask, margin=0.0, iou_threshold=0.1)

        original = BBox(
            class_id=0, x_center=0.5, y_center=0.5,
            width=0.5, height=0.5,
            confidence=0.77, source=AnnotationSource.grounding_dino,
        )

        result = refiner.refine(
            np.zeros((100, 100, 3), dtype=np.uint8), [original]
        )
        assert result[0].confidence == pytest.approx(0.77)

    def test_margin_expands_bbox(self, mock_torch):
        """Margin should expand the mask-derived bbox."""
        # Tight mask at center
        mask = _make_mask(100, 100, 40, 40, 60, 60)  # 20x20 center
        refiner_no_margin = _setup_sam_refiner(
            mock_torch, mask, margin=0.0, iou_threshold=0.01
        )
        refiner_with_margin = _setup_sam_refiner(
            mock_torch, mask, margin=0.1, iou_threshold=0.01
        )

        original = BBox(
            class_id=0, x_center=0.5, y_center=0.5,
            width=0.2, height=0.2,
            confidence=0.9, source=AnnotationSource.grounding_dino,
        )

        img = np.zeros((100, 100, 3), dtype=np.uint8)
        result_no = refiner_no_margin.refine(img, [original])
        result_with = refiner_with_margin.refine(img, [original])

        # With margin, bbox should be wider/taller
        assert result_with[0].width > result_no[0].width
        assert result_with[0].height > result_no[0].height

    def test_device_auto(self, mock_torch):
        """Auto device resolves to cpu when CUDA unavailable."""
        from synthdet.annotate.sam_refiner import SAMRefiner

        refiner = SAMRefiner(device="auto")
        mock_torch.cuda.is_available.return_value = False
        assert refiner._resolve_device() == "cpu"

"""Tests for synthdet.annotate.grounding_dino.

All tests are mock-based — no real model loading or GPU required.
torch and transformers are mocked via sys.modules when not installed.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from synthdet.annotate.base import Annotator
from synthdet.types import AnnotationSource

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


def _make_mock_results(boxes, scores, labels, img_h=64, img_w=64):
    """Create mock post-processed detection results."""
    return [
        {
            "boxes": boxes,
            "scores": scores,
            "labels": labels,
        }
    ]


class TestGroundingDINOAnnotator:
    """Unit tests for GroundingDINOAnnotator."""

    def test_satisfies_protocol(self):
        from synthdet.annotate.grounding_dino import GroundingDINOAnnotator

        annotator = GroundingDINOAnnotator()
        assert isinstance(annotator, Annotator)

    def test_lazy_load(self):
        """Model should not be loaded at construction."""
        from synthdet.annotate.grounding_dino import GroundingDINOAnnotator

        annotator = GroundingDINOAnnotator()
        assert annotator._model is None
        assert annotator._processor is None

    def test_bgr_to_rgb_conversion(self, mock_torch):
        """Verify BGR input is converted to RGB for the model."""
        from synthdet.annotate.grounding_dino import GroundingDINOAnnotator

        annotator = GroundingDINOAnnotator()

        # Pure blue in BGR (255,0,0)
        bgr = np.zeros((64, 64, 3), dtype=np.uint8)
        bgr[:, :, 0] = 255  # Blue channel in BGR

        received_images = []

        def capture_processor(images=None, text=None, return_tensors=None, **kw):
            received_images.append(np.array(images))
            return {"input_ids": MagicMock()}

        mock_proc = MagicMock(side_effect=capture_processor)
        mock_proc.post_process_grounded_object_detection.return_value = _make_mock_results(
            [], [], []
        )
        mock_model = MagicMock()
        mock_model.return_value = MagicMock()

        annotator._model = mock_model
        annotator._processor = mock_proc
        annotator._resolved_device = "cpu"

        annotator.annotate(bgr, ["scratch"])

        assert len(received_images) == 1
        # After BGR→RGB, blue channel (idx 0 in BGR) should be in idx 2 (RGB)
        assert received_images[0][:, :, 0].mean() < 10  # R ~0
        assert received_images[0][:, :, 2].mean() > 245  # B ~255

    def test_confidence_filtering(self, mock_torch):
        """Detections below confidence threshold should be filtered out."""
        from synthdet.annotate.grounding_dino import GroundingDINOAnnotator

        annotator = GroundingDINOAnnotator(confidence_threshold=0.5)

        mock_proc = MagicMock()
        mock_proc.return_value = {"input_ids": MagicMock()}
        mock_proc.post_process_grounded_object_detection.return_value = _make_mock_results(
            boxes=[[10, 10, 50, 50], [5, 5, 20, 20]],
            scores=[0.8, 0.3],  # second below threshold
            labels=["scratch", "scratch"],
        )
        mock_model = MagicMock()

        annotator._model = mock_model
        annotator._processor = mock_proc
        annotator._resolved_device = "cpu"

        bboxes = annotator.annotate(np.zeros((64, 64, 3), dtype=np.uint8), ["scratch"])
        assert len(bboxes) == 1
        assert bboxes[0].confidence == pytest.approx(0.8)

    def test_prompt_format(self, mock_torch):
        """GDINO text prompt should be '. '-separated class names ending with '.'."""
        from synthdet.annotate.grounding_dino import GroundingDINOAnnotator

        annotator = GroundingDINOAnnotator()

        received_texts = []

        def capture_processor(images=None, text=None, return_tensors=None, **kw):
            received_texts.append(text)
            return {"input_ids": MagicMock()}

        mock_proc = MagicMock(side_effect=capture_processor)
        mock_proc.post_process_grounded_object_detection.return_value = _make_mock_results(
            [], [], []
        )
        mock_model = MagicMock()

        annotator._model = mock_model
        annotator._processor = mock_proc
        annotator._resolved_device = "cpu"

        annotator.annotate(np.zeros((64, 64, 3), dtype=np.uint8), ["scratch", "stain"])

        assert received_texts[0] == "scratch. stain."

    def test_bbox_yolo_conversion(self, mock_torch):
        """Pixel boxes should be converted to normalized YOLO format."""
        from synthdet.annotate.grounding_dino import GroundingDINOAnnotator

        annotator = GroundingDINOAnnotator(confidence_threshold=0.1)

        # 100x100 image, box from (20,30) to (60,70)
        mock_proc = MagicMock()
        mock_proc.return_value = {"input_ids": MagicMock()}
        mock_proc.post_process_grounded_object_detection.return_value = _make_mock_results(
            boxes=[[20, 30, 60, 70]],
            scores=[0.9],
            labels=["scratch"],
        )
        mock_model = MagicMock()

        annotator._model = mock_model
        annotator._processor = mock_proc
        annotator._resolved_device = "cpu"

        bboxes = annotator.annotate(np.zeros((100, 100, 3), dtype=np.uint8), ["scratch"])
        assert len(bboxes) == 1
        bbox = bboxes[0]
        assert bbox.x_center == pytest.approx(0.4, abs=0.01)
        assert bbox.y_center == pytest.approx(0.5, abs=0.01)
        assert bbox.width == pytest.approx(0.4, abs=0.01)
        assert bbox.height == pytest.approx(0.4, abs=0.01)

    def test_annotation_source(self, mock_torch):
        """All produced bboxes should have source=grounding_dino."""
        from synthdet.annotate.grounding_dino import GroundingDINOAnnotator

        annotator = GroundingDINOAnnotator(confidence_threshold=0.1)

        mock_proc = MagicMock()
        mock_proc.return_value = {"input_ids": MagicMock()}
        mock_proc.post_process_grounded_object_detection.return_value = _make_mock_results(
            boxes=[[10, 10, 50, 50]],
            scores=[0.9],
            labels=["scratch"],
        )
        mock_model = MagicMock()

        annotator._model = mock_model
        annotator._processor = mock_proc
        annotator._resolved_device = "cpu"

        bboxes = annotator.annotate(np.zeros((64, 64, 3), dtype=np.uint8), ["scratch"])
        assert bboxes[0].source == AnnotationSource.grounding_dino

    def test_empty_detections(self, mock_torch):
        """No detections should return empty list."""
        from synthdet.annotate.grounding_dino import GroundingDINOAnnotator

        annotator = GroundingDINOAnnotator()

        mock_proc = MagicMock()
        mock_proc.return_value = {"input_ids": MagicMock()}
        mock_proc.post_process_grounded_object_detection.return_value = _make_mock_results(
            [], [], []
        )
        mock_model = MagicMock()

        annotator._model = mock_model
        annotator._processor = mock_proc
        annotator._resolved_device = "cpu"

        bboxes = annotator.annotate(np.zeros((64, 64, 3), dtype=np.uint8), ["scratch"])
        assert bboxes == []

    def test_missing_torch_error(self):
        """Clear error when torch is not installed."""
        from synthdet.annotate.grounding_dino import GroundingDINOAnnotator

        annotator = GroundingDINOAnnotator()

        saved = sys.modules.get("torch")
        sys.modules["torch"] = None  # type: ignore[assignment]
        try:
            with pytest.raises(ImportError, match="PyTorch is required"):
                annotator.annotate(np.zeros((64, 64, 3), dtype=np.uint8), ["scratch"])
        finally:
            if saved is None:
                sys.modules.pop("torch", None)
            else:
                sys.modules["torch"] = saved

    def test_device_auto_cuda(self, mock_torch):
        """Auto device resolves to cuda when available."""
        from synthdet.annotate.grounding_dino import GroundingDINOAnnotator

        annotator = GroundingDINOAnnotator(device="auto")
        mock_torch.cuda.is_available.return_value = True
        assert annotator._resolve_device() == "cuda"

    def test_device_auto_cpu(self, mock_torch):
        """Auto device resolves to cpu when CUDA unavailable."""
        from synthdet.annotate.grounding_dino import GroundingDINOAnnotator

        annotator = GroundingDINOAnnotator(device="auto")
        mock_torch.cuda.is_available.return_value = False
        assert annotator._resolve_device() == "cpu"

    def test_multiple_classes(self, mock_torch):
        """Label-to-class mapping should work for multiple classes."""
        from synthdet.annotate.grounding_dino import GroundingDINOAnnotator

        annotator = GroundingDINOAnnotator(confidence_threshold=0.1)

        mock_proc = MagicMock()
        mock_proc.return_value = {"input_ids": MagicMock()}
        mock_proc.post_process_grounded_object_detection.return_value = _make_mock_results(
            boxes=[[10, 10, 30, 30], [40, 40, 60, 60]],
            scores=[0.9, 0.8],
            labels=["scratch", "stain"],
        )
        mock_model = MagicMock()

        annotator._model = mock_model
        annotator._processor = mock_proc
        annotator._resolved_device = "cpu"

        bboxes = annotator.annotate(
            np.zeros((64, 64, 3), dtype=np.uint8), ["scratch", "stain"]
        )
        assert len(bboxes) == 2
        assert bboxes[0].class_id == 0  # scratch
        assert bboxes[1].class_id == 1  # stain

    def test_confidence_on_bbox(self, mock_torch):
        """Model confidence should be preserved on the BBox object."""
        from synthdet.annotate.grounding_dino import GroundingDINOAnnotator

        annotator = GroundingDINOAnnotator(confidence_threshold=0.1)

        mock_proc = MagicMock()
        mock_proc.return_value = {"input_ids": MagicMock()}
        mock_proc.post_process_grounded_object_detection.return_value = _make_mock_results(
            boxes=[[10, 10, 50, 50]],
            scores=[0.75],
            labels=["scratch"],
        )
        mock_model = MagicMock()

        annotator._model = mock_model
        annotator._processor = mock_proc
        annotator._resolved_device = "cpu"

        bboxes = annotator.annotate(np.zeros((64, 64, 3), dtype=np.uint8), ["scratch"])
        assert bboxes[0].confidence == pytest.approx(0.75)


class TestLabelMapping:
    """Tests for the _map_label_to_class_id helper."""

    def test_exact_match(self):
        from synthdet.annotate.grounding_dino import _map_label_to_class_id

        assert _map_label_to_class_id("scratch", ["scratch", "stain"]) == 0

    def test_containment_match(self):
        from synthdet.annotate.grounding_dino import _map_label_to_class_id

        assert _map_label_to_class_id("deep scratch", ["scratch", "stain"]) == 0

    def test_reverse_containment(self):
        from synthdet.annotate.grounding_dino import _map_label_to_class_id

        assert _map_label_to_class_id("scr", ["scratch", "stain"]) == 0

    def test_no_match_returns_zero(self):
        from synthdet.annotate.grounding_dino import _map_label_to_class_id

        assert _map_label_to_class_id("unknown", ["scratch", "stain"]) == 0

    def test_case_insensitive(self):
        from synthdet.annotate.grounding_dino import _map_label_to_class_id

        assert _map_label_to_class_id("SCRATCH", ["scratch", "stain"]) == 0

"""Tests for synthdet.annotate.owlvit.

All tests are mock-based — no real model loading or GPU required.
"""

from __future__ import annotations

import sys
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
    mock.tensor.return_value = MagicMock()

    saved = sys.modules.get("torch")
    sys.modules["torch"] = mock
    try:
        yield mock
    finally:
        if saved is None:
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = saved


def _make_mock_results(boxes, scores, labels):
    """Create mock post-processed detection results."""
    return [
        {
            "boxes": boxes,
            "scores": scores,
            "labels": labels,
        }
    ]


class TestOWLViTAnnotator:
    """Unit tests for OWLViTAnnotator."""

    def test_satisfies_protocol(self):
        from synthdet.annotate.owlvit import OWLViTAnnotator

        annotator = OWLViTAnnotator()
        assert isinstance(annotator, Annotator)

    def test_lazy_load(self):
        """Model should not be loaded at construction."""
        from synthdet.annotate.owlvit import OWLViTAnnotator

        annotator = OWLViTAnnotator()
        assert annotator._model is None
        assert annotator._processor is None

    def test_bgr_to_rgb_conversion(self, mock_torch):
        """Verify BGR input is converted to RGB for the model."""
        from synthdet.annotate.owlvit import OWLViTAnnotator

        annotator = OWLViTAnnotator()

        bgr = np.zeros((64, 64, 3), dtype=np.uint8)
        bgr[:, :, 0] = 255  # Blue in BGR

        received_images = []

        def capture_processor(text=None, images=None, return_tensors=None, **kw):
            received_images.append(np.array(images))
            return {"input_ids": MagicMock()}

        mock_proc = MagicMock(side_effect=capture_processor)
        mock_proc.post_process_object_detection.return_value = _make_mock_results(
            [], [], []
        )
        mock_model = MagicMock()

        annotator._model = mock_model
        annotator._processor = mock_proc
        annotator._resolved_device = "cpu"

        annotator.annotate(bgr, ["scratch"])

        assert len(received_images) == 1
        assert received_images[0][:, :, 0].mean() < 10  # R ~0
        assert received_images[0][:, :, 2].mean() > 245  # B ~255

    def test_confidence_filtering(self, mock_torch):
        """Detections below confidence threshold should be filtered out."""
        from synthdet.annotate.owlvit import OWLViTAnnotator

        annotator = OWLViTAnnotator(confidence_threshold=0.5)

        mock_proc = MagicMock()
        mock_proc.return_value = {"input_ids": MagicMock()}
        mock_proc.post_process_object_detection.return_value = _make_mock_results(
            boxes=[[10, 10, 50, 50], [5, 5, 20, 20]],
            scores=[0.8, 0.3],
            labels=[0, 0],
        )
        mock_model = MagicMock()

        annotator._model = mock_model
        annotator._processor = mock_proc
        annotator._resolved_device = "cpu"

        bboxes = annotator.annotate(np.zeros((64, 64, 3), dtype=np.uint8), ["scratch"])
        assert len(bboxes) == 1
        assert bboxes[0].confidence == pytest.approx(0.8)

    def test_bbox_yolo_conversion(self, mock_torch):
        """Pixel boxes should be converted to normalized YOLO format."""
        from synthdet.annotate.owlvit import OWLViTAnnotator

        annotator = OWLViTAnnotator(confidence_threshold=0.1)

        mock_proc = MagicMock()
        mock_proc.return_value = {"input_ids": MagicMock()}
        mock_proc.post_process_object_detection.return_value = _make_mock_results(
            boxes=[[20, 30, 60, 70]],
            scores=[0.9],
            labels=[0],
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
        """All produced bboxes should have source=owl_vit."""
        from synthdet.annotate.owlvit import OWLViTAnnotator

        annotator = OWLViTAnnotator(confidence_threshold=0.1)

        mock_proc = MagicMock()
        mock_proc.return_value = {"input_ids": MagicMock()}
        mock_proc.post_process_object_detection.return_value = _make_mock_results(
            boxes=[[10, 10, 50, 50]],
            scores=[0.9],
            labels=[0],
        )
        mock_model = MagicMock()

        annotator._model = mock_model
        annotator._processor = mock_proc
        annotator._resolved_device = "cpu"

        bboxes = annotator.annotate(np.zeros((64, 64, 3), dtype=np.uint8), ["scratch"])
        assert bboxes[0].source == AnnotationSource.owl_vit

    def test_empty_detections(self, mock_torch):
        """No detections should return empty list."""
        from synthdet.annotate.owlvit import OWLViTAnnotator

        annotator = OWLViTAnnotator()

        mock_proc = MagicMock()
        mock_proc.return_value = {"input_ids": MagicMock()}
        mock_proc.post_process_object_detection.return_value = _make_mock_results(
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
        from synthdet.annotate.owlvit import OWLViTAnnotator

        annotator = OWLViTAnnotator()

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

    def test_text_as_list_of_lists(self, mock_torch):
        """OWL-ViT expects text as [class_names], not joined string."""
        from synthdet.annotate.owlvit import OWLViTAnnotator

        annotator = OWLViTAnnotator()

        received_texts = []

        def capture_processor(text=None, images=None, return_tensors=None, **kw):
            received_texts.append(text)
            return {"input_ids": MagicMock()}

        mock_proc = MagicMock(side_effect=capture_processor)
        mock_proc.post_process_object_detection.return_value = _make_mock_results(
            [], [], []
        )
        mock_model = MagicMock()

        annotator._model = mock_model
        annotator._processor = mock_proc
        annotator._resolved_device = "cpu"

        annotator.annotate(
            np.zeros((64, 64, 3), dtype=np.uint8), ["scratch", "stain"]
        )

        assert received_texts[0] == [["scratch", "stain"]]

    def test_label_index_to_class_id(self, mock_torch):
        """OWL-ViT integer labels should map directly to class_id."""
        from synthdet.annotate.owlvit import OWLViTAnnotator

        annotator = OWLViTAnnotator(confidence_threshold=0.1)

        mock_proc = MagicMock()
        mock_proc.return_value = {"input_ids": MagicMock()}
        mock_proc.post_process_object_detection.return_value = _make_mock_results(
            boxes=[[10, 10, 30, 30], [40, 40, 60, 60]],
            scores=[0.9, 0.8],
            labels=[0, 1],
        )
        mock_model = MagicMock()

        annotator._model = mock_model
        annotator._processor = mock_proc
        annotator._resolved_device = "cpu"

        bboxes = annotator.annotate(
            np.zeros((64, 64, 3), dtype=np.uint8), ["scratch", "stain"]
        )
        assert bboxes[0].class_id == 0
        assert bboxes[1].class_id == 1

    def test_device_auto(self, mock_torch):
        """Auto device resolves to cpu when CUDA unavailable."""
        from synthdet.annotate.owlvit import OWLViTAnnotator

        annotator = OWLViTAnnotator(device="auto")
        mock_torch.cuda.is_available.return_value = False
        assert annotator._resolve_device() == "cpu"

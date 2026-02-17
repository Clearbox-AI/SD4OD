"""Tests for synthdet.annotate.verifier.

All tests are mock-based — no real model loading or GPU required.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from synthdet.annotate.base import AnnotationVerifier
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

    # sigmoid returns a mock with indexing support
    def mock_sigmoid(x):
        result = MagicMock()
        result.__getitem__ = lambda self, idx: MagicMock(
            __float__=lambda self: 0.85
        )
        return result

    mock.sigmoid = mock_sigmoid

    saved = sys.modules.get("torch")
    sys.modules["torch"] = mock
    try:
        yield mock
    finally:
        if saved is None:
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = saved


def _setup_verifier(mock_torch, score=0.85):
    """Create a CLIPVerifier with mocked model returning a specific score."""
    from synthdet.annotate.verifier import CLIPVerifier

    verifier = CLIPVerifier(class_names=["scratch", "stain"])

    mock_outputs = MagicMock()
    # logits_per_image[0, 0] → score for sigmoid
    logits_mock = MagicMock()
    mock_outputs.logits_per_image = logits_mock

    mock_model = MagicMock()
    mock_model.return_value = mock_outputs

    mock_proc = MagicMock()
    mock_proc.return_value = {"input_ids": MagicMock(), "pixel_values": MagicMock()}

    verifier._model = mock_model
    verifier._processor = mock_proc
    verifier._resolved_device = "cpu"

    return verifier


class TestCLIPVerifier:
    """Unit tests for CLIPVerifier."""

    def test_satisfies_protocol(self):
        from synthdet.annotate.verifier import CLIPVerifier

        verifier = CLIPVerifier(class_names=["scratch"])
        assert isinstance(verifier, AnnotationVerifier)

    def test_crop_extraction(self, mock_torch):
        """Verifier should crop the bbox region from the image."""
        from synthdet.annotate.verifier import CLIPVerifier

        verifier = _setup_verifier(mock_torch)

        received_images = []

        def capture_processor(text=None, images=None, return_tensors=None, **kw):
            received_images.append(images)
            return {"input_ids": MagicMock(), "pixel_values": MagicMock()}

        verifier._processor = MagicMock(side_effect=capture_processor)

        # 100x100 image, bbox at center covering 40x40
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[30:70, 30:70] = 128  # Mark the crop region

        bbox = BBox(
            class_id=0, x_center=0.5, y_center=0.5,
            width=0.4, height=0.4,
            source=AnnotationSource.grounding_dino,
        )

        verifier.verify(img, [bbox])

        assert len(received_images) == 1
        crop = received_images[0]
        # Crop should be approximately 40x40 pixels
        assert abs(crop.size[0] - 40) <= 2
        assert abs(crop.size[1] - 40) <= 2

    def test_scoring(self, mock_torch):
        """Each bbox should get a score between 0 and 1."""
        verifier = _setup_verifier(mock_torch)

        bbox = BBox(
            class_id=0, x_center=0.5, y_center=0.5,
            width=0.2, height=0.2,
            source=AnnotationSource.grounding_dino,
        )

        results = verifier.verify(
            np.zeros((64, 64, 3), dtype=np.uint8), [bbox]
        )
        assert len(results) == 1
        returned_bbox, score = results[0]
        assert returned_bbox is bbox
        assert 0.0 <= score <= 1.0

    def test_empty_input(self):
        """Empty bbox list should return empty list without loading model."""
        from synthdet.annotate.verifier import CLIPVerifier

        verifier = CLIPVerifier(class_names=["scratch"])
        results = verifier.verify(np.zeros((64, 64, 3), dtype=np.uint8), [])
        assert results == []
        assert verifier._model is None  # Model shouldn't have been loaded

    def test_lazy_load(self):
        """Model should not be loaded at construction."""
        from synthdet.annotate.verifier import CLIPVerifier

        verifier = CLIPVerifier(class_names=["scratch"])
        assert verifier._model is None
        assert verifier._processor is None

    def test_missing_torch_error(self):
        """Clear error when torch is not installed."""
        from synthdet.annotate.verifier import CLIPVerifier

        verifier = CLIPVerifier(class_names=["scratch"])

        saved = sys.modules.get("torch")
        sys.modules["torch"] = None  # type: ignore[assignment]
        try:
            bbox = BBox(class_id=0, x_center=0.5, y_center=0.5, width=0.1, height=0.1)
            with pytest.raises(ImportError, match="PyTorch is required"):
                verifier.verify(np.zeros((64, 64, 3), dtype=np.uint8), [bbox])
        finally:
            if saved is None:
                sys.modules.pop("torch", None)
            else:
                sys.modules["torch"] = saved

    def test_multiple_bboxes(self, mock_torch):
        """Verifier should return one (bbox, score) per input bbox."""
        verifier = _setup_verifier(mock_torch)

        bboxes = [
            BBox(class_id=0, x_center=0.3, y_center=0.3, width=0.1, height=0.1),
            BBox(class_id=1, x_center=0.7, y_center=0.7, width=0.1, height=0.1),
        ]

        results = verifier.verify(
            np.zeros((64, 64, 3), dtype=np.uint8), bboxes
        )
        assert len(results) == 2
        assert results[0][0] is bboxes[0]
        assert results[1][0] is bboxes[1]

    def test_class_name_lookup(self):
        """Class name should be looked up by class_id."""
        from synthdet.annotate.verifier import CLIPVerifier

        verifier = CLIPVerifier(class_names=["scratch", "stain", "crack"])
        assert verifier._get_class_name(0) == "scratch"
        assert verifier._get_class_name(1) == "stain"
        assert verifier._get_class_name(2) == "crack"

    def test_class_name_fallback(self):
        """Out-of-range class_id should get generic fallback."""
        from synthdet.annotate.verifier import CLIPVerifier

        verifier = CLIPVerifier(class_names=["scratch"])
        assert verifier._get_class_name(5) == "class 5"

    def test_device_auto(self, mock_torch):
        """Auto device resolves to cpu when CUDA unavailable."""
        from synthdet.annotate.verifier import CLIPVerifier

        verifier = CLIPVerifier(device="auto", class_names=["scratch"])
        mock_torch.cuda.is_available.return_value = False
        assert verifier._resolve_device() == "cpu"

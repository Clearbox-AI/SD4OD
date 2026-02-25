"""Tests for synthdet.training.evaluator."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from synthdet.config import ActiveLearningConfig
from synthdet.training.evaluator import ModelEvaluator, _compute_iou, _extract_predictions
from synthdet.types import (
    AnnotationSource,
    BBox,
    BBoxSizeBucket,
    Dataset,
    ImageRecord,
    SpatialRegion,
)


# ---------------------------------------------------------------------------
# _compute_iou tests
# ---------------------------------------------------------------------------


class TestComputeIoU:
    def test_full_overlap(self):
        box = (10.0, 10.0, 50.0, 50.0)
        assert _compute_iou(box, box) == pytest.approx(1.0)

    def test_no_overlap(self):
        a = (0.0, 0.0, 10.0, 10.0)
        b = (20.0, 20.0, 30.0, 30.0)
        assert _compute_iou(a, b) == pytest.approx(0.0)

    def test_partial_overlap(self):
        a = (0.0, 0.0, 20.0, 20.0)
        b = (10.0, 10.0, 30.0, 30.0)
        # Intersection: (10,10)-(20,20) = 10*10 = 100
        # Union: 400 + 400 - 100 = 700
        assert _compute_iou(a, b) == pytest.approx(100.0 / 700.0)

    def test_zero_area_box(self):
        a = (10.0, 10.0, 10.0, 10.0)  # zero-area
        b = (0.0, 0.0, 20.0, 20.0)
        assert _compute_iou(a, b) == pytest.approx(0.0)

    def test_contained_box(self):
        outer = (0.0, 0.0, 100.0, 100.0)
        inner = (25.0, 25.0, 75.0, 75.0)
        # Intersection: 50*50=2500, Union: 10000+2500-2500=10000
        assert _compute_iou(outer, inner) == pytest.approx(2500.0 / 10000.0)

    def test_touching_edges(self):
        a = (0.0, 0.0, 10.0, 10.0)
        b = (10.0, 0.0, 20.0, 10.0)
        assert _compute_iou(a, b) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# _extract_predictions tests
# ---------------------------------------------------------------------------


class TestExtractPredictions:
    def test_empty_result(self):
        result = MagicMock()
        result.boxes = None
        assert _extract_predictions(result, 640, 480) == []

    def test_empty_boxes(self):
        result = MagicMock()
        result.boxes = MagicMock()
        result.boxes.__len__ = lambda s: 0
        assert _extract_predictions(result, 640, 480) == []

    def test_populated_result(self):
        boxes = MagicMock()
        boxes.__len__ = lambda s: 2

        cls_tensor = MagicMock()
        cls_tensor.__getitem__ = lambda s, i: MagicMock(item=lambda: i)
        boxes.cls = cls_tensor

        conf_tensor = MagicMock()
        conf_tensor.__getitem__ = lambda s, i: MagicMock(item=lambda: 0.9 - i * 0.1)
        boxes.conf = conf_tensor

        xyxy_tensor = MagicMock()
        xyxy_data = [[10.0, 20.0, 30.0, 40.0], [50.0, 60.0, 70.0, 80.0]]
        xyxy_tensor.__getitem__ = lambda s, i: MagicMock(tolist=lambda: xyxy_data[i])
        boxes.xyxy = xyxy_tensor

        result = MagicMock()
        result.boxes = boxes

        preds = _extract_predictions(result, 640, 480)
        assert len(preds) == 2
        assert preds[0] == (0, 0.9, 10.0, 20.0, 30.0, 40.0)
        assert preds[1] == (1, 0.8, 50.0, 60.0, 70.0, 80.0)


# ---------------------------------------------------------------------------
# ModelEvaluator tests
# ---------------------------------------------------------------------------


def _make_val_record(
    bboxes: list[BBox],
    img_path: str = "/tmp/test_img.jpg",
) -> ImageRecord:
    """Helper to create a val record with a mock image."""
    import numpy as np

    return ImageRecord(
        image_path=Path(img_path),
        bboxes=bboxes,
        image=np.zeros((480, 640, 3), dtype=np.uint8),
    )


class TestModelEvaluator:
    @patch("synthdet.training.evaluator.ModelEvaluator._compute_fine_grained_metrics")
    def test_evaluate_calls_ultralytics(self, mock_fg):
        """evaluate() loads model, calls val, and returns profile."""
        mock_fg.return_value = (
            {b: 0.5 for b in BBoxSizeBucket},
            {r: 0.5 for r in SpatialRegion},
            {b: 0 for b in BBoxSizeBucket},
            {r: 0 for r in SpatialRegion},
            [],
        )

        mock_model = MagicMock()
        mock_val = MagicMock()
        mock_val.box.map50 = 0.75
        mock_val.box.map = 0.55
        mock_val.box.ap50 = [0.75]
        mock_model.val.return_value = mock_val

        with patch("synthdet.training.evaluator.YOLO", return_value=mock_model, create=True):
            # Need to patch the import inside the method
            import synthdet.training.evaluator as ev_mod
            original = ev_mod.__dict__.get("YOLO")
            try:
                # Patch the import inside evaluate()
                with patch.dict("sys.modules", {"ultralytics": MagicMock(YOLO=lambda p: mock_model)}):
                    evaluator = ModelEvaluator()
                    dataset = Dataset(
                        root=Path("/tmp"),
                        class_names=["scratch"],
                        train=[],
                        valid=[],
                        test=[],
                    )
                    profile = evaluator.evaluate(
                        Path("/tmp/best.pt"), Path("/tmp/data.yaml"), dataset
                    )

                assert profile.overall_map50 == 0.75
                assert profile.overall_map50_95 == 0.55
                assert 0 in profile.per_class_map
            finally:
                if original is not None:
                    ev_mod.__dict__["YOLO"] = original

    def test_evaluate_raises_without_ultralytics(self):
        """evaluate() raises ImportError when ultralytics is missing."""
        import sys
        # Temporarily remove ultralytics from sys.modules
        saved = sys.modules.pop("ultralytics", None)
        try:
            with patch.dict("sys.modules", {"ultralytics": None}):
                evaluator = ModelEvaluator()
                dataset = Dataset(
                    root=Path("/tmp"),
                    class_names=["scratch"],
                    train=[],
                    valid=[],
                    test=[],
                )
                with pytest.raises(ImportError, match="ultralytics"):
                    evaluator.evaluate(
                        Path("/tmp/best.pt"), Path("/tmp/data.yaml"), dataset
                    )
        finally:
            if saved is not None:
                sys.modules["ultralytics"] = saved

    def test_fine_grained_empty_val_set(self):
        """Empty val set produces default recall of 1.0 everywhere."""
        evaluator = ModelEvaluator()
        model = MagicMock()
        per_bucket, per_region, fn_b, fn_r, confusion = (
            evaluator._compute_fine_grained_metrics(model, [])
        )
        # No GT → recall defaults to 1.0
        for b in BBoxSizeBucket:
            assert per_bucket[b] == 1.0
            assert fn_b[b] == 0
        for r in SpatialRegion:
            assert per_region[r] == 1.0
            assert fn_r[r] == 0
        assert confusion == []

    def test_false_negatives_counted(self):
        """When model returns no predictions, all GT count as FN."""
        import numpy as np

        evaluator = ModelEvaluator()
        model = MagicMock()
        model.predict.return_value = []  # no predictions

        bbox = BBox(
            class_id=0, x_center=0.5, y_center=0.5, width=0.1, height=0.1,
            source=AnnotationSource.human,
        )
        record = ImageRecord(
            image_path=Path("/tmp/img.jpg"),
            bboxes=[bbox],
            image=np.zeros((480, 640, 3), dtype=np.uint8),
        )
        # Mock is_file to return True
        with patch.object(Path, "is_file", return_value=True):
            per_bucket, per_region, fn_b, fn_r, confusion = (
                evaluator._compute_fine_grained_metrics(model, [record])
            )

        bucket = bbox.size_bucket
        region = bbox.spatial_region
        assert fn_b[bucket] == 1
        assert fn_r[region] == 1
        assert per_bucket[bucket] == 0.0  # 0 TP / 1 GT

    def test_confusion_pairs_tracked(self):
        """Class mismatch between GT and pred creates confusion pair."""
        import numpy as np

        evaluator = ModelEvaluator(ActiveLearningConfig(iou_threshold=0.3))
        model = MagicMock()

        # Create a prediction result with wrong class
        mock_result = MagicMock()
        mock_boxes = MagicMock()
        mock_boxes.__len__ = lambda s: 1
        mock_boxes.cls = MagicMock(__getitem__=lambda s, i: MagicMock(item=lambda: 1))  # pred class=1
        mock_boxes.conf = MagicMock(__getitem__=lambda s, i: MagicMock(item=lambda: 0.9))
        # Overlapping box in pixel coords (for 640x480 image, 0.5±0.05 → 288-352 x 216-264)
        mock_boxes.xyxy = MagicMock(
            __getitem__=lambda s, i: MagicMock(tolist=lambda: [288.0, 216.0, 352.0, 264.0])
        )
        mock_result.boxes = mock_boxes
        model.predict.return_value = [mock_result]

        # GT has class_id=0, pred has class_id=1
        bbox = BBox(
            class_id=0, x_center=0.5, y_center=0.5, width=0.1, height=0.1,
            source=AnnotationSource.human,
        )
        record = ImageRecord(
            image_path=Path("/tmp/img.jpg"),
            bboxes=[bbox],
            image=np.zeros((480, 640, 3), dtype=np.uint8),
        )

        with patch.object(Path, "is_file", return_value=True):
            _, _, _, _, confusion = evaluator._compute_fine_grained_metrics(model, [record])

        assert len(confusion) == 1
        assert confusion[0] == (0, 1, 1)  # (true_class=0, pred_class=1, count=1)

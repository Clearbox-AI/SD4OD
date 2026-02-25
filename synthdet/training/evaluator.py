"""Model evaluation with per-bucket/region recall breakdown.

Runs inference on the validation set, matches predictions to ground truth
by IoU, and groups results by size bucket and spatial region to produce
a ``ModelPerformanceProfile``.
"""

from __future__ import annotations

import logging
from pathlib import Path

from synthdet.config import ActiveLearningConfig
from synthdet.types import (
    BBox,
    BBoxSizeBucket,
    Dataset,
    ModelPerformanceProfile,
    SpatialRegion,
)
from synthdet.utils.bbox import pascal_voc_to_yolo

logger = logging.getLogger(__name__)


def _compute_iou(
    box_a: tuple[float, float, float, float],
    box_b: tuple[float, float, float, float],
) -> float:
    """Compute IoU between two (x1, y1, x2, y2) pixel-coordinate boxes.

    Returns 0.0 for zero-area or non-overlapping boxes.
    """
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter

    if union <= 0:
        return 0.0
    return inter / union


def _extract_predictions(
    result: object,
    img_w: int,
    img_h: int,
) -> list[tuple[int, float, float, float, float, float]]:
    """Extract (class_id, conf, x1, y1, x2, y2) from an ultralytics result.

    ``result.boxes`` has ``.xyxy``, ``.conf``, and ``.cls`` tensors.
    Returns pixel coordinates.
    """
    preds: list[tuple[int, float, float, float, float, float]] = []
    boxes = result.boxes  # type: ignore[attr-defined]
    if boxes is None or len(boxes) == 0:
        return preds
    for i in range(len(boxes)):
        cls_id = int(boxes.cls[i].item())
        conf = float(boxes.conf[i].item())
        x1, y1, x2, y2 = boxes.xyxy[i].tolist()
        preds.append((cls_id, conf, x1, y1, x2, y2))
    return preds


class ModelEvaluator:
    """Evaluate a trained YOLO model and produce a ``ModelPerformanceProfile``.

    Uses per-bucket/region **recall** (TP / total GT in that bucket) rather
    than full mAP. This directly answers the question the strategy module
    needs: "which buckets does the model miss?"
    """

    def __init__(
        self,
        config: ActiveLearningConfig | None = None,
        device: str = "auto",
    ) -> None:
        self.config = config or ActiveLearningConfig()
        self._device = device

    def evaluate(
        self,
        weights_path: Path,
        data_yaml: Path,
        dataset: Dataset,
    ) -> ModelPerformanceProfile:
        """Run evaluation and return fine-grained performance profile.

        Args:
            weights_path: Path to trained YOLO .pt weights.
            data_yaml: Path to the dataset ``data.yaml``.
            dataset: Loaded Dataset (needed for GT bbox metadata).

        Returns:
            ModelPerformanceProfile with per-bucket/region recall.
        """
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "ultralytics is required for model evaluation. "
                "Install with: pip install ultralytics"
            ) from exc

        model = YOLO(str(weights_path))

        # Run standard validation to get overall metrics
        device = self._device if self._device != "auto" else None
        val_results = model.val(
            data=str(data_yaml),
            device=device,
            verbose=False,
            conf=self.config.confidence_threshold,
            iou=self.config.iou_threshold,
        )

        overall_map50 = float(val_results.box.map50)
        overall_map50_95 = float(val_results.box.map)

        # Per-class mAP from val results
        per_class_map: dict[int, float] = {}
        if hasattr(val_results.box, "ap50") and len(val_results.box.ap50) > 0:
            for i, ap in enumerate(val_results.box.ap50):
                per_class_map[i] = float(ap)

        # Fine-grained per-bucket/region analysis via per-image inference
        val_records = dataset.valid
        per_bucket, per_region, fn_buckets, fn_regions, confusion = (
            self._compute_fine_grained_metrics(model, val_records)
        )

        return ModelPerformanceProfile(
            overall_map50=overall_map50,
            overall_map50_95=overall_map50_95,
            per_class_map=per_class_map,
            per_bucket_map=per_bucket,
            per_region_map=per_region,
            false_negative_buckets=fn_buckets,
            false_negative_regions=fn_regions,
            confusion_pairs=confusion,
        )

    def _compute_fine_grained_metrics(
        self,
        model: object,
        val_records: list,
    ) -> tuple[
        dict[BBoxSizeBucket, float],
        dict[SpatialRegion, float],
        dict[BBoxSizeBucket, int],
        dict[SpatialRegion, int],
        list[tuple[int, int, int]],
    ]:
        """Run per-image inference and compute recall by bucket/region.

        Returns:
            (per_bucket_recall, per_region_recall, fn_buckets, fn_regions, confusion_pairs)
        """
        # Counters: total GT and TP per bucket/region
        gt_per_bucket: dict[BBoxSizeBucket, int] = {b: 0 for b in BBoxSizeBucket}
        tp_per_bucket: dict[BBoxSizeBucket, int] = {b: 0 for b in BBoxSizeBucket}
        gt_per_region: dict[SpatialRegion, int] = {r: 0 for r in SpatialRegion}
        tp_per_region: dict[SpatialRegion, int] = {r: 0 for r in SpatialRegion}
        fn_buckets: dict[BBoxSizeBucket, int] = {b: 0 for b in BBoxSizeBucket}
        fn_regions: dict[SpatialRegion, int] = {r: 0 for r in SpatialRegion}
        confusion_counts: dict[tuple[int, int], int] = {}

        iou_thresh = self.config.iou_threshold
        conf_thresh = self.config.confidence_threshold

        for record in val_records:
            if not record.image_path.is_file():
                continue

            img = record.load_image()
            img_h, img_w = img.shape[:2]

            results = model.predict(  # type: ignore[attr-defined]
                str(record.image_path),
                conf=conf_thresh,
                iou=iou_thresh,
                verbose=False,
            )
            if not results:
                # All GT are false negatives
                for gt_bbox in record.bboxes:
                    gt_per_bucket[gt_bbox.size_bucket] += 1
                    gt_per_region[gt_bbox.spatial_region] += 1
                    fn_buckets[gt_bbox.size_bucket] += 1
                    fn_regions[gt_bbox.spatial_region] += 1
                continue

            preds = _extract_predictions(results[0], img_w, img_h)

            # Match GT to predictions greedily by IoU
            matched_preds: set[int] = set()
            for gt_bbox in record.bboxes:
                bucket = gt_bbox.size_bucket
                region = gt_bbox.spatial_region
                gt_per_bucket[bucket] += 1
                gt_per_region[region] += 1

                # Convert GT to pixel coords
                gt_x1 = (gt_bbox.x_center - gt_bbox.width / 2) * img_w
                gt_y1 = (gt_bbox.y_center - gt_bbox.height / 2) * img_h
                gt_x2 = (gt_bbox.x_center + gt_bbox.width / 2) * img_w
                gt_y2 = (gt_bbox.y_center + gt_bbox.height / 2) * img_h
                gt_box = (gt_x1, gt_y1, gt_x2, gt_y2)

                best_iou = 0.0
                best_idx = -1
                best_pred_cls = -1
                for pi, (pred_cls, pred_conf, px1, py1, px2, py2) in enumerate(preds):
                    if pi in matched_preds:
                        continue
                    iou = _compute_iou(gt_box, (px1, py1, px2, py2))
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = pi
                        best_pred_cls = pred_cls

                if best_iou >= iou_thresh and best_idx >= 0:
                    matched_preds.add(best_idx)
                    if best_pred_cls == gt_bbox.class_id:
                        tp_per_bucket[bucket] += 1
                        tp_per_region[region] += 1
                    else:
                        # Class mismatch — counts as FN + confusion
                        fn_buckets[bucket] += 1
                        fn_regions[region] += 1
                        pair = (gt_bbox.class_id, best_pred_cls)
                        confusion_counts[pair] = confusion_counts.get(pair, 0) + 1
                else:
                    fn_buckets[bucket] += 1
                    fn_regions[region] += 1

        # Compute recall
        per_bucket_recall: dict[BBoxSizeBucket, float] = {}
        for b in BBoxSizeBucket:
            total = gt_per_bucket[b]
            per_bucket_recall[b] = tp_per_bucket[b] / total if total > 0 else 1.0

        per_region_recall: dict[SpatialRegion, float] = {}
        for r in SpatialRegion:
            total = gt_per_region[r]
            per_region_recall[r] = tp_per_region[r] / total if total > 0 else 1.0

        confusion = [(tc, pc, cnt) for (tc, pc), cnt in confusion_counts.items()]

        return per_bucket_recall, per_region_recall, fn_buckets, fn_regions, confusion

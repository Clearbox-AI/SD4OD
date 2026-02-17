"""Bounding box format conversions and YOLO label parsing."""

from __future__ import annotations

from pathlib import Path

from synthdet.types import AnnotationSource, BBox


# ---------------------------------------------------------------------------
# Format conversions
# ---------------------------------------------------------------------------


def yolo_to_pascal_voc(
    bbox: BBox, img_width: int, img_height: int
) -> tuple[int, int, int, int]:
    """Convert YOLO normalized bbox to Pascal VOC (x_min, y_min, x_max, y_max) in pixels."""
    x_min = int((bbox.x_center - bbox.width / 2) * img_width)
    y_min = int((bbox.y_center - bbox.height / 2) * img_height)
    x_max = int((bbox.x_center + bbox.width / 2) * img_width)
    y_max = int((bbox.y_center + bbox.height / 2) * img_height)
    return (x_min, y_min, x_max, y_max)


def pascal_voc_to_yolo(
    x_min: int, y_min: int, x_max: int, y_max: int,
    img_width: int, img_height: int, class_id: int = 0,
    source: AnnotationSource = AnnotationSource.unknown,
) -> BBox:
    """Convert Pascal VOC pixel coords to a YOLO normalized BBox."""
    w = (x_max - x_min) / img_width
    h = (y_max - y_min) / img_height
    xc = (x_min + x_max) / 2 / img_width
    yc = (y_min + y_max) / 2 / img_height
    return BBox(class_id=class_id, x_center=xc, y_center=yc, width=w, height=h, source=source)


def yolo_to_coco(
    bbox: BBox, img_width: int, img_height: int
) -> tuple[int, int, int, int]:
    """Convert YOLO normalized bbox to COCO (x_min, y_min, width, height) in pixels."""
    x_min = int((bbox.x_center - bbox.width / 2) * img_width)
    y_min = int((bbox.y_center - bbox.height / 2) * img_height)
    w = int(bbox.width * img_width)
    h = int(bbox.height * img_height)
    return (x_min, y_min, w, h)


def coco_to_yolo(
    x_min: int, y_min: int, w: int, h: int,
    img_width: int, img_height: int, class_id: int = 0,
    source: AnnotationSource = AnnotationSource.unknown,
) -> BBox:
    """Convert COCO pixel coords to a YOLO normalized BBox."""
    xc = (x_min + w / 2) / img_width
    yc = (y_min + h / 2) / img_height
    nw = w / img_width
    nh = h / img_height
    return BBox(class_id=class_id, x_center=xc, y_center=yc, width=nw, height=nh, source=source)


# ---------------------------------------------------------------------------
# Bbox utilities
# ---------------------------------------------------------------------------


def clip_bbox(bbox: BBox) -> BBox:
    """Clip bbox coordinates to [0, 1]."""
    x1 = max(0.0, bbox.x_center - bbox.width / 2)
    y1 = max(0.0, bbox.y_center - bbox.height / 2)
    x2 = min(1.0, bbox.x_center + bbox.width / 2)
    y2 = min(1.0, bbox.y_center + bbox.height / 2)
    w = x2 - x1
    h = y2 - y1
    return BBox(
        class_id=bbox.class_id,
        x_center=x1 + w / 2,
        y_center=y1 + h / 2,
        width=w,
        height=h,
        confidence=bbox.confidence,
        source=bbox.source,
    )


def bbox_iou(a: BBox, b: BBox) -> float:
    """Compute IoU between two YOLO-format bboxes."""
    ax1 = a.x_center - a.width / 2
    ay1 = a.y_center - a.height / 2
    ax2 = a.x_center + a.width / 2
    ay2 = a.y_center + a.height / 2

    bx1 = b.x_center - b.width / 2
    by1 = b.y_center - b.height / 2
    bx2 = b.x_center + b.width / 2
    by2 = b.y_center + b.height / 2

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = a.width * a.height
    area_b = b.width * b.height
    union = area_a + area_b - inter

    if union <= 0:
        return 0.0
    return inter / union


def validate_bbox(bbox: BBox, num_classes: int | None = None) -> list[str]:
    """Validate a bbox, returning a list of error messages (empty = valid)."""
    errors: list[str] = []
    if bbox.class_id < 0:
        errors.append(f"Negative class_id: {bbox.class_id}")
    if num_classes is not None and bbox.class_id >= num_classes:
        errors.append(f"class_id {bbox.class_id} >= num_classes {num_classes}")
    if bbox.width <= 0 or bbox.height <= 0:
        errors.append(f"Non-positive dimensions: w={bbox.width}, h={bbox.height}")
    if not (0 <= bbox.x_center <= 1):
        errors.append(f"x_center out of [0,1]: {bbox.x_center}")
    if not (0 <= bbox.y_center <= 1):
        errors.append(f"y_center out of [0,1]: {bbox.y_center}")
    if bbox.width > 1:
        errors.append(f"width > 1: {bbox.width}")
    if bbox.height > 1:
        errors.append(f"height > 1: {bbox.height}")
    return errors


# ---------------------------------------------------------------------------
# YOLO label file parsing
# ---------------------------------------------------------------------------


def parse_yolo_label_file(
    label_path: Path,
    source: AnnotationSource = AnnotationSource.human,
) -> list[BBox]:
    """Parse a YOLO label file into a list of BBox objects.

    Handles: empty files, missing trailing newlines, blank lines, extra whitespace.
    """
    text = label_path.read_text().strip()
    if not text:
        return []
    bboxes: list[BBox] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        bboxes.append(BBox.from_yolo_line(line, source=source))
    return bboxes

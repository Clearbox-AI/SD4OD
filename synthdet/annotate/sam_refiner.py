"""SAM-based bounding box refinement.

Uses facebook/sam-vit-base via HuggingFace transformers to refine
coarse bounding boxes into tighter fits using segmentation masks.
Model is lazy-loaded on first refine() call.
"""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np
from PIL import Image

from synthdet.types import AnnotationSource, BBox
from synthdet.utils.bbox import bbox_iou, clip_bbox, yolo_to_pascal_voc

logger = logging.getLogger(__name__)


class SAMRefiner:
    """Refine bounding boxes using SAM segmentation masks.

    NOT an Annotator — has a different interface (refines existing bboxes
    rather than detecting from scratch).
    """

    def __init__(
        self,
        model: str = "facebook/sam-vit-base",
        device: str = "auto",
        iou_threshold: float = 0.5,
        margin: float = 0.05,
    ) -> None:
        self._model_id = model
        self._device_setting = device
        self._iou_threshold = iou_threshold
        self._margin = margin
        self._model: Any = None
        self._processor: Any = None
        self._resolved_device: str | None = None

    def _resolve_device(self) -> str:
        """Resolve 'auto' to 'cuda' or 'cpu'."""
        if self._device_setting != "auto":
            return self._device_setting
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def _ensure_model(self) -> None:
        """Lazy-load SAM model and processor on first use."""
        if self._model is not None:
            return

        try:
            import torch  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "PyTorch is required for SAMRefiner. "
                "Install with: pip install torch"
            ) from exc

        try:
            from transformers import SamModel, SamProcessor
        except ImportError as exc:
            raise ImportError(
                "transformers is required for SAMRefiner. "
                "Install with: pip install 'synthdet[annotation]'"
            ) from exc

        self._resolved_device = self._resolve_device()
        logger.info(
            "Loading SAM %s on %s", self._model_id, self._resolved_device
        )

        self._processor = SamProcessor.from_pretrained(self._model_id)
        self._model = SamModel.from_pretrained(self._model_id).to(
            self._resolved_device
        )

    def refine(self, image: np.ndarray, bboxes: list[BBox]) -> list[BBox]:
        """Refine bounding boxes using SAM segmentation masks.

        For each bbox, SAM generates a segmentation mask using the bbox
        as a box prompt. The mask contour is used to derive a tighter bbox.
        If the refined bbox has sufficient IoU with the original, it is
        accepted; otherwise the original is kept.

        Args:
            image: BGR image array.
            bboxes: Bounding boxes to refine.

        Returns:
            List of refined (or original) BBox objects.
        """
        if not bboxes:
            return []

        self._ensure_model()

        import torch

        # BGR → RGB → PIL
        rgb = image[:, :, ::-1]
        pil_image = Image.fromarray(rgb)
        img_h, img_w = image.shape[:2]

        refined: list[BBox] = []
        for bbox in bboxes:
            # Convert YOLO bbox to pixel box for SAM prompt
            x_min, y_min, x_max, y_max = yolo_to_pascal_voc(bbox, img_w, img_h)
            input_box = [[[[x_min, y_min, x_max, y_max]]]]

            inputs = self._processor(
                pil_image, input_boxes=input_box, return_tensors="pt"
            )
            inputs = {k: v.to(self._resolved_device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._model(**inputs)

            masks = self._processor.image_processor.post_process_masks(
                outputs.pred_masks.cpu(),
                inputs["original_sizes"].cpu(),
                inputs["reshaped_input_sizes"].cpu(),
            )

            # Take best mask (highest IoU score)
            mask_np = masks[0][0, 0].numpy().astype(np.uint8)

            refined_bbox = self._mask_to_bbox(
                mask_np, bbox, img_w, img_h
            )
            if refined_bbox is not None:
                refined.append(refined_bbox)
            else:
                refined.append(bbox)

        return refined

    def _mask_to_bbox(
        self,
        mask: np.ndarray,
        original: BBox,
        img_w: int,
        img_h: int,
    ) -> BBox | None:
        """Convert a binary mask to a tight bbox with margin.

        Returns the refined bbox if IoU with original exceeds threshold,
        otherwise returns None (caller keeps original).
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        # Merge all contours into one bounding rect
        all_points = np.concatenate(contours)
        rx, ry, rw, rh = cv2.boundingRect(all_points)

        # Add margin
        margin_x = int(self._margin * img_w)
        margin_y = int(self._margin * img_h)
        rx = max(0, rx - margin_x)
        ry = max(0, ry - margin_y)
        rw = min(img_w - rx, rw + 2 * margin_x)
        rh = min(img_h - ry, rh + 2 * margin_y)

        # Convert to YOLO normalized
        xc = (rx + rw / 2) / img_w
        yc = (ry + rh / 2) / img_h
        w = rw / img_w
        h = rh / img_h

        refined = BBox(
            class_id=original.class_id,
            x_center=xc,
            y_center=yc,
            width=w,
            height=h,
            confidence=original.confidence,
            source=AnnotationSource.sam_refined,
        )
        refined = clip_bbox(refined)

        # Accept only if IoU with original is above threshold
        if bbox_iou(original, refined) >= self._iou_threshold:
            return refined
        return None

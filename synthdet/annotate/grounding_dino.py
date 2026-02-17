"""Grounding DINO zero-shot object detection annotator.

Uses IDEA-Research/grounding-dino-tiny via HuggingFace transformers.
Model and processor are lazy-loaded on first annotate() call.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from PIL import Image

from synthdet.types import AnnotationSource, BBox
from synthdet.utils.bbox import clip_bbox

logger = logging.getLogger(__name__)


def _map_label_to_class_id(label: str, class_names: list[str]) -> int:
    """Map a Grounding DINO text label to a class index via fuzzy containment."""
    label_lower = label.lower().strip()
    for i, name in enumerate(class_names):
        if name.lower() in label_lower or label_lower in name.lower():
            return i
    return 0


class GroundingDINOAnnotator:
    """Auto-annotator using Grounding DINO zero-shot object detection.

    Satisfies the ``Annotator`` protocol from ``synthdet.annotate.base``.
    """

    def __init__(
        self,
        model: str = "IDEA-Research/grounding-dino-tiny",
        device: str = "auto",
        confidence_threshold: float = 0.3,
        box_threshold: float = 0.25,
    ) -> None:
        self._model_id = model
        self._device_setting = device
        self._confidence_threshold = confidence_threshold
        self._box_threshold = box_threshold
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
        """Lazy-load model and processor on first use."""
        if self._model is not None:
            return

        try:
            import torch  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "PyTorch is required for GroundingDINOAnnotator. "
                "Install with: pip install torch"
            ) from exc

        try:
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        except ImportError as exc:
            raise ImportError(
                "transformers is required for GroundingDINOAnnotator. "
                "Install with: pip install 'synthdet[annotation]'"
            ) from exc

        self._resolved_device = self._resolve_device()
        logger.info(
            "Loading Grounding DINO %s on %s", self._model_id, self._resolved_device
        )

        self._processor = AutoProcessor.from_pretrained(self._model_id)
        self._model = AutoModelForZeroShotObjectDetection.from_pretrained(
            self._model_id
        ).to(self._resolved_device)

    def annotate(self, image: np.ndarray, class_names: list[str]) -> list[BBox]:
        """Detect objects and return YOLO-format bounding boxes.

        Args:
            image: BGR image array (OpenCV convention).
            class_names: Class names to detect (e.g. ["scratch", "stain"]).

        Returns:
            List of detected BBox objects with source=grounding_dino.
        """
        self._ensure_model()

        # BGR → RGB → PIL
        rgb = image[:, :, ::-1]
        pil_image = Image.fromarray(rgb)
        img_h, img_w = image.shape[:2]

        # Grounding DINO text prompt convention: "class1. class2."
        text_prompt = ". ".join(class_names) + "."

        inputs = self._processor(images=pil_image, text=text_prompt, return_tensors="pt")
        inputs = {k: v.to(self._resolved_device) for k, v in inputs.items()}

        import torch

        with torch.no_grad():
            outputs = self._model(**inputs)

        results = self._processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            box_threshold=self._box_threshold,
            text_threshold=self._confidence_threshold,
            target_sizes=[(img_h, img_w)],
        )[0]

        bboxes: list[BBox] = []
        for box, score, label in zip(
            results["boxes"], results["scores"], results["labels"]
        ):
            conf = float(score)
            if conf < self._confidence_threshold:
                continue

            # box is [x_min, y_min, x_max, y_max] in pixels
            x_min, y_min, x_max, y_max = [float(v) for v in box]
            class_id = _map_label_to_class_id(label, class_names)

            # Convert to YOLO normalized format
            w = (x_max - x_min) / img_w
            h = (y_max - y_min) / img_h
            xc = (x_min + x_max) / 2 / img_w
            yc = (y_min + y_max) / 2 / img_h

            bbox = BBox(
                class_id=class_id,
                x_center=xc,
                y_center=yc,
                width=w,
                height=h,
                confidence=conf,
                source=AnnotationSource.grounding_dino,
            )
            bboxes.append(clip_bbox(bbox))

        return bboxes

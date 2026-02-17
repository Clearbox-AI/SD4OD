"""OWL-ViT zero-shot object detection annotator.

Uses google/owlvit-base-patch32 via HuggingFace transformers.
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


class OWLViTAnnotator:
    """Auto-annotator using OWL-ViT zero-shot object detection.

    Satisfies the ``Annotator`` protocol from ``synthdet.annotate.base``.
    """

    def __init__(
        self,
        model: str = "google/owlvit-base-patch32",
        device: str = "auto",
        confidence_threshold: float = 0.3,
    ) -> None:
        self._model_id = model
        self._device_setting = device
        self._confidence_threshold = confidence_threshold
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
                "PyTorch is required for OWLViTAnnotator. "
                "Install with: pip install torch"
            ) from exc

        try:
            from transformers import OwlViTForObjectDetection, OwlViTProcessor
        except ImportError as exc:
            raise ImportError(
                "transformers is required for OWLViTAnnotator. "
                "Install with: pip install 'synthdet[annotation]'"
            ) from exc

        self._resolved_device = self._resolve_device()
        logger.info(
            "Loading OWL-ViT %s on %s", self._model_id, self._resolved_device
        )

        self._processor = OwlViTProcessor.from_pretrained(self._model_id)
        self._model = OwlViTForObjectDetection.from_pretrained(self._model_id).to(
            self._resolved_device
        )

    def annotate(self, image: np.ndarray, class_names: list[str]) -> list[BBox]:
        """Detect objects and return YOLO-format bounding boxes.

        Args:
            image: BGR image array (OpenCV convention).
            class_names: Class names to detect (e.g. ["scratch", "stain"]).

        Returns:
            List of detected BBox objects with source=owl_vit.
        """
        self._ensure_model()

        # BGR → RGB → PIL
        rgb = image[:, :, ::-1]
        pil_image = Image.fromarray(rgb)
        img_h, img_w = image.shape[:2]

        # OWL-ViT expects text as list-of-lists: [["scratch", "stain"]]
        inputs = self._processor(
            text=[class_names], images=pil_image, return_tensors="pt"
        )
        inputs = {k: v.to(self._resolved_device) for k, v in inputs.items()}

        import torch

        with torch.no_grad():
            outputs = self._model(**inputs)

        target_sizes = torch.tensor([(img_h, img_w)])
        results = self._processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=self._confidence_threshold
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
            # OWL-ViT labels are integer indices matching the input class list
            class_id = int(label)

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
                source=AnnotationSource.owl_vit,
            )
            bboxes.append(clip_bbox(bbox))

        return bboxes

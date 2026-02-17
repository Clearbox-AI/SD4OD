"""CLIP-based annotation quality verifier.

Uses openai/clip-vit-base-patch32 via HuggingFace transformers to score
how well each bbox crop matches its class label.
Model is lazy-loaded on first verify() call.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from PIL import Image

from synthdet.types import BBox
from synthdet.utils.bbox import yolo_to_pascal_voc

logger = logging.getLogger(__name__)


class CLIPVerifier:
    """Verify annotation quality using CLIP similarity scoring.

    Satisfies the ``AnnotationVerifier`` protocol from ``synthdet.annotate.base``.

    Crops each bbox region, encodes it with CLIP alongside the class label text,
    and returns a cosine similarity score (0–1).
    """

    def __init__(
        self,
        model: str = "openai/clip-vit-base-patch32",
        device: str = "auto",
        min_confidence: float = 0.5,
        class_names: list[str] | None = None,
    ) -> None:
        self._model_id = model
        self._device_setting = device
        self._min_confidence = min_confidence
        self._class_names = class_names or []
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
        """Lazy-load CLIP model and processor on first use."""
        if self._model is not None:
            return

        try:
            import torch  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "PyTorch is required for CLIPVerifier. "
                "Install with: pip install torch"
            ) from exc

        try:
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as exc:
            raise ImportError(
                "transformers is required for CLIPVerifier. "
                "Install with: pip install 'synthdet[annotation]'"
            ) from exc

        self._resolved_device = self._resolve_device()
        logger.info(
            "Loading CLIP %s on %s", self._model_id, self._resolved_device
        )

        self._processor = CLIPProcessor.from_pretrained(self._model_id)
        self._model = CLIPModel.from_pretrained(self._model_id).to(
            self._resolved_device
        )

    def verify(
        self, image: np.ndarray, bboxes: list[BBox]
    ) -> list[tuple[BBox, float]]:
        """Score each bbox crop against its class label using CLIP.

        Args:
            image: BGR image array.
            bboxes: Bounding boxes to verify.

        Returns:
            List of (bbox, score) tuples where score is CLIP similarity (0–1).
        """
        if not bboxes:
            return []

        self._ensure_model()

        import torch

        # BGR → RGB → PIL
        rgb = image[:, :, ::-1]
        pil_image = Image.fromarray(rgb)
        img_h, img_w = image.shape[:2]

        results: list[tuple[BBox, float]] = []
        for bbox in bboxes:
            # Crop the bbox region
            x_min, y_min, x_max, y_max = yolo_to_pascal_voc(bbox, img_w, img_h)
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(img_w, x_max)
            y_max = min(img_h, y_max)

            if x_max <= x_min or y_max <= y_min:
                results.append((bbox, 0.0))
                continue

            crop = pil_image.crop((x_min, y_min, x_max, y_max))

            # Get class name for this bbox
            class_name = self._get_class_name(bbox.class_id)
            text = f"a photo of a {class_name}"

            inputs = self._processor(
                text=[text], images=crop, return_tensors="pt", padding=True
            )
            inputs = {k: v.to(self._resolved_device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._model(**inputs)

            # CLIP logits_per_image gives similarity scores
            # Normalize to 0-1 range using sigmoid
            score = float(torch.sigmoid(outputs.logits_per_image[0, 0]))
            results.append((bbox, score))

        return results

    def _get_class_name(self, class_id: int) -> str:
        """Get class name by index, falling back to generic label."""
        if class_id < len(self._class_names):
            return self._class_names[class_id]
        return f"class {class_id}"

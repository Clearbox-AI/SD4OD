"""Local diffusers-based inpainting provider (zero API cost, requires GPU).

Default model: ``runwayml/stable-diffusion-inpainting`` (SD 1.5, ~4 GB VRAM, 512x512).
Switch to SDXL via config for higher quality at ~6 GB VRAM (1024x1024).

Satisfies the ``InpaintingProvider`` protocol defined in
``synthdet.generate.inpainting``.
"""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np
from PIL import Image

from synthdet.generate.errors import InpaintingAPIError

logger = logging.getLogger(__name__)

# Model resolution defaults (width, height)
_MODEL_RESOLUTIONS: dict[str, tuple[int, int]] = {
    "runwayml/stable-diffusion-inpainting": (512, 512),
}
_SDXL_RESOLUTION = (1024, 1024)
_DEFAULT_RESOLUTION = (512, 512)


def _get_model_resolution(model: str) -> tuple[int, int]:
    """Return the native resolution for a model identifier."""
    if model in _MODEL_RESOLUTIONS:
        return _MODEL_RESOLUTIONS[model]
    if "xl" in model.lower():
        return _SDXL_RESOLUTION
    return _DEFAULT_RESOLUTION


class DiffusersInpaintingProvider:
    """Inpainting via HuggingFace ``diffusers`` running locally.

    The pipeline is lazy-loaded on the first ``inpaint()`` call so that
    construction is cheap and the module can be imported without GPU
    dependencies installed.
    """

    def __init__(
        self,
        model: str = "runwayml/stable-diffusion-inpainting",
        device: str = "auto",
        use_fp16: bool = True,
        num_inference_steps: int = 50,
        strength: float = 0.75,
        guidance_scale: float = 7.5,
    ) -> None:
        self._model = model
        self._device_setting = device
        self._use_fp16 = use_fp16
        self._num_inference_steps = num_inference_steps
        self._strength = strength
        self._guidance_scale = guidance_scale
        self._pipeline: Any = None
        self._resolved_device: str | None = None

    @property
    def cost_per_image(self) -> float:
        return 0.0

    def inpaint(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        prompt: str,
        *,
        seed: int | None = None,
        num_images: int = 1,
    ) -> list[np.ndarray]:
        """Run local diffusion inpainting.

        Args:
            image: BGR uint8 numpy array.
            mask: uint8 mask (255 = edit region, 0 = preserve).
            prompt: Text prompt describing the desired content.
            seed: Optional generation seed.
            num_images: Number of images to return.

        Returns:
            List of BGR uint8 numpy arrays matching original dimensions.
        """
        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                "PyTorch is required for the diffusers provider. "
                "Install with: pip install 'synthdet[generation]'"
            ) from exc

        try:
            self._ensure_pipeline()
        except ImportError as exc:
            raise ImportError(
                "diffusers is required for the diffusers provider. "
                "Install with: pip install 'synthdet[generation]'"
            ) from exc

        orig_h, orig_w = image.shape[:2]
        model_w, model_h = _get_model_resolution(self._model)

        # BGR → RGB → PIL, resize to model resolution
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb).resize(
            (model_w, model_h), Image.LANCZOS,
        )

        # uint8 mask → PIL "L", resize to model resolution
        mask_pil = Image.fromarray(mask, mode="L").resize(
            (model_w, model_h), Image.NEAREST,
        )

        # Build generator for deterministic seeding
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self._resolved_device).manual_seed(seed)

        try:
            output = self._pipeline(
                prompt=prompt,
                image=img_pil,
                mask_image=mask_pil,
                num_inference_steps=self._num_inference_steps,
                strength=self._strength,
                guidance_scale=self._guidance_scale,
                num_images_per_prompt=num_images,
                generator=generator,
            )
        except RuntimeError as exc:
            is_oom = "out of memory" in str(exc).lower()
            raise InpaintingAPIError(
                "diffusers", str(exc), retryable=is_oom,
            ) from exc

        # Convert outputs: PIL RGB → resize back to original → BGR np
        results: list[np.ndarray] = []
        for pil_img in output.images:
            pil_img = pil_img.resize((orig_w, orig_h), Image.LANCZOS)
            rgb_arr = np.array(pil_img)
            bgr_arr = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)
            results.append(bgr_arr)
        return results

    def _resolve_device(self) -> str:
        """Resolve 'auto' to 'cuda' or 'cpu'."""
        if self._device_setting != "auto":
            return self._device_setting
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def _ensure_pipeline(self) -> None:
        """Lazy-load the diffusers pipeline on first use."""
        if self._pipeline is not None:
            return

        import torch
        from diffusers import AutoPipelineForInpainting  # type: ignore[import-untyped]

        self._resolved_device = self._resolve_device()
        use_fp16 = self._use_fp16 and self._resolved_device != "cpu"
        dtype = torch.float16 if use_fp16 else torch.float32

        logger.info(
            "Loading diffusers pipeline %s on %s (dtype=%s)",
            self._model, self._resolved_device, dtype,
        )

        self._pipeline = AutoPipelineForInpainting.from_pretrained(
            self._model,
            torch_dtype=dtype,
        ).to(self._resolved_device)

        # Memory optimizations
        self._pipeline.enable_attention_slicing()
        if hasattr(self._pipeline, "enable_vae_slicing"):
            self._pipeline.enable_vae_slicing()

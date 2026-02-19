"""Google Imagen 3 image generation provider via Vertex AI.

Generates isolated defect images on neutral backgrounds, which are then
composited onto target images using Poisson blending. This avoids the
fundamental problem with inpainting (model reconstructs clean surface
instead of generating defects).

Workflow:
    1. Call ``generate_image`` with a prompt describing an isolated defect
    2. Extract the defect region from the generated image via thresholding
    3. Return the defect patch + alpha mask for compositing
"""

from __future__ import annotations

import io
import logging
import os
from typing import Any

import numpy as np
from PIL import Image

from synthdet.generate.errors import InpaintingAPIError

logger = logging.getLogger(__name__)


class ImagenGenerateProvider:
    """Generate isolated defect patches via Google Imagen 3 on Vertex AI.

    Uses ``generate_image`` (not ``edit_image``) to create defect textures
    on neutral backgrounds, then extracts them via background subtraction.

    Authentication: same as ImagenInpaintingProvider (Vertex AI or API key).
    """

    def __init__(
        self,
        model: str = "imagen-4.0-generate-001",
        api_key: str | None = None,
        api_key_env_var: str = "GOOGLE_API_KEY",
        project: str | None = None,
        location: str | None = None,
        bg_color: tuple[int, int, int] = (220, 220, 220),
        threshold_delta: int = 30,
    ) -> None:
        """
        Args:
            model: Imagen model ID for generation.
            bg_color: RGB background color used in prompts and subtraction.
            threshold_delta: Pixel difference threshold for mask extraction.
                Higher = only strong defects extracted. Lower = more sensitive.
        """
        self._model = model
        self._bg_color = bg_color
        self._threshold_delta = threshold_delta
        self._client: Any = None

        resolved_key = api_key or os.environ.get(api_key_env_var)
        self._api_key = resolved_key
        self._project = project or os.environ.get("GOOGLE_CLOUD_PROJECT")
        self._location = location or os.environ.get(
            "GOOGLE_CLOUD_LOCATION", "us-central1"
        )

        if not self._api_key and not self._project:
            raise ValueError(
                f"No credentials provided. Either:\n"
                f"  1. Set {api_key_env_var} env var (API key auth), or\n"
                f"  2. Set GOOGLE_CLOUD_PROJECT env var and run "
                f"'gcloud auth application-default login' (Vertex AI auth)"
            )

    @property
    def cost_per_image(self) -> float:
        return 0.04  # generate is slightly more expensive than edit

    def generate_defect_patch(
        self,
        prompt: str,
        size: tuple[int, int] = (512, 512),
        *,
        seed: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate a defect image and extract it as a patch + mask.

        Args:
            prompt: Describes the defect to generate.
            size: Desired output size (width, height). Not all sizes are
                  supported by Imagen; the result will be resized.
            seed: Optional generation seed.

        Returns:
            (patch_bgr, mask) where:
                patch_bgr: BGR uint8 numpy array of the defect image
                mask: uint8 mask (255 = defect, 0 = background)
        """
        genai = self._get_genai()
        client = self._get_client(genai)

        # Build prompt that asks for isolated defect on neutral background
        bg_desc = f"light gray (RGB {self._bg_color[0]},{self._bg_color[1]},{self._bg_color[2]})"
        full_prompt = (
            f"{prompt}. "
            f"Isolated on a plain {bg_desc} background. "
            f"Top-down view, photorealistic close-up. "
            f"No other objects, just the defect on a flat surface."
        )

        try:
            config_kwargs: dict[str, Any] = {
                "number_of_images": 1,
                "negative_prompt": (
                    "food, bowl, ramen, noodles, text, words, letters, "
                    "people, person, hands, fingers, face, animals, "
                    "furniture, plants, toys, cartoon, drawing, painting, "
                    "watermark, logo, brand name, 3D render, illustration"
                ),
            }

            response = client.models.generate_images(
                model=self._model,
                prompt=full_prompt,
                config=genai.types.GenerateImagesConfig(**config_kwargs),
            )

            if not response.generated_images:
                raise InpaintingAPIError(
                    "imagen_generate", "No images returned", retryable=True
                )

            # Extract PIL image from response
            gen_image = response.generated_images[0]
            pil_img = _extract_pil_image(gen_image.image)
            rgb_arr = np.array(pil_img.convert("RGB"))

            # Resize to desired size
            if rgb_arr.shape[:2] != (size[1], size[0]):
                import cv2
                rgb_arr = cv2.resize(rgb_arr, size, interpolation=cv2.INTER_LINEAR)

            # Convert to BGR
            bgr_arr = rgb_arr[:, :, ::-1].copy()

            # Extract defect mask via background subtraction
            mask = self._extract_mask(rgb_arr)

            logger.debug(
                "Generated defect patch: %dx%d, mask coverage: %.1f%%",
                bgr_arr.shape[1], bgr_arr.shape[0],
                100.0 * np.count_nonzero(mask) / mask.size,
            )

            return bgr_arr, mask

        except InpaintingAPIError:
            raise
        except Exception as exc:
            retryable = _is_retryable(exc)
            raise InpaintingAPIError(
                "imagen_generate", str(exc), retryable=retryable
            ) from exc

    def _extract_mask(self, rgb_image: np.ndarray) -> np.ndarray:
        """Extract defect mask by comparing against expected background color.

        Uses per-channel absolute difference, then thresholds.
        Also applies morphological cleanup to remove noise.
        """
        import cv2

        bg = np.array(self._bg_color, dtype=np.float32)
        diff = np.abs(rgb_image.astype(np.float32) - bg)
        max_diff = diff.max(axis=2)  # max channel difference per pixel

        mask = np.where(max_diff > self._threshold_delta, 255, 0).astype(np.uint8)

        # Morphological cleanup: remove small noise, fill small holes
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        # If mask is nearly empty (< 2% coverage), lower the threshold
        coverage = np.count_nonzero(mask) / mask.size
        if coverage < 0.02:
            logger.debug(
                "Low mask coverage (%.1f%%), trying lower threshold", coverage * 100
            )
            lower_thresh = max(10, self._threshold_delta // 2)
            mask = np.where(max_diff > lower_thresh, 255, 0).astype(np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Feather edges with Gaussian blur for smooth compositing
        ksize = max(3, min(*mask.shape[:2]) // 20) | 1  # ensure odd
        mask = cv2.GaussianBlur(mask, (ksize, ksize), 0)

        return mask

    def _get_genai(self) -> Any:
        try:
            import google.genai as genai  # type: ignore[import-untyped]
            return genai
        except ImportError as exc:
            raise ImportError(
                "google-genai is required for the Imagen provider. "
                "Install it with: pip install 'google-genai>=1.0'"
            ) from exc

    def _get_client(self, genai: Any) -> Any:
        if self._client is None:
            if self._project:
                self._client = genai.Client(
                    vertexai=True,
                    project=self._project,
                    location=self._location,
                )
            else:
                self._client = genai.Client(api_key=self._api_key)
        return self._client


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_pil_image(image_obj: Any) -> Image.Image:
    """Extract a PIL Image from the SDK's Image object."""
    if hasattr(image_obj, "_pil_image") and image_obj._pil_image is not None:
        return image_obj._pil_image
    if hasattr(image_obj, "image_bytes") and image_obj.image_bytes:
        return Image.open(io.BytesIO(image_obj.image_bytes))
    if hasattr(image_obj, "data") and image_obj.data:
        return Image.open(io.BytesIO(image_obj.data))
    raise AttributeError(
        f"Cannot extract PIL image from {type(image_obj).__name__}. "
        f"Available attributes: {[a for a in dir(image_obj) if not a.startswith('__')]}"
    )


def _is_retryable(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(kw in msg for kw in ("rate limit", "429", "503", "timeout", "unavailable"))

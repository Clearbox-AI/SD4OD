"""Google Imagen 3 inpainting provider via Vertex AI."""

from __future__ import annotations

import io
import os
from typing import Any

import numpy as np
from PIL import Image

from synthdet.generate.errors import InpaintingAPIError


class ImagenInpaintingProvider:
    """Inpainting via Google Imagen 3 on Vertex AI (``google-genai`` SDK).

    Imagen edit_image is only available through the Vertex AI client.
    Requires a GCP project with Vertex AI enabled.

    Authentication (in order of precedence):
        1. Explicit ``api_key`` parameter
        2. Environment variable named by ``api_key_env_var`` (default GOOGLE_API_KEY)
        3. Vertex AI with Application Default Credentials (``gcloud auth application-default login``)
           — set GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION env vars

    Satisfies the ``InpaintingProvider`` protocol.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "imagen-3.0-capability-001",
        guidance_scale: float = 75.0,
        api_key_env_var: str = "GOOGLE_API_KEY",
        project: str | None = None,
        location: str | None = None,
    ) -> None:
        self._model = model
        self._guidance_scale = guidance_scale
        self._client: Any = None

        # Resolve auth: explicit key → env var → Vertex AI ADC
        resolved_key = api_key or os.environ.get(api_key_env_var)
        self._api_key = resolved_key

        self._project = project or os.environ.get("GOOGLE_CLOUD_PROJECT")
        self._location = location or os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")

        if not self._api_key and not self._project:
            raise ValueError(
                f"No credentials provided. Either:\n"
                f"  1. Set {api_key_env_var} env var (API key auth), or\n"
                f"  2. Set GOOGLE_CLOUD_PROJECT env var and run "
                f"'gcloud auth application-default login' (Vertex AI auth)"
            )

    @property
    def cost_per_image(self) -> float:
        return 0.02

    # ------------------------------------------------------------------

    def inpaint(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        prompt: str,
        *,
        seed: int | None = None,
        num_images: int = 1,
    ) -> list[np.ndarray]:
        """Send an inpainting request to Imagen 3 via Vertex AI.

        Args:
            image: BGR uint8 numpy array.
            mask: uint8 mask (255 = edit region, 0 = preserve).
            prompt: Text prompt describing the desired content.
            seed: Optional generation seed.
            num_images: Number of images to return.

        Returns:
            List of BGR uint8 numpy arrays.
        """
        genai = self._get_genai()
        client = self._get_client(genai)

        # Convert image: BGR np → RGB PIL → PNG bytes
        img_rgb = Image.fromarray(image[:, :, ::-1])
        img_bytes = _pil_to_png_bytes(img_rgb)

        # Convert mask: uint8 np → PIL mode-L → PNG bytes
        mask_pil = Image.fromarray(mask, mode="L")
        mask_bytes = _pil_to_png_bytes(mask_pil)

        try:
            ref_image = genai.types.RawReferenceImage(
                reference_image=genai.types.Image(image_bytes=img_bytes),
                reference_id=0,
            )
            mask_ref = genai.types.MaskReferenceImage(
                reference_id=1,
                reference_image=genai.types.Image(image_bytes=mask_bytes),
                config=genai.types.MaskReferenceConfig(
                    mask_mode="MASK_MODE_USER_PROVIDED",
                    mask_dilation=0.0,
                ),
            )

            config_kwargs: dict[str, Any] = {
                "edit_mode": "EDIT_MODE_INPAINT_INSERTION",
                "number_of_images": num_images,
            }
            if seed is not None:
                config_kwargs["seed"] = seed

            response = client.models.edit_image(
                model=self._model,
                prompt=prompt,
                reference_images=[ref_image, mask_ref],
                config=genai.types.EditImageConfig(**config_kwargs),
            )

            results: list[np.ndarray] = []
            for gen_image in response.generated_images:
                pil_img = _extract_pil_image(gen_image.image)
                rgb_arr = np.array(pil_img.convert("RGB"))
                bgr_arr = rgb_arr[:, :, ::-1].copy()
                results.append(bgr_arr)
            return results

        except Exception as exc:
            retryable = _is_retryable(exc)
            raise InpaintingAPIError("imagen", str(exc), retryable=retryable) from exc

    # ------------------------------------------------------------------

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
                # Vertex AI client (required for edit_image)
                self._client = genai.Client(
                    vertexai=True,
                    project=self._project,
                    location=self._location,
                )
            else:
                # API key client (will fail for edit_image, but useful for
                # other genai features; kept as fallback)
                self._client = genai.Client(api_key=self._api_key)
        return self._client


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_pil_image(image_obj: Any) -> Image.Image:
    """Extract a PIL Image from the SDK's Image object.

    The google-genai SDK exposes the PIL image via ``._pil_image``.
    We try multiple access patterns for forward-compatibility.
    """
    # Official pattern used in Google's own notebooks
    if hasattr(image_obj, "_pil_image") and image_obj._pil_image is not None:
        return image_obj._pil_image
    # Some SDK versions may expose image_bytes
    if hasattr(image_obj, "image_bytes") and image_obj.image_bytes:
        return Image.open(io.BytesIO(image_obj.image_bytes))
    # Fallback: try .data (Blob-style)
    if hasattr(image_obj, "data") and image_obj.data:
        return Image.open(io.BytesIO(image_obj.data))
    raise AttributeError(
        f"Cannot extract PIL image from {type(image_obj).__name__}. "
        f"Available attributes: {[a for a in dir(image_obj) if not a.startswith('__')]}"
    )


def _pil_to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _is_retryable(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(kw in msg for kw in ("rate limit", "429", "503", "timeout", "unavailable"))

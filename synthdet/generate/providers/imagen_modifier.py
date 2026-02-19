"""Google Imagen 3 controlled-editing provider for whole-image transformation.

Uses ``edit_image`` with ``EDIT_MODE_CONTROLLED_EDITING`` to transform a clean
source image into a damaged version while preserving overall structure. The source
image is passed as a ``ControlReferenceImage`` so that edges and geometry are
maintained while the model applies damage described in the text prompt.

This is the provider backend for the modify-and-annotate pipeline.
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


class ImagenModifierProvider:
    """Whole-image transformation via Google Imagen 3 controlled editing.

    Sends a clean image + damage prompt to the Imagen ``edit_image`` API using
    ``EDIT_MODE_CONTROLLED_EDITING``.  The source is passed as a
    ``ControlReferenceImage`` which preserves structure (edges, geometry) while
    the model applies the requested transformation.

    Authentication (in order of precedence):
        1. Explicit ``api_key`` parameter
        2. Environment variable named by ``api_key_env_var``
        3. Vertex AI with Application Default Credentials

    Satisfies the modifier provider interface expected by
    ``ModifyAndAnnotateGenerator``.
    """

    def __init__(
        self,
        model: str = "imagen-3.0-capability-001",
        control_type: str = "CONTROL_TYPE_CANNY",
        api_key: str | None = None,
        api_key_env_var: str = "GOOGLE_API_KEY",
        project: str | None = None,
        location: str | None = None,
    ) -> None:
        self._model = model
        self._control_type = control_type
        self._client: Any = None

        self._project = project or os.environ.get("GOOGLE_CLOUD_PROJECT")
        self._location = location or os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")

        if not self._project:
            raise ValueError(
                "ImagenModifierProvider requires Vertex AI (edit_image is not "
                "available via API key). Set GOOGLE_CLOUD_PROJECT env var and "
                "run 'gcloud auth application-default login'."
            )

    @property
    def cost_per_image(self) -> float:
        return 0.02

    def modify(
        self,
        image: np.ndarray,
        prompt: str,
        *,
        num_images: int = 1,
        seed: int | None = None,
    ) -> list[np.ndarray]:
        """Transform a clean image into a damaged version.

        Args:
            image: BGR uint8 numpy array (the clean source image).
            prompt: Text prompt describing the desired damage/transformation.
            num_images: Number of result images to generate.
            seed: Optional generation seed.

        Returns:
            List of BGR uint8 numpy arrays (modified images).
        """
        genai = self._get_genai()
        client = self._get_client(genai)

        # Convert BGR numpy → RGB PIL → PNG bytes
        img_rgb = Image.fromarray(image[:, :, ::-1])
        img_bytes = _pil_to_png_bytes(img_rgb)

        try:
            control_ref = genai.types.ControlReferenceImage(
                reference_image=genai.types.Image(image_bytes=img_bytes),
                reference_id=0,
                config=genai.types.ControlReferenceConfig(
                    control_type=self._control_type,
                ),
            )

            config_kwargs: dict[str, Any] = {
                "edit_mode": "EDIT_MODE_CONTROLLED_EDITING",
                "number_of_images": num_images,
            }
            if seed is not None:
                config_kwargs["seed"] = seed

            response = client.models.edit_image(
                model=self._model,
                prompt=prompt,
                reference_images=[control_ref],
                config=genai.types.EditImageConfig(**config_kwargs),
            )

            results: list[np.ndarray] = []
            for gen_image in response.generated_images:
                pil_img = _extract_pil_image(gen_image.image)
                rgb_arr = np.array(pil_img.convert("RGB"))
                bgr_arr = rgb_arr[:, :, ::-1].copy()
                results.append(bgr_arr)
            return results

        except InpaintingAPIError:
            raise
        except Exception as exc:
            retryable = _is_retryable(exc)
            raise InpaintingAPIError(
                "imagen_modifier", str(exc), retryable=retryable
            ) from exc

    # ------------------------------------------------------------------

    def _get_genai(self) -> Any:
        try:
            import google.genai as genai  # type: ignore[import-untyped]
            return genai
        except ImportError as exc:
            raise ImportError(
                "google-genai is required for the Imagen modifier provider. "
                "Install it with: pip install 'google-genai>=1.0'"
            ) from exc

    def _get_client(self, genai: Any) -> Any:
        if self._client is None:
            # edit_image requires Vertex AI — always use vertexai=True
            self._client = genai.Client(
                vertexai=True,
                project=self._project,
                location=self._location,
            )
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


def _pil_to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _is_retryable(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(kw in msg for kw in ("rate limit", "429", "503", "timeout", "unavailable"))

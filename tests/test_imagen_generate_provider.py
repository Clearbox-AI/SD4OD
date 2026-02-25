"""Tests for synthdet.generate.providers.imagen_generate.

All tests are mock-based — no real API calls or GPU required.
"""

from __future__ import annotations

import io
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from synthdet.generate.errors import InpaintingAPIError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_provider(bg_color=(220, 220, 220), threshold_delta=30, **kwargs):
    """Create an ImagenGenerateProvider with a fake API key."""
    from synthdet.generate.providers.imagen_generate import ImagenGenerateProvider

    return ImagenGenerateProvider(
        api_key="test-key-123",
        bg_color=bg_color,
        threshold_delta=threshold_delta,
        **kwargs,
    )


def _make_pil_image(width=512, height=512, color=(220, 220, 220)):
    """Create a solid-color PIL image."""
    arr = np.full((height, width, 3), color, dtype=np.uint8)
    return Image.fromarray(arr, "RGB")


def _make_pil_image_with_defect(
    width=512, height=512, bg_color=(220, 220, 220), defect_color=(50, 50, 50),
):
    """Create a PIL image with a central defect region (darker than background)."""
    arr = np.full((height, width, 3), bg_color, dtype=np.uint8)
    # Place a defect patch in center
    cy, cx = height // 2, width // 2
    r = min(height, width) // 6
    arr[cy - r : cy + r, cx - r : cx + r] = defect_color
    return Image.fromarray(arr, "RGB")


def _mock_genai_response(pil_img: Image.Image):
    """Build a fake genai response from a PIL image."""
    image_obj = SimpleNamespace(_pil_image=pil_img, image_bytes=None, data=None)
    gen_image = SimpleNamespace(image=image_obj)
    return SimpleNamespace(generated_images=[gen_image])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestImagenGenerateProvider:
    def test_cost_per_image(self):
        provider = _make_provider()
        assert provider.cost_per_image == 0.04

    def test_missing_credentials_error(self, monkeypatch):
        """Error when no API key or project is provided."""
        from synthdet.generate.providers.imagen_generate import ImagenGenerateProvider

        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
        with pytest.raises(ValueError, match="No credentials"):
            ImagenGenerateProvider()

    def test_api_key_auth_constructs(self):
        """Provider can be constructed with just an API key."""
        provider = _make_provider()
        assert provider._api_key == "test-key-123"

    def test_vertex_ai_auth_constructs(self, monkeypatch):
        """Provider can be constructed with project ID (Vertex AI)."""
        from synthdet.generate.providers.imagen_generate import ImagenGenerateProvider

        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "my-project")
        provider = ImagenGenerateProvider()
        assert provider._project == "my-project"

    def test_generate_defect_patch_returns_bgr_and_mask(self):
        """generate_defect_patch returns (BGR array, uint8 mask)."""
        provider = _make_provider()
        pil_img = _make_pil_image_with_defect()
        response = _mock_genai_response(pil_img)

        mock_genai = MagicMock()
        mock_client = MagicMock()
        mock_client.models.generate_images.return_value = response
        mock_genai.types.GenerateImagesConfig.return_value = MagicMock()

        provider._get_genai = lambda: mock_genai
        provider._get_client = lambda genai: mock_client

        patch_bgr, mask = provider.generate_defect_patch("test prompt", size=(256, 256))

        assert patch_bgr.shape == (256, 256, 3)
        assert patch_bgr.dtype == np.uint8
        assert mask.shape == (256, 256)
        assert mask.dtype == np.uint8

    def test_mask_extraction_detects_defect(self):
        """_extract_mask should detect pixels that differ from bg_color."""
        provider = _make_provider(bg_color=(220, 220, 220), threshold_delta=30)

        # Image with a dark square on light gray background
        rgb = np.full((100, 100, 3), 220, dtype=np.uint8)
        rgb[30:70, 30:70] = 50  # defect region

        mask = provider._extract_mask(rgb)

        assert mask.shape == (100, 100)
        assert mask.dtype == np.uint8
        # Center region should be nonzero (defect detected)
        center_mask = mask[40:60, 40:60]
        assert center_mask.mean() > 100
        # Corners should be mostly zero (background)
        corner_mask = mask[0:10, 0:10]
        assert corner_mask.mean() < 50

    def test_mask_extraction_low_coverage_fallback(self):
        """When initial threshold yields <2% coverage, threshold is lowered."""
        provider = _make_provider(bg_color=(220, 220, 220), threshold_delta=80)

        # Image with a subtle defect (small color difference)
        rgb = np.full((100, 100, 3), 220, dtype=np.uint8)
        # Defect with difference of 50 — above threshold_delta/2=40 but below 80
        rgb[40:60, 40:60] = 170

        mask = provider._extract_mask(rgb)

        # With the fallback (threshold_delta//2=40), the defect should be detected
        center_mask = mask[45:55, 45:55]
        assert center_mask.mean() > 50

    def test_mask_extraction_uniform_image(self):
        """Uniform background should produce near-zero mask."""
        provider = _make_provider(bg_color=(220, 220, 220), threshold_delta=30)
        rgb = np.full((100, 100, 3), 220, dtype=np.uint8)
        mask = provider._extract_mask(rgb)
        assert mask.mean() < 10

    def test_no_images_returned_raises(self):
        """InpaintingAPIError when API returns empty results."""
        provider = _make_provider()
        response = SimpleNamespace(generated_images=[])

        mock_genai = MagicMock()
        mock_client = MagicMock()
        mock_client.models.generate_images.return_value = response
        mock_genai.types.GenerateImagesConfig.return_value = MagicMock()

        provider._get_genai = lambda: mock_genai
        provider._get_client = lambda genai: mock_client

        with pytest.raises(InpaintingAPIError, match="No images returned"):
            provider.generate_defect_patch("test")

    def test_api_error_wrapped(self):
        """General exceptions are wrapped as InpaintingAPIError."""
        provider = _make_provider()

        mock_genai = MagicMock()
        mock_client = MagicMock()
        mock_client.models.generate_images.side_effect = RuntimeError("something broke")
        mock_genai.types.GenerateImagesConfig.return_value = MagicMock()

        provider._get_genai = lambda: mock_genai
        provider._get_client = lambda genai: mock_client

        with pytest.raises(InpaintingAPIError) as exc_info:
            provider.generate_defect_patch("test")
        assert exc_info.value.provider == "imagen_generate"
        assert exc_info.value.retryable is False

    def test_retryable_error_detection(self):
        """Rate limit / 429 / 503 errors should be retryable."""
        provider = _make_provider()

        for error_msg in ["rate limit exceeded", "HTTP 429", "503 Service Unavailable", "timeout"]:
            mock_genai = MagicMock()
            mock_client = MagicMock()
            mock_client.models.generate_images.side_effect = RuntimeError(error_msg)
            mock_genai.types.GenerateImagesConfig.return_value = MagicMock()

            provider._get_genai = lambda: mock_genai
            provider._get_client = lambda genai: mock_client

            with pytest.raises(InpaintingAPIError) as exc_info:
                provider.generate_defect_patch("test")
            assert exc_info.value.retryable is True, f"Expected retryable for: {error_msg}"

    def test_resize_to_requested_size(self):
        """Output is resized to match requested size parameter."""
        provider = _make_provider()
        # Generate a 512x512 image but request 200x100
        pil_img = _make_pil_image_with_defect(512, 512)
        response = _mock_genai_response(pil_img)

        mock_genai = MagicMock()
        mock_client = MagicMock()
        mock_client.models.generate_images.return_value = response
        mock_genai.types.GenerateImagesConfig.return_value = MagicMock()

        provider._get_genai = lambda: mock_genai
        provider._get_client = lambda genai: mock_client

        patch_bgr, mask = provider.generate_defect_patch("test", size=(200, 100))

        assert patch_bgr.shape == (100, 200, 3)
        assert mask.shape == (100, 200)

    def test_prompt_includes_bg_color(self):
        """The prompt sent to API should include the background color description."""
        provider = _make_provider(bg_color=(200, 200, 200))
        pil_img = _make_pil_image_with_defect()
        response = _mock_genai_response(pil_img)

        mock_genai = MagicMock()
        mock_client = MagicMock()
        mock_client.models.generate_images.return_value = response
        mock_genai.types.GenerateImagesConfig.return_value = MagicMock()

        provider._get_genai = lambda: mock_genai
        provider._get_client = lambda genai: mock_client

        provider.generate_defect_patch("A deep scratch", size=(64, 64))

        call_args = mock_client.models.generate_images.call_args
        prompt_sent = call_args[1]["prompt"] if "prompt" in call_args[1] else call_args.kwargs["prompt"]
        assert "200,200,200" in prompt_sent


class TestExtractPilImage:
    def test_from_pil_image_attr(self):
        from synthdet.generate.providers.imagen_generate import _extract_pil_image

        pil = Image.new("RGB", (10, 10))
        obj = SimpleNamespace(_pil_image=pil, image_bytes=None, data=None)
        assert _extract_pil_image(obj) is pil

    def test_from_image_bytes(self):
        from synthdet.generate.providers.imagen_generate import _extract_pil_image

        pil = Image.new("RGB", (10, 10))
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        obj = SimpleNamespace(_pil_image=None, image_bytes=buf.getvalue(), data=None)
        result = _extract_pil_image(obj)
        assert result.size == (10, 10)

    def test_from_data_attr(self):
        from synthdet.generate.providers.imagen_generate import _extract_pil_image

        pil = Image.new("RGB", (10, 10))
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        obj = SimpleNamespace(_pil_image=None, image_bytes=None, data=buf.getvalue())
        result = _extract_pil_image(obj)
        assert result.size == (10, 10)

    def test_no_attributes_raises(self):
        from synthdet.generate.providers.imagen_generate import _extract_pil_image

        obj = SimpleNamespace(_pil_image=None, image_bytes=None, data=None)
        with pytest.raises(AttributeError, match="Cannot extract PIL image"):
            _extract_pil_image(obj)


class TestIsRetryable:
    @pytest.mark.parametrize("msg", [
        "rate limit exceeded",
        "HTTP 429 Too Many Requests",
        "503 Service Unavailable",
        "Connection timeout",
        "Service unavailable",
    ])
    def test_retryable_messages(self, msg):
        from synthdet.generate.providers.imagen_generate import _is_retryable

        assert _is_retryable(RuntimeError(msg)) is True

    @pytest.mark.parametrize("msg", [
        "Invalid request",
        "Permission denied",
        "Unknown error",
    ])
    def test_non_retryable_messages(self, msg):
        from synthdet.generate.providers.imagen_generate import _is_retryable

        assert _is_retryable(RuntimeError(msg)) is False

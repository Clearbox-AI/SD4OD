"""Tests for synthdet.generate.providers.imagen_modifier.

All tests are mock-based — no real API calls or GPU required.
"""

from __future__ import annotations

import io
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image

from synthdet.generate.errors import InpaintingAPIError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_provider(**kwargs):
    """Create an ImagenModifierProvider with fake Vertex AI credentials."""
    from synthdet.generate.providers.imagen_modifier import ImagenModifierProvider

    return ImagenModifierProvider(project="test-project", **kwargs)


def _mock_genai_response(pil_images: list[Image.Image]):
    """Build a fake genai edit_image response."""
    gen_images = []
    for pil_img in pil_images:
        img_obj = SimpleNamespace(_pil_image=pil_img, image_bytes=None, data=None)
        gen_images.append(SimpleNamespace(image=img_obj))
    return SimpleNamespace(generated_images=gen_images)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestImagenModifierProvider:
    def test_cost_per_image(self):
        provider = _make_provider()
        assert provider.cost_per_image == 0.02

    def test_requires_vertex_ai(self, monkeypatch):
        """Provider requires Vertex AI project (edit_image not available via API key)."""
        from synthdet.generate.providers.imagen_modifier import ImagenModifierProvider

        monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
        with pytest.raises(ValueError, match="requires Vertex AI"):
            ImagenModifierProvider()

    def test_project_from_env(self, monkeypatch):
        """Provider reads project from GOOGLE_CLOUD_PROJECT env var."""
        from synthdet.generate.providers.imagen_modifier import ImagenModifierProvider

        monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "env-project")
        provider = ImagenModifierProvider()
        assert provider._project == "env-project"

    def test_modify_returns_bgr_arrays(self):
        """modify() returns list of BGR uint8 numpy arrays."""
        provider = _make_provider()

        input_bgr = np.random.randint(0, 255, (100, 150, 3), dtype=np.uint8)
        output_pil = Image.fromarray(np.zeros((100, 150, 3), dtype=np.uint8))
        response = _mock_genai_response([output_pil])

        mock_genai = MagicMock()
        mock_client = MagicMock()
        mock_client.models.edit_image.return_value = response
        mock_genai.types.ControlReferenceImage.return_value = MagicMock()
        mock_genai.types.Image.return_value = MagicMock()
        mock_genai.types.ControlReferenceConfig.return_value = MagicMock()
        mock_genai.types.EditImageConfig.return_value = MagicMock()

        provider._get_genai = lambda: mock_genai
        provider._get_client = lambda genai: mock_client

        results = provider.modify(input_bgr, "add scratches")

        assert len(results) == 1
        assert results[0].dtype == np.uint8
        assert results[0].shape == (100, 150, 3)

    def test_modify_multiple_images(self):
        """modify() with num_images>1 returns multiple arrays."""
        provider = _make_provider()

        input_bgr = np.zeros((64, 64, 3), dtype=np.uint8)
        output_pils = [
            Image.fromarray(np.full((64, 64, 3), i * 30, dtype=np.uint8))
            for i in range(3)
        ]
        response = _mock_genai_response(output_pils)

        mock_genai = MagicMock()
        mock_client = MagicMock()
        mock_client.models.edit_image.return_value = response
        mock_genai.types.ControlReferenceImage.return_value = MagicMock()
        mock_genai.types.Image.return_value = MagicMock()
        mock_genai.types.ControlReferenceConfig.return_value = MagicMock()
        mock_genai.types.EditImageConfig.return_value = MagicMock()

        provider._get_genai = lambda: mock_genai
        provider._get_client = lambda genai: mock_client

        results = provider.modify(input_bgr, "add damage", num_images=3)

        assert len(results) == 3

    def test_bgr_to_rgb_conversion(self):
        """Input BGR image should be converted to RGB for the API."""
        provider = _make_provider()

        # Pure blue in BGR = (255, 0, 0) → should be (0, 0, 255) in RGB
        input_bgr = np.zeros((64, 64, 3), dtype=np.uint8)
        input_bgr[:, :, 0] = 255  # blue channel in BGR
        output_pil = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
        response = _mock_genai_response([output_pil])

        captured_bytes = []

        mock_genai = MagicMock()
        mock_client = MagicMock()
        mock_client.models.edit_image.return_value = response

        def capture_image(image_bytes=None, **kw):
            if image_bytes:
                captured_bytes.append(image_bytes)
            return MagicMock()

        mock_genai.types.Image = capture_image
        mock_genai.types.ControlReferenceImage.return_value = MagicMock()
        mock_genai.types.ControlReferenceConfig.return_value = MagicMock()
        mock_genai.types.EditImageConfig.return_value = MagicMock()

        provider._get_genai = lambda: mock_genai
        provider._get_client = lambda genai: mock_client

        provider.modify(input_bgr, "test")

        assert len(captured_bytes) == 1
        decoded = Image.open(io.BytesIO(captured_bytes[0]))
        arr = np.array(decoded)
        # After BGR→RGB conversion, the image should be blue in RGB: R=0, G=0, B=255
        assert arr[:, :, 0].mean() < 10   # R channel ~0
        assert arr[:, :, 2].mean() > 245   # B channel ~255

    def test_api_error_wrapped(self):
        """General exceptions are wrapped as InpaintingAPIError."""
        provider = _make_provider()

        mock_genai = MagicMock()
        mock_client = MagicMock()
        mock_client.models.edit_image.side_effect = RuntimeError("API failure")
        mock_genai.types.ControlReferenceImage.return_value = MagicMock()
        mock_genai.types.Image.return_value = MagicMock()
        mock_genai.types.ControlReferenceConfig.return_value = MagicMock()
        mock_genai.types.EditImageConfig.return_value = MagicMock()

        provider._get_genai = lambda: mock_genai
        provider._get_client = lambda genai: mock_client

        with pytest.raises(InpaintingAPIError) as exc_info:
            provider.modify(np.zeros((64, 64, 3), dtype=np.uint8), "test")
        assert exc_info.value.provider == "imagen_modifier"
        assert exc_info.value.retryable is False

    def test_retryable_errors(self):
        """Rate limit and transient errors should be marked retryable."""
        provider = _make_provider()

        for error_msg in ["rate limit", "429", "503", "timeout"]:
            mock_genai = MagicMock()
            mock_client = MagicMock()
            mock_client.models.edit_image.side_effect = RuntimeError(error_msg)
            mock_genai.types.ControlReferenceImage.return_value = MagicMock()
            mock_genai.types.Image.return_value = MagicMock()
            mock_genai.types.ControlReferenceConfig.return_value = MagicMock()
            mock_genai.types.EditImageConfig.return_value = MagicMock()

            provider._get_genai = lambda: mock_genai
            provider._get_client = lambda genai: mock_client

            with pytest.raises(InpaintingAPIError) as exc_info:
                provider.modify(np.zeros((64, 64, 3), dtype=np.uint8), "test")
            assert exc_info.value.retryable is True, f"Expected retryable for: {error_msg}"

    def test_seed_passed_to_config(self):
        """Seed should be included in EditImageConfig when provided."""
        provider = _make_provider()

        input_bgr = np.zeros((64, 64, 3), dtype=np.uint8)
        output_pil = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
        response = _mock_genai_response([output_pil])

        config_kwargs_received = []

        mock_genai = MagicMock()
        mock_client = MagicMock()
        mock_client.models.edit_image.return_value = response
        mock_genai.types.ControlReferenceImage.return_value = MagicMock()
        mock_genai.types.Image.return_value = MagicMock()
        mock_genai.types.ControlReferenceConfig.return_value = MagicMock()

        def capture_config(**kwargs):
            config_kwargs_received.append(kwargs)
            return MagicMock()

        mock_genai.types.EditImageConfig = capture_config

        provider._get_genai = lambda: mock_genai
        provider._get_client = lambda genai: mock_client

        provider.modify(input_bgr, "test", seed=42)

        assert len(config_kwargs_received) == 1
        assert config_kwargs_received[0]["seed"] == 42

    def test_no_seed_omitted_from_config(self):
        """Seed should NOT be in EditImageConfig when not provided."""
        provider = _make_provider()

        input_bgr = np.zeros((64, 64, 3), dtype=np.uint8)
        output_pil = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
        response = _mock_genai_response([output_pil])

        config_kwargs_received = []

        mock_genai = MagicMock()
        mock_client = MagicMock()
        mock_client.models.edit_image.return_value = response
        mock_genai.types.ControlReferenceImage.return_value = MagicMock()
        mock_genai.types.Image.return_value = MagicMock()
        mock_genai.types.ControlReferenceConfig.return_value = MagicMock()

        def capture_config(**kwargs):
            config_kwargs_received.append(kwargs)
            return MagicMock()

        mock_genai.types.EditImageConfig = capture_config

        provider._get_genai = lambda: mock_genai
        provider._get_client = lambda genai: mock_client

        provider.modify(input_bgr, "test")

        assert "seed" not in config_kwargs_received[0]

    def test_control_type_passed(self):
        """Control type should be passed to ControlReferenceConfig."""
        provider = _make_provider(control_type="CONTROL_TYPE_EDGE")

        input_bgr = np.zeros((64, 64, 3), dtype=np.uint8)
        output_pil = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
        response = _mock_genai_response([output_pil])

        control_configs_received = []

        mock_genai = MagicMock()
        mock_client = MagicMock()
        mock_client.models.edit_image.return_value = response
        mock_genai.types.Image.return_value = MagicMock()
        mock_genai.types.ControlReferenceImage.return_value = MagicMock()
        mock_genai.types.EditImageConfig.return_value = MagicMock()

        def capture_control_config(**kwargs):
            control_configs_received.append(kwargs)
            return MagicMock()

        mock_genai.types.ControlReferenceConfig = capture_control_config

        provider._get_genai = lambda: mock_genai
        provider._get_client = lambda genai: mock_client

        provider.modify(input_bgr, "test")

        assert control_configs_received[0]["control_type"] == "CONTROL_TYPE_EDGE"


class TestModifierHelpers:
    def test_pil_to_png_bytes_roundtrip(self):
        from synthdet.generate.providers.imagen_modifier import _pil_to_png_bytes

        pil_img = Image.fromarray(
            np.random.randint(0, 255, (50, 80, 3), dtype=np.uint8)
        )
        png_bytes = _pil_to_png_bytes(pil_img)
        assert isinstance(png_bytes, bytes)
        assert len(png_bytes) > 0
        decoded = Image.open(io.BytesIO(png_bytes))
        assert decoded.size == (80, 50)

    def test_extract_pil_image_from_pil_attr(self):
        from synthdet.generate.providers.imagen_modifier import _extract_pil_image

        pil = Image.new("RGB", (10, 10))
        obj = SimpleNamespace(_pil_image=pil, image_bytes=None, data=None)
        assert _extract_pil_image(obj) is pil

    def test_extract_pil_image_from_bytes(self):
        from synthdet.generate.providers.imagen_modifier import _extract_pil_image

        pil = Image.new("RGB", (10, 10))
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        obj = SimpleNamespace(_pil_image=None, image_bytes=buf.getvalue(), data=None)
        result = _extract_pil_image(obj)
        assert result.size == (10, 10)

    def test_extract_pil_image_no_attrs_raises(self):
        from synthdet.generate.providers.imagen_modifier import _extract_pil_image

        obj = SimpleNamespace(_pil_image=None, image_bytes=None, data=None)
        with pytest.raises(AttributeError, match="Cannot extract PIL image"):
            _extract_pil_image(obj)

    @pytest.mark.parametrize("msg,expected", [
        ("rate limit exceeded", True),
        ("429 Too Many Requests", True),
        ("503 Service Unavailable", True),
        ("timeout", True),
        ("unavailable", True),
        ("Permission denied", False),
        ("invalid argument", False),
    ])
    def test_is_retryable(self, msg, expected):
        from synthdet.generate.providers.imagen_modifier import _is_retryable

        assert _is_retryable(RuntimeError(msg)) is expected

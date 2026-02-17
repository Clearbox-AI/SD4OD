"""Tests for synthdet.generate.providers.imagen."""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from synthdet.generate.inpainting import InpaintingProvider


class TestImagenProvider:
    def test_satisfies_protocol(self):
        """ImagenInpaintingProvider satisfies InpaintingProvider protocol."""
        # Import without API key should raise ValueError, but the class itself
        # should exist and be protocol-compatible.
        from synthdet.generate.providers.imagen import ImagenInpaintingProvider

        # Use a dummy key to construct — it won't hit the API
        provider = ImagenInpaintingProvider(api_key="test-key-123")
        assert isinstance(provider, InpaintingProvider)

    def test_image_conversion_roundtrip(self):
        """BGR np → PIL RGB → PNG bytes roundtrip preserves shape."""
        from synthdet.generate.providers.imagen import _pil_to_png_bytes

        bgr = np.random.randint(0, 255, (100, 150, 3), dtype=np.uint8)
        rgb = Image.fromarray(bgr[:, :, ::-1])
        png_bytes = _pil_to_png_bytes(rgb)
        assert isinstance(png_bytes, bytes)
        assert len(png_bytes) > 0
        # Decode back
        decoded = Image.open(io.BytesIO(png_bytes))
        assert decoded.size == (150, 100)

    def test_mask_conversion(self):
        """uint8 mask → PIL L → PNG bytes."""
        from synthdet.generate.providers.imagen import _pil_to_png_bytes

        mask = np.zeros((100, 150), dtype=np.uint8)
        mask[20:80, 30:120] = 255
        mask_pil = Image.fromarray(mask, mode="L")
        png_bytes = _pil_to_png_bytes(mask_pil)
        assert isinstance(png_bytes, bytes)
        assert len(png_bytes) > 0

    def test_cost_per_image(self):
        from synthdet.generate.providers.imagen import ImagenInpaintingProvider

        provider = ImagenInpaintingProvider(api_key="test-key")
        assert provider.cost_per_image == 0.02

    def test_missing_credentials_error(self, monkeypatch):
        """Clear error when no credentials are available."""
        from synthdet.generate.providers.imagen import ImagenInpaintingProvider

        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
        with pytest.raises(ValueError, match="No credentials"):
            ImagenInpaintingProvider()

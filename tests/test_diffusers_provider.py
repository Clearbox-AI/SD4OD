"""Tests for synthdet.generate.providers.diffusers_local.

All tests are mock-based — no real model loading or GPU required.
torch and diffusers are mocked via sys.modules when not installed.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from synthdet.generate.errors import InpaintingAPIError
from synthdet.generate.inpainting import InpaintingProvider

# ---------------------------------------------------------------------------
# Mock torch if not installed
# ---------------------------------------------------------------------------

_torch_available = True
try:
    import torch as _real_torch
except ImportError:
    _torch_available = False


@pytest.fixture()
def mock_torch():
    """Provide a mock torch module when torch is not installed."""
    if _torch_available:
        yield _real_torch
        return

    mock = MagicMock()
    mock.float16 = "float16"
    mock.float32 = "float32"

    # torch.Generator returns a mock with .manual_seed()
    gen_instance = MagicMock()
    gen_instance.manual_seed.return_value = gen_instance
    mock.Generator.return_value = gen_instance

    mock.cuda.is_available.return_value = False

    saved = sys.modules.get("torch")
    sys.modules["torch"] = mock
    try:
        yield mock
    finally:
        if saved is None:
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = saved


def _make_fake_pipeline():
    """Create a callable mock pipeline that returns black 512x512 images."""
    def fake_pipeline(**kwargs):
        out_pil = Image.fromarray(np.zeros((512, 512, 3), dtype=np.uint8))
        return SimpleNamespace(images=[out_pil])
    return MagicMock(side_effect=fake_pipeline)


class TestDiffusersProvider:
    """Unit tests for DiffusersInpaintingProvider."""

    def test_satisfies_protocol(self):
        from synthdet.generate.providers.diffusers_local import DiffusersInpaintingProvider

        provider = DiffusersInpaintingProvider()
        assert isinstance(provider, InpaintingProvider)

    def test_cost_per_image_is_zero(self):
        from synthdet.generate.providers.diffusers_local import DiffusersInpaintingProvider

        provider = DiffusersInpaintingProvider()
        assert provider.cost_per_image == 0.0

    def test_image_conversion_bgr_to_rgb(self, mock_torch):
        """Verify BGR input is converted to RGB before the pipeline sees it."""
        from synthdet.generate.providers.diffusers_local import DiffusersInpaintingProvider

        provider = DiffusersInpaintingProvider()

        # Pure blue in BGR → should become pure blue in RGB (R=0, G=0, B=255)
        bgr = np.zeros((64, 64, 3), dtype=np.uint8)
        bgr[:, :, 0] = 255  # BGR blue channel
        mask = np.full((64, 64), 255, dtype=np.uint8)

        received_images = []

        def capture_pipeline(**kwargs):
            received_images.append(kwargs["image"])
            out_pil = Image.fromarray(np.zeros((512, 512, 3), dtype=np.uint8))
            return SimpleNamespace(images=[out_pil])

        mock_pipe = MagicMock(side_effect=capture_pipeline)
        provider._pipeline = mock_pipe
        provider._resolved_device = "cpu"

        provider.inpaint(bgr, mask, "test prompt")

        assert len(received_images) == 1
        img_arr = np.array(received_images[0])
        # After BGR→RGB, the original blue channel (idx 0 in BGR) becomes idx 2 in RGB
        assert img_arr[:, :, 0].mean() < 10  # R channel ~0
        assert img_arr[:, :, 2].mean() > 245  # B channel ~255

    def test_mask_conversion(self, mock_torch):
        """Verify uint8 mask is converted to PIL mode 'L'."""
        from synthdet.generate.providers.diffusers_local import DiffusersInpaintingProvider

        provider = DiffusersInpaintingProvider()

        bgr = np.zeros((64, 64, 3), dtype=np.uint8)
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[10:50, 10:50] = 255

        received_masks = []

        def capture_pipeline(**kwargs):
            received_masks.append(kwargs["mask_image"])
            out_pil = Image.fromarray(np.zeros((512, 512, 3), dtype=np.uint8))
            return SimpleNamespace(images=[out_pil])

        mock_pipe = MagicMock(side_effect=capture_pipeline)
        provider._pipeline = mock_pipe
        provider._resolved_device = "cpu"

        provider.inpaint(bgr, mask, "test")

        assert len(received_masks) == 1
        assert received_masks[0].mode == "L"

    def test_resize_to_model_resolution_and_back(self, mock_torch):
        """860x640 input → 512x512 for model → 860x640 output."""
        from synthdet.generate.providers.diffusers_local import DiffusersInpaintingProvider

        provider = DiffusersInpaintingProvider(model="runwayml/stable-diffusion-inpainting")

        bgr = np.random.randint(0, 255, (640, 860, 3), dtype=np.uint8)
        mask = np.full((640, 860), 255, dtype=np.uint8)

        received_sizes = []

        def capture_pipeline(**kwargs):
            received_sizes.append(kwargs["image"].size)
            out_pil = Image.fromarray(np.zeros((512, 512, 3), dtype=np.uint8))
            return SimpleNamespace(images=[out_pil])

        mock_pipe = MagicMock(side_effect=capture_pipeline)
        provider._pipeline = mock_pipe
        provider._resolved_device = "cpu"

        results = provider.inpaint(bgr, mask, "test")

        assert received_sizes[0] == (512, 512)
        assert results[0].shape == (640, 860, 3)

    def test_seed_creates_generator(self, mock_torch):
        """Passing seed should create a torch.Generator with manual_seed."""
        from synthdet.generate.providers.diffusers_local import DiffusersInpaintingProvider

        provider = DiffusersInpaintingProvider()

        bgr = np.zeros((64, 64, 3), dtype=np.uint8)
        mask = np.full((64, 64), 255, dtype=np.uint8)

        received_generators = []

        def capture_pipeline(**kwargs):
            received_generators.append(kwargs.get("generator"))
            out_pil = Image.fromarray(np.zeros((512, 512, 3), dtype=np.uint8))
            return SimpleNamespace(images=[out_pil])

        mock_pipe = MagicMock(side_effect=capture_pipeline)
        provider._pipeline = mock_pipe
        provider._resolved_device = "cpu"

        provider.inpaint(bgr, mask, "test", seed=42)

        assert received_generators[0] is not None

    def test_no_seed_no_generator(self, mock_torch):
        """Without seed, generator should be None."""
        from synthdet.generate.providers.diffusers_local import DiffusersInpaintingProvider

        provider = DiffusersInpaintingProvider()

        bgr = np.zeros((64, 64, 3), dtype=np.uint8)
        mask = np.full((64, 64), 255, dtype=np.uint8)

        received_generators = []

        def capture_pipeline(**kwargs):
            received_generators.append(kwargs.get("generator"))
            out_pil = Image.fromarray(np.zeros((512, 512, 3), dtype=np.uint8))
            return SimpleNamespace(images=[out_pil])

        mock_pipe = MagicMock(side_effect=capture_pipeline)
        provider._pipeline = mock_pipe
        provider._resolved_device = "cpu"

        provider.inpaint(bgr, mask, "test")

        assert received_generators[0] is None

    def test_missing_torch_error(self):
        """Clear error message when torch is not installed."""
        from synthdet.generate.providers.diffusers_local import DiffusersInpaintingProvider

        provider = DiffusersInpaintingProvider()

        bgr = np.zeros((64, 64, 3), dtype=np.uint8)
        mask = np.full((64, 64), 255, dtype=np.uint8)

        # Remove torch from sys.modules to simulate missing dependency
        saved = sys.modules.get("torch")
        sys.modules["torch"] = None  # type: ignore[assignment]  # forces ImportError
        try:
            with pytest.raises(ImportError, match="PyTorch is required"):
                provider.inpaint(bgr, mask, "test")
        finally:
            if saved is None:
                sys.modules.pop("torch", None)
            else:
                sys.modules["torch"] = saved

    def test_cuda_oom_is_retryable(self, mock_torch):
        """CUDA OOM errors should be wrapped as retryable."""
        from synthdet.generate.providers.diffusers_local import DiffusersInpaintingProvider

        provider = DiffusersInpaintingProvider()
        provider._resolved_device = "cpu"

        bgr = np.zeros((64, 64, 3), dtype=np.uint8)
        mask = np.full((64, 64), 255, dtype=np.uint8)

        mock_pipe = MagicMock(side_effect=RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB"))
        provider._pipeline = mock_pipe

        with pytest.raises(InpaintingAPIError) as exc_info:
            provider.inpaint(bgr, mask, "test")
        assert exc_info.value.retryable is True
        assert exc_info.value.provider == "diffusers"

    def test_other_runtime_error_not_retryable(self, mock_torch):
        """Non-OOM RuntimeErrors should not be retryable."""
        from synthdet.generate.providers.diffusers_local import DiffusersInpaintingProvider

        provider = DiffusersInpaintingProvider()
        provider._resolved_device = "cpu"

        bgr = np.zeros((64, 64, 3), dtype=np.uint8)
        mask = np.full((64, 64), 255, dtype=np.uint8)

        mock_pipe = MagicMock(side_effect=RuntimeError("something else went wrong"))
        provider._pipeline = mock_pipe

        with pytest.raises(InpaintingAPIError) as exc_info:
            provider.inpaint(bgr, mask, "test")
        assert exc_info.value.retryable is False

    def test_device_auto_cuda(self, mock_torch):
        """Auto device resolves to cuda when available."""
        from synthdet.generate.providers.diffusers_local import DiffusersInpaintingProvider

        provider = DiffusersInpaintingProvider(device="auto")
        mock_torch.cuda.is_available.return_value = True
        assert provider._resolve_device() == "cuda"

    def test_device_auto_cpu(self, mock_torch):
        """Auto device resolves to cpu when CUDA unavailable."""
        from synthdet.generate.providers.diffusers_local import DiffusersInpaintingProvider

        provider = DiffusersInpaintingProvider(device="auto")
        mock_torch.cuda.is_available.return_value = False
        assert provider._resolve_device() == "cpu"

    def test_fp16_disabled_on_cpu(self, mock_torch):
        """fp16 should fall back to float32 on CPU."""
        from synthdet.generate.providers.diffusers_local import DiffusersInpaintingProvider

        provider = DiffusersInpaintingProvider(device="cpu", use_fp16=True)

        mock_auto_pipeline = MagicMock()
        mock_pipe_instance = MagicMock()
        mock_auto_pipeline.from_pretrained.return_value.to.return_value = mock_pipe_instance

        # Mock diffusers module
        mock_diffusers = MagicMock()
        mock_diffusers.AutoPipelineForInpainting = mock_auto_pipeline
        saved = sys.modules.get("diffusers")
        sys.modules["diffusers"] = mock_diffusers
        try:
            provider._ensure_pipeline()
        finally:
            if saved is None:
                sys.modules.pop("diffusers", None)
            else:
                sys.modules["diffusers"] = saved

        call_kwargs = mock_auto_pipeline.from_pretrained.call_args
        assert call_kwargs[1]["torch_dtype"] == mock_torch.float32

    def test_lazy_loading(self, mock_torch):
        """Pipeline should not be loaded at construction, only on first inpaint()."""
        from synthdet.generate.providers.diffusers_local import DiffusersInpaintingProvider

        provider = DiffusersInpaintingProvider()
        assert provider._pipeline is None

        bgr = np.zeros((64, 64, 3), dtype=np.uint8)
        mask = np.full((64, 64), 255, dtype=np.uint8)

        mock_pipe = _make_fake_pipeline()
        provider._pipeline = mock_pipe
        provider._resolved_device = "cpu"

        provider.inpaint(bgr, mask, "test")
        # Pipeline is still the same object (not reloaded)
        assert provider._pipeline is mock_pipe

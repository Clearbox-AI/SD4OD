"""Tests for synthdet.utils.embeddings — mock open_clip."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from synthdet.config import EmbeddingConfig
from synthdet.utils.embeddings import EmbeddingComputer


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def _make_mock_open_clip(embedding_dim: int = 512):
    """Create a mock open_clip module that returns predictable embeddings."""
    import torch

    mock_model = MagicMock()

    def fake_encode_image(tensor):
        batch_size = tensor.shape[0]
        # Deterministic embeddings based on input mean
        emb = torch.randn(batch_size, embedding_dim)
        return emb

    mock_model.encode_image = fake_encode_image
    mock_model.eval.return_value = mock_model
    mock_model.to.return_value = mock_model

    mock_preprocess = MagicMock(side_effect=lambda img: torch.randn(3, 224, 224))

    mock_open_clip = MagicMock()
    mock_open_clip.create_model_and_transforms.return_value = (
        mock_model, None, mock_preprocess,
    )
    return mock_open_clip


class TestEmbeddingComputerInit:
    def test_stores_config(self):
        ec = EmbeddingComputer(model_name="ViT-L-14", device="cpu", batch_size=16)
        assert ec.model_name == "ViT-L-14"
        assert ec.device == "cpu"
        assert ec.batch_size == 16

    def test_from_config(self):
        config = EmbeddingConfig(model_name="ViT-L-14", batch_size=64)
        ec = EmbeddingComputer.from_config(config)
        assert ec.model_name == "ViT-L-14"
        assert ec.batch_size == 64


class TestResolveDevice:
    def test_explicit_device(self):
        assert EmbeddingComputer._resolve_device("cpu") == "cpu"
        assert EmbeddingComputer._resolve_device("cuda") == "cuda"


class TestEnsureModel:
    def test_raises_import_error_without_open_clip(self):
        ec = EmbeddingComputer(device="cpu")
        with patch.dict("sys.modules", {"open_clip": None}):
            with patch("builtins.__import__", side_effect=_make_import_blocker("open_clip")):
                with pytest.raises(ImportError, match="open-clip-torch"):
                    ec._ensure_model()


@pytest.mark.skipif(not _torch_available(), reason="torch not installed")
class TestCompute:
    def test_returns_normalized_array(self):
        mock_oc = _make_mock_open_clip(embedding_dim=128)
        ec = EmbeddingComputer(device="cpu", batch_size=4)

        with patch.dict("sys.modules", {"open_clip": mock_oc}):
            with patch("synthdet.utils.embeddings.EmbeddingComputer._ensure_model") as mock_ensure:
                # Manually set up model state
                import torch

                ec._model = mock_oc.create_model_and_transforms()[0]
                ec._preprocess = mock_oc.create_model_and_transforms()[2]
                ec._embedding_dim = 128

                images = [Image.new("RGB", (64, 64)) for _ in range(5)]
                result = ec.compute(images)

        assert result.shape == (5, 128)
        assert result.dtype == np.float32
        # Check L2 normalization
        norms = np.linalg.norm(result, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_empty_input_returns_empty(self):
        ec = EmbeddingComputer(device="cpu")
        ec._model = MagicMock()
        ec._embedding_dim = 256
        ec._preprocess = MagicMock()

        result = ec.compute([])
        assert result.shape == (0, 256)

    def test_batching(self):
        """Verify images are processed in batches."""
        import torch

        call_count = 0
        batch_sizes_seen = []

        mock_model = MagicMock()

        def fake_encode(tensor):
            nonlocal call_count
            call_count += 1
            batch_sizes_seen.append(tensor.shape[0])
            emb = torch.randn(tensor.shape[0], 64)
            return emb

        mock_model.encode_image = fake_encode

        ec = EmbeddingComputer(device="cpu", batch_size=3)
        ec._model = mock_model
        ec._preprocess = lambda img: torch.randn(3, 224, 224)
        ec._embedding_dim = 64

        images = [Image.new("RGB", (64, 64)) for _ in range(7)]
        result = ec.compute(images)

        assert result.shape == (7, 64)
        assert call_count == 3  # 3 + 3 + 1
        assert batch_sizes_seen == [3, 3, 1]


@pytest.mark.skipif(not _torch_available(), reason="torch not installed")
class TestCache:
    def test_cache_roundtrip(self):
        import torch

        with tempfile.TemporaryDirectory() as tmpdir:
            ec = EmbeddingComputer(device="cpu", cache_dir=tmpdir)
            ec._embedding_dim = 64
            ec._model = MagicMock()
            mock_model = MagicMock()

            def fake_encode(tensor):
                return torch.randn(tensor.shape[0], 64)

            mock_model.encode_image = fake_encode
            ec._model = mock_model
            ec._preprocess = lambda img: torch.randn(3, 224, 224)

            images = [Image.new("RGB", (64, 64)) for _ in range(3)]
            result1 = ec.compute(images, cache_key="test_cache")

            # Second call should load from cache
            ec._model.encode_image = MagicMock(side_effect=AssertionError("should not be called"))
            result2 = ec.compute(images, cache_key="test_cache")

            np.testing.assert_array_equal(result1, result2)

    def test_no_cache_dir_skips_caching(self):
        ec = EmbeddingComputer(device="cpu")
        assert ec._load_cache("anything") is None


def _make_import_blocker(blocked_module: str):
    real_import = __builtins__.__import__ if hasattr(__builtins__, '__import__') else __import__

    def blocker(name, *args, **kwargs):
        if name == blocked_module:
            raise ImportError(f"No module named '{blocked_module}'")
        return real_import(name, *args, **kwargs)

    return blocker

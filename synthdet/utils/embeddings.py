"""Reusable embedding computation with lazy CLIP model loading.

Requires ``open-clip-torch`` (optional ``embeddings`` extra).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from synthdet.config import EmbeddingConfig


class EmbeddingComputer:
    """Compute image embeddings using OpenCLIP models.

    The model is lazy-loaded on first call to ``compute()`` or
    ``compute_from_paths()``, so importing is always safe.
    """

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        device: str = "auto",
        batch_size: int = 32,
        cache_dir: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.pretrained = pretrained
        self.device = self._resolve_device(device)
        self.batch_size = batch_size
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._model: Any = None
        self._preprocess: Any = None
        self._embedding_dim: int | None = None

    @classmethod
    def from_config(cls, config: EmbeddingConfig) -> EmbeddingComputer:
        return cls(
            model_name=config.model_name,
            pretrained=config.pretrained,
            device=config.device,
            batch_size=config.batch_size,
            cache_dir=config.cache_dir,
        )

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device != "auto":
            return device
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def _ensure_model(self) -> None:
        """Lazy-load the OpenCLIP model."""
        if self._model is not None:
            return
        try:
            import open_clip
        except ImportError as exc:
            raise ImportError(
                "open-clip-torch is required for embedding computation. "
                "Install it with: pip install 'synthdet[embeddings]'"
            ) from exc
        import torch

        model, _, preprocess = open_clip.create_model_and_transforms(
            self.model_name, pretrained=self.pretrained
        )
        model = model.to(self.device).eval()
        self._model = model
        self._preprocess = preprocess
        # Determine embedding dim from a dummy forward pass
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224, device=self.device)
            out = model.encode_image(dummy)
            self._embedding_dim = out.shape[-1]

    @property
    def embedding_dim(self) -> int:
        """Dimensionality of the embedding vectors."""
        self._ensure_model()
        assert self._embedding_dim is not None
        return self._embedding_dim

    def compute(
        self,
        images: list[Image.Image],
        cache_key: str | None = None,
    ) -> np.ndarray:
        """Compute L2-normalized embeddings for a list of PIL images.

        Returns:
            (N, D) float32 array of L2-normalized embeddings.
        """
        if cache_key:
            cached = self._load_cache(cache_key)
            if cached is not None:
                return cached

        if len(images) == 0:
            self._ensure_model()
            return np.zeros((0, self.embedding_dim), dtype=np.float32)

        self._ensure_model()
        import torch

        all_embeddings: list[np.ndarray] = []
        for start in range(0, len(images), self.batch_size):
            batch_imgs = images[start : start + self.batch_size]
            tensors = torch.stack(
                [self._preprocess(img) for img in batch_imgs]
            ).to(self.device)
            with torch.no_grad():
                emb = self._model.encode_image(tensors)
                emb = emb / emb.norm(dim=-1, keepdim=True)
                all_embeddings.append(emb.cpu().numpy())

        result = np.concatenate(all_embeddings, axis=0).astype(np.float32)
        if cache_key:
            self._save_cache(cache_key, result)
        return result

    def compute_from_paths(
        self,
        image_paths: list[str | Path],
        cache_key: str | None = None,
    ) -> np.ndarray:
        """Load images from paths and compute embeddings.

        Returns:
            (N, D) float32 array of L2-normalized embeddings.
        """
        if cache_key:
            cached = self._load_cache(cache_key)
            if cached is not None:
                return cached

        images = [Image.open(p).convert("RGB") for p in image_paths]
        return self.compute(images, cache_key=cache_key)

    def _load_cache(self, cache_key: str) -> np.ndarray | None:
        if self.cache_dir is None:
            return None
        cache_path = self.cache_dir / f"{cache_key}.npy"
        if cache_path.exists():
            return np.load(cache_path)
        return None

    def _save_cache(self, cache_key: str, embeddings: np.ndarray) -> None:
        if self.cache_dir is None:
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = self.cache_dir / f"{cache_key}.npy"
        np.save(cache_path, embeddings)

"""Error types for inpainting operations."""

from __future__ import annotations


class InpaintingError(Exception):
    """Base exception for inpainting operations."""


class InpaintingAPIError(InpaintingError):
    """Raised when an inpainting API call fails."""

    def __init__(self, provider: str, message: str, retryable: bool = False) -> None:
        self.provider = provider
        self.retryable = retryable
        super().__init__(f"[{provider}] {message}")

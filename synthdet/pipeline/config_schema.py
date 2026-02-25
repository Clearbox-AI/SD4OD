"""Pipeline configuration schema.

Wraps SynthDetConfig with orchestration-specific knobs (method selection,
train/valid split ratio, validation toggle, etc.).
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator

from synthdet.config import SynthDetConfig

VALID_METHODS = frozenset({
    "compositor",
    "inpainting",
    "generative_compositor",
    "modify_annotate",
})


class PipelineConfig(BaseModel):
    """Top-level pipeline configuration."""

    synthdet: SynthDetConfig = Field(default_factory=SynthDetConfig)
    methods: list[str] = Field(
        default_factory=lambda: ["compositor"],
        description="Generation methods to run.",
    )
    train_split_ratio: float = Field(
        0.85, description="Fraction of generated records for the train split."
    )
    validate_output: bool = Field(True, description="Run dataset validation after writing.")
    augment: bool = Field(False, description="Apply classical augmentation.")
    dry_run: bool = Field(False, description="Estimate cost without calling APIs.")

    @field_validator("methods")
    @classmethod
    def _check_methods(cls, v: list[str]) -> list[str]:
        for m in v:
            if m not in VALID_METHODS:
                raise ValueError(
                    f"Unknown method {m!r}. Valid methods: {sorted(VALID_METHODS)}"
                )
        return v

    @field_validator("train_split_ratio")
    @classmethod
    def _check_split(cls, v: float) -> float:
        if not 0 < v < 1:
            raise ValueError("train_split_ratio must be between 0 and 1 (exclusive)")
        return v

    @classmethod
    def from_yaml(cls, path: Path) -> PipelineConfig:
        """Load pipeline config from a YAML file.

        Top-level keys ``methods``, ``train_split_ratio``, ``validate_output``,
        ``augment``, and ``dry_run`` are peeled off as pipeline knobs; everything
        else is forwarded to ``SynthDetConfig``.
        """
        with open(path) as f:
            data = yaml.safe_load(f) or {}

        pipeline_keys = {"methods", "train_split_ratio", "validate_output", "augment", "dry_run"}
        pipeline_kwargs: dict = {}
        synthdet_kwargs: dict = {}

        for k, v in data.items():
            if k in pipeline_keys:
                pipeline_kwargs[k] = v
            else:
                synthdet_kwargs[k] = v

        if synthdet_kwargs:
            pipeline_kwargs["synthdet"] = SynthDetConfig.model_validate(synthdet_kwargs)

        return cls.model_validate(pipeline_kwargs)

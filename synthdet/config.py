"""Configuration models for SynthDet.

Pydantic v2 models with sensible defaults — works without a config file.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class AnalysisConfig(BaseModel):
    """Configuration for dataset analysis and strategy generation."""

    # Size bucket area thresholds (tiny/small boundary, small/medium, medium/large)
    size_bucket_thresholds: list[float] = [0.005, 0.02, 0.08]

    # Spatial grid dimensions for region analysis
    spatial_grid_cols: int = 3
    spatial_grid_rows: int = 3

    # Strategy targets
    negative_ratio: float = Field(0.15, description="Target fraction of negative examples")
    multiplier: float = Field(5.0, description="Target dataset size = current * multiplier")
    min_per_bucket: int = Field(50, description="Minimum annotations per size bucket")
    min_per_region: int = Field(30, description="Minimum annotations per spatial region")
    min_per_aspect_bin: int = Field(20, description="Minimum annotations per aspect ratio bin")

    # Aspect ratio histogram (log-spaced bins)
    aspect_ratio_num_bins: int = 8

    # Preferred generation method for synthesis tasks
    preferred_method: str = Field("compositor", description="Default method: compositor or inpainting")


class CompositorConfig(BaseModel):
    """Configuration for the defect compositor pipeline."""

    patch_margin: float = Field(0.15, description="Fractional padding around bbox for extraction")
    blend_mode: str = Field("mixed", description="Poisson blend mode: 'mixed' or 'normal'")
    inpaint_radius: int = Field(5, description="cv2.inpaint radius for background generation")
    inpaint_method: str = Field("telea", description="Inpaint method: 'telea' or 'ns'")
    min_patch_pixels: int = Field(16, description="Skip patches smaller than this")
    max_defects_per_image: int = Field(4, description="Max defects composited per synthetic image")
    max_placement_attempts: int = Field(20, description="Attempts to find non-overlapping placement")
    max_overlap_iou: float = Field(0.3, description="Max IoU allowed between placements")
    scale_jitter: tuple[float, float] = Field(
        (0.8, 1.2), description="Random scale factor range"
    )
    rotation_jitter: float = Field(15.0, description="Max rotation in degrees")
    valid_zone_margin: float = Field(
        0.05, description="Fractional margin to expand valid zone convex hull"
    )
    output_format: str = Field("jpg", description="Output image format")
    output_quality: int = Field(95, description="JPEG quality (1-100)")


class AugmentationConfig(BaseModel):
    """Configuration for classical data augmentation."""

    enabled: bool = True
    variants_per_image: int = 2
    horizontal_flip_p: float = 0.5
    brightness_contrast_p: float = 0.3
    hue_saturation_p: float = 0.2
    noise_p: float = 0.2
    blur_p: float = 0.15
    shift_scale_rotate_p: float = 0.3
    rotate_limit: int = 10


class InpaintingConfig(BaseModel):
    """Configuration for the API-based inpainting pipeline."""

    provider: str = Field("imagen", description="Provider: 'imagen', 'diffusers', 'openai', 'stability'")
    api_key_env_var: str = Field("GOOGLE_API_KEY", description="Env var holding the API key")
    project: str | None = Field(None, description="GCP project for Vertex AI (or GOOGLE_CLOUD_PROJECT env)")
    location: str | None = Field(None, description="GCP region for Vertex AI (default: us-central1)")
    model: str = Field("imagen-3.0-capability-001", description="Model identifier")
    guidance_scale: float = Field(75.0, description="Guidance scale for generation")
    max_defects_per_image: int = Field(3, description="Max defects inpainted per image")
    mask_shape: str = Field("rectangle", description="Mask shape: 'rectangle' or 'ellipse'")
    mask_padding: float = Field(0.1, description="Fractional padding around bbox for mask")
    max_placement_attempts: int = Field(20, description="Attempts to find non-overlapping placement")
    max_overlap_iou: float = Field(0.3, description="Max IoU allowed between placements")
    requests_per_minute: float = Field(600.0, description="API rate limit (0=unlimited)")
    max_cost_usd: float = Field(10.0, description="Safety cost limit (0=unlimited)")
    cost_per_image: float = Field(0.02, description="Cost per API call in USD")
    max_retries: int = Field(3, description="Max retries per API call")
    retry_delay_seconds: float = Field(1.0, description="Exponential backoff base")
    output_format: str = Field("jpg", description="Output image format")
    output_quality: int = Field(95, description="JPEG quality (1-100)")

    # --- Local diffusion settings (diffusers provider only) ---
    num_inference_steps: int = Field(50, description="Denoising steps (diffusers only)")
    strength: float = Field(0.75, description="Mask transform strength 0-1 (diffusers only)")
    device: str = Field("auto", description="'auto' (CUDA if available), 'cuda', 'cpu'")
    use_fp16: bool = Field(True, description="Use float16 on GPU (diffusers only)")

    # --- Placement constraints ---
    placement_region: list[float] | None = Field(
        None,
        description=(
            "Explicit valid region for defect placement as "
            "[x_min, y_min, x_max, y_max] in normalized 0-1 coordinates. "
            "When set, defects are only placed inside this rectangle. "
            "When None, the region is auto-computed from the annotation "
            "center distribution (5th-95th percentile)."
        ),
    )
    placement_percentile: float = Field(
        5.0,
        description=(
            "Percentile used for auto-computing the placement region "
            "from annotation centers. Lower = wider region, higher = tighter. "
            "Only used when placement_region is None."
        ),
    )

    # --- Prompt configuration ---
    prompt_template: str = Field(
        "{prompt}. The {class_name} should appear directly on the surface of "
        "the main object in the image, matching the existing material and "
        "lighting. Keep the rest of the image unchanged. Photorealistic.",
        description=(
            "Template applied to every inpainting prompt. "
            "Available placeholders: {prompt}, {class_name}."
        ),
    )
    class_prompts: dict[str, list[str]] = Field(
        default_factory=dict,
        description=(
            "Per-class prompt templates. Keys are class names (case-insensitive "
            "match), values are lists of prompt strings. A random prompt is "
            "chosen per defect."
        ),
    )
    default_prompts: list[str] = Field(
        default_factory=lambda: [
            "A realistic surface imperfection, subtle and natural-looking",
            "A visible mark on the object surface",
        ],
        description="Fallback prompts when no class-specific prompts are configured.",
    )


class ModifyAnnotateConfig(BaseModel):
    """Configuration for the modify-and-annotate pipeline.

    Sends clean images to Imagen controlled editing to produce damaged versions,
    then runs an auto-annotator to detect where defects appeared.
    """

    provider: str = Field("imagen", description="Modifier provider: 'imagen'")
    annotator: str = Field("grounding_dino", description="Auto-annotation backend: 'grounding_dino' or 'owlvit'")
    model: str = Field("imagen-3.0-capability-001", description="Imagen model for editing")
    control_type: str = Field("CONTROL_TYPE_CANNY", description="Control reference type (preserves edges/structure)")
    confidence_threshold: float = Field(0.25, description="Annotator detection threshold")

    class_prompts: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Per-class damage prompts. Keys are class names, values are prompt lists.",
    )
    default_prompts: list[str] = Field(
        default_factory=lambda: [
            "Add realistic surface damage and defects to this laptop top cover",
            "Add visible scratches and marks to this laptop surface",
        ],
        description="Fallback prompts when no class-specific prompts are configured.",
    )

    requests_per_minute: float = Field(15.0, description="API rate limit")
    max_cost_usd: float = Field(10.0, description="Safety cost limit (0=unlimited)")
    cost_per_image: float = Field(0.02, description="Cost per API call in USD")
    max_retries: int = Field(3, description="Max retries per API call")
    retry_delay_seconds: float = Field(1.0, description="Exponential backoff base")

    sam_refine: bool = Field(False, description="Enable SAM post-processing on detected bboxes")
    clip_verify: bool = Field(False, description="Enable CLIP-based filtering of detections")
    min_clip_score: float = Field(0.4, description="Min CLIP score to keep a detection")

    output_format: str = Field("jpg", description="Output image format")
    output_quality: int = Field(95, description="JPEG quality (1-100)")

    # Provider auth (same pattern as InpaintingConfig)
    api_key_env_var: str = Field("GOOGLE_API_KEY", description="Env var holding the API key")
    project: str | None = Field(None, description="GCP project for Vertex AI")
    location: str | None = Field(None, description="GCP region for Vertex AI")


class AnnotationConfig(BaseModel):
    """Configuration for auto-annotation (Grounding DINO / OWL-ViT)."""

    annotator: str = Field("grounding_dino", description="'grounding_dino' or 'owlvit'")
    model: str = Field("IDEA-Research/grounding-dino-tiny", description="HF model ID")
    device: str = Field("auto", description="'auto', 'cuda', 'cpu'")
    confidence_threshold: float = Field(0.3, description="Min detection confidence")
    box_threshold: float = Field(0.25, description="Box score threshold (grounding_dino only)")


class SAMConfig(BaseModel):
    """Configuration for SAM-based bbox refinement."""

    enabled: bool = Field(False, description="Enable SAM refinement post-annotation")
    model: str = Field("facebook/sam-vit-base", description="SAM model ID")
    device: str = Field("auto")
    iou_threshold: float = Field(0.5, description="Min IoU original vs refined to accept")
    margin: float = Field(0.05, description="Fractional margin on mask-derived bbox")


class VerifierConfig(BaseModel):
    """Configuration for CLIP-based annotation verification."""

    enabled: bool = Field(False, description="Enable CLIP-based verification")
    model: str = Field("openai/clip-vit-base-patch32", description="CLIP model ID")
    device: str = Field("auto")
    min_confidence: float = Field(0.5, description="Min CLIP score to keep annotation")


class QualityMonitoringConfig(BaseModel):
    """Configuration for SPC-based quality monitoring."""

    control_limit_sigma: float = Field(3.0, description="Sigma for Shewhart UCL/LCL")
    activation_layers: list[str] = Field(
        default_factory=lambda: ["backbone.stage3", "backbone.stage4", "neck.fpn"],
        description="Model layers to monitor activation distributions",
    )
    snapshot_percentiles: list[int] = Field(
        default_factory=lambda: [5, 25, 50, 75, 95],
        description="Percentiles to record in activation snapshots",
    )
    trend_window: int = Field(7, description="Consecutive points for Western Electric trend rule")


class SynthDetConfig(BaseModel):
    """Top-level configuration for SynthDet."""

    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)
    compositor: CompositorConfig = Field(default_factory=CompositorConfig)
    inpainting: InpaintingConfig = Field(default_factory=InpaintingConfig)
    modify_annotate: ModifyAnnotateConfig = Field(default_factory=ModifyAnnotateConfig)
    augmentation: AugmentationConfig = Field(default_factory=AugmentationConfig)
    annotation: AnnotationConfig = Field(default_factory=AnnotationConfig)
    sam: SAMConfig = Field(default_factory=SAMConfig)
    verifier: VerifierConfig = Field(default_factory=VerifierConfig)
    quality_monitoring: QualityMonitoringConfig = Field(default_factory=QualityMonitoringConfig)

    @classmethod
    def from_yaml(cls, path: Path) -> SynthDetConfig:
        """Load configuration from a YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls.model_validate(data)

    @classmethod
    def default(cls) -> SynthDetConfig:
        """Return configuration with all defaults."""
        return cls()

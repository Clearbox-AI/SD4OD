# SynthDet Roadmap

## Completed Phases

### Phase 1: Foundation + Analysis
- Core types (`BBox`, `ImageRecord`, `Dataset`, `SynthesisStrategy`)
- YOLO dataset loader, statistics, strategy generation
- CLI: `python -m synthdet.analysis`

### Phase 2: Compositor Pipeline
- Defect patch extraction, clean background generation, Poisson blending
- Bbox-aware classical augmentation (albumentations)
- YOLO label writer
- CLI: `python -m synthdet.generate`

### Phase 3: API-Based Inpainting (Google Imagen 3)
- Mask placement, inpainting pipeline, Imagen 3 provider
- Local Stable Diffusion alternative (zero-cost)
- Rate limiting, cost estimation, dry-run mode

### Phase 3.5: Alternative Generation Pipelines
- **Generative Compositor**: API-generated isolated patches + Poisson blending
- **Modify-and-Annotate**: Whole-image transformation + auto-annotation

### Phase 4: Auto-Annotation + Verification
- Grounding DINO and OWL-ViT zero-shot detection
- SAM-based bbox refinement
- CLIP-based annotation quality scoring

### Phase 5: Pipeline Orchestration
- End-to-end pipeline with config-driven method selection
- Dataset validation (bbox sanity, image integrity, split balance)
- CLI: `python -m synthdet.pipeline`

### Phase 6a: SPC Quality Monitoring + Embedding Diversity
- Shewhart X-bar control charts with Western Electric rules
- PyTorch forward-hook activation capture
- CLIP/DINOv2 embedding-based diversity analysis

### Phase 6b: Active Learning Loop
- YOLO trainer wrapper, per-bucket/region evaluation
- N-iteration generate → train → evaluate → refine coordinator

### Phase 6c: Copy-Paste Augmentation + Web Acquisition
- **Copy-paste augmentation**: Paste defect patches onto existing annotated images with Poisson blending (complements compositor which uses clean backgrounds)
- **Web scraper**: Acquire background images from Google/Bing via icrawler with resolution filtering and perceptual deduplication
- **Relevance filter**: CLIP-based filtering to keep only domain-relevant acquired images

## Remaining Work

### Mosaic / MixUp Augmentation
- `augment/mosaic.py`: 4-image mosaic and MixUp for detection
- Standard YOLO augmentation technique for small-object performance

### Style Transfer
- `augment/style.py`: Neural style transfer for domain adaptation
- Transfer visual styles between datasets to improve generalization

## Future Directions

- **Multi-class expansion**: Extend beyond scratches to stains, broken bezels, dents
- **Dataset scaling**: Larger base datasets with more diverse laptop models
- **Model zoo**: Pre-trained weights for common defect detection tasks
- **Evaluation dashboard**: Rich visualization of training progress, data quality, and model performance
- **Edge deployment**: Model optimization (quantization, pruning) for real-time grading

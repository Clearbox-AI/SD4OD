"""Inpainting and generation provider implementations.

Available providers:
- ``imagen``: Google Imagen 3 inpainting via Vertex AI (API-based, $0.02/image)
- ``imagen_generate``: Google Imagen 3 image generation via Vertex AI ($0.04/image)
  — generates isolated defect patches for compositing (no inpainting conflict)
- ``imagen_modifier``: Google Imagen 3 controlled editing via Vertex AI ($0.02/image)
  — whole-image transformation for modify-and-annotate pipeline
- ``diffusers``: Local HuggingFace diffusers pipeline (free, requires GPU)
"""

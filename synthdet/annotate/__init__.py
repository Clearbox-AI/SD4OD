"""Annotation generation and verification.

Modules:
    base             — Annotator and AnnotationVerifier protocols
    grounding_dino   — Grounding DINO zero-shot detection (Phase 4)
    owlvit           — OWL-ViT zero-shot detection (Phase 4)
    sam_refiner      — SAM-based bbox refinement (Phase 4)
    verifier         — CLIP-based annotation verification (Phase 4)
    compositor_annotator — Annotations from known composite placements
    yolo_writer      — Write YOLO format labels
"""

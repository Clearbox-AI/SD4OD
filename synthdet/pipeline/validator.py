"""Standalone YOLO dataset validator.

Checks structural integrity of a YOLO dataset on disk:
- bbox coordinates in [0, 1] with positive width/height
- label ↔ image pairing (missing labels or orphan labels)
- split balance warnings
- data.yaml existence and validity
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class ValidationIssue:
    """A single validation finding."""

    severity: str  # "error" | "warning"
    category: str  # "bbox" | "image" | "pairing" | "balance" | "structure"
    file: str | None
    message: str


@dataclass
class ValidationReport:
    """Aggregated validation results for a dataset."""

    issues: list[ValidationIssue] = field(default_factory=list)
    total_images: int = 0
    total_labels: int = 0

    @property
    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "warning"]

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0

    def summary(self) -> str:
        lines = [
            f"Images: {self.total_images}, Labels: {self.total_labels}",
            f"Errors: {len(self.errors)}, Warnings: {len(self.warnings)}",
        ]
        if self.is_valid:
            lines.append("Status: VALID")
        else:
            lines.append("Status: INVALID")
            for err in self.errors[:10]:
                lines.append(f"  - [{err.category}] {err.message}")
        return "\n".join(lines)


def validate_bbox_line(line: str, file_path: str) -> list[ValidationIssue]:
    """Validate a single YOLO label line.

    Expected format: ``class_id x_center y_center width height``
    """
    issues: list[ValidationIssue] = []
    stripped = line.strip()
    if not stripped:
        return issues

    parts = stripped.split()
    if len(parts) != 5:
        issues.append(ValidationIssue(
            severity="error",
            category="bbox",
            file=file_path,
            message=f"Expected 5 fields, got {len(parts)}: {stripped!r}",
        ))
        return issues

    try:
        class_id = int(parts[0])
    except ValueError:
        issues.append(ValidationIssue(
            severity="error",
            category="bbox",
            file=file_path,
            message=f"Non-integer class_id: {parts[0]!r}",
        ))
        return issues

    if class_id < 0:
        issues.append(ValidationIssue(
            severity="error",
            category="bbox",
            file=file_path,
            message=f"Negative class_id: {class_id}",
        ))

    for i, name in enumerate(["x_center", "y_center", "width", "height"], start=1):
        try:
            val = float(parts[i])
        except ValueError:
            issues.append(ValidationIssue(
                severity="error",
                category="bbox",
                file=file_path,
                message=f"Non-numeric {name}: {parts[i]!r}",
            ))
            continue

        if not 0 <= val <= 1:
            issues.append(ValidationIssue(
                severity="error",
                category="bbox",
                file=file_path,
                message=f"{name}={val:.4f} outside [0, 1]",
            ))

        if name in ("width", "height") and val <= 0:
            issues.append(ValidationIssue(
                severity="error",
                category="bbox",
                file=file_path,
                message=f"{name}={val:.4f} must be positive",
            ))

    return issues


def validate_split_balance(
    split_counts: dict[str, int],
    min_ratio: float = 0.05,
    max_ratio: float = 0.95,
) -> list[ValidationIssue]:
    """Warn if train/valid split ratio is extreme."""
    issues: list[ValidationIssue] = []
    total = sum(split_counts.values())
    if total == 0:
        issues.append(ValidationIssue(
            severity="warning",
            category="balance",
            file=None,
            message="Dataset is empty (0 total images across all splits)",
        ))
        return issues

    for split, count in split_counts.items():
        ratio = count / total
        if count > 0 and ratio < min_ratio:
            issues.append(ValidationIssue(
                severity="warning",
                category="balance",
                file=None,
                message=f"Split '{split}' has only {count}/{total} images ({ratio:.1%})",
            ))
        if ratio > max_ratio and len(split_counts) > 1:
            issues.append(ValidationIssue(
                severity="warning",
                category="balance",
                file=None,
                message=f"Split '{split}' has {count}/{total} images ({ratio:.1%}) — very imbalanced",
            ))

    return issues


def validate_dataset(
    dataset_dir: Path,
    *,
    check_images: bool = False,
) -> ValidationReport:
    """Validate a YOLO dataset on disk.

    Args:
        dataset_dir: Root directory containing data.yaml and split folders.
        check_images: If True, verify that image files are readable (slower).

    Returns:
        ValidationReport with all findings.
    """
    report = ValidationReport()

    # Check data.yaml
    data_yaml = dataset_dir / "data.yaml"
    if not data_yaml.is_file():
        report.issues.append(ValidationIssue(
            severity="error",
            category="structure",
            file=str(data_yaml),
            message="data.yaml not found",
        ))
        return report

    try:
        with open(data_yaml) as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        report.issues.append(ValidationIssue(
            severity="error",
            category="structure",
            file=str(data_yaml),
            message=f"Invalid YAML: {e}",
        ))
        return report

    if not isinstance(data, dict) or "names" not in data:
        report.issues.append(ValidationIssue(
            severity="error",
            category="structure",
            file=str(data_yaml),
            message="data.yaml missing 'names' field",
        ))
        return report

    # Discover splits
    split_counts: dict[str, int] = {}
    for split in ("train", "valid", "test"):
        images_dir = dataset_dir / split / "images"
        labels_dir = dataset_dir / split / "labels"
        if not images_dir.is_dir():
            continue

        image_files = {
            p.stem: p
            for p in images_dir.iterdir()
            if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        }
        label_files = {
            p.stem: p
            for p in labels_dir.iterdir()
            if p.suffix == ".txt"
        } if labels_dir.is_dir() else {}

        report.total_images += len(image_files)
        report.total_labels += len(label_files)
        split_counts[split] = len(image_files)

        # Check image→label pairing
        for stem in image_files:
            if stem not in label_files:
                report.issues.append(ValidationIssue(
                    severity="warning",
                    category="pairing",
                    file=str(images_dir / f"{stem}.*"),
                    message=f"Image '{stem}' has no corresponding label file",
                ))

        # Check label→image pairing
        for stem in label_files:
            if stem not in image_files:
                report.issues.append(ValidationIssue(
                    severity="warning",
                    category="pairing",
                    file=str(labels_dir / f"{stem}.txt"),
                    message=f"Label '{stem}.txt' has no corresponding image",
                ))

        # Validate label contents
        for stem, label_path in label_files.items():
            text = label_path.read_text()
            for line in text.splitlines():
                if line.strip():
                    report.issues.extend(
                        validate_bbox_line(line, str(label_path))
                    )

        # Optional image readability check
        if check_images:
            import cv2

            for stem, img_path in image_files.items():
                img = cv2.imread(str(img_path))
                if img is None:
                    report.issues.append(ValidationIssue(
                        severity="error",
                        category="image",
                        file=str(img_path),
                        message=f"Image '{img_path.name}' is unreadable",
                    ))

    # Split balance
    if split_counts:
        report.issues.extend(validate_split_balance(split_counts))

    return report

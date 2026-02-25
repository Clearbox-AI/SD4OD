"""Reusable matplotlib visualization helpers for SynthDet demos."""

from __future__ import annotations

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

BUCKET_ORDER = ["tiny", "small", "medium", "large"]
BUCKET_COLORS = {
    "tiny": "#e74c3c",
    "small": "#f39c12",
    "medium": "#3498db",
    "large": "#2ecc71",
}
REGION_ORDER = [
    "top_left",
    "top_center",
    "top_right",
    "middle_left",
    "middle_center",
    "middle_right",
    "bottom_left",
    "bottom_center",
    "bottom_right",
]
REGION_LABELS = [
    ["TL", "TC", "TR"],
    ["ML", "MC", "MR"],
    ["BL", "BC", "BR"],
]

# Color palette for bboxes per class
BBOX_COLORS = [
    "#e74c3c",
    "#3498db",
    "#2ecc71",
    "#f39c12",
    "#9b59b6",
    "#1abc9c",
]


def _ensure_ax(ax):
    if ax is None:
        _, ax = plt.subplots()
    return ax


def show_image_with_bboxes(
    image_rgb: np.ndarray,
    bboxes,
    class_names: list[str] | None = None,
    ax=None,
    title: str | None = None,
):
    """Display image with bbox rectangles overlaid.

    bboxes: list of BBox objects (with x_center, y_center, width, height)
    or list of dicts with those keys. All coords normalized 0-1.
    """
    ax = _ensure_ax(ax)
    h, w = image_rgb.shape[:2]
    ax.imshow(image_rgb)

    for bbox in bboxes:
        if hasattr(bbox, "x_center"):
            xc, yc, bw, bh = bbox.x_center, bbox.y_center, bbox.width, bbox.height
            cid = bbox.class_id
        else:
            xc, yc, bw, bh = bbox["x_center"], bbox["y_center"], bbox["width"], bbox["height"]
            cid = bbox.get("class_id", 0)

        x1 = (xc - bw / 2) * w
        y1 = (yc - bh / 2) * h
        pw = bw * w
        ph = bh * h
        color = BBOX_COLORS[cid % len(BBOX_COLORS)]

        rect = mpatches.FancyBboxPatch(
            (x1, y1),
            pw,
            ph,
            linewidth=2,
            edgecolor=color,
            facecolor="none",
            boxstyle="round,pad=0",
        )
        ax.add_patch(rect)

        label = class_names[cid] if class_names and cid < len(class_names) else str(cid)
        ax.text(
            x1, y1 - 3, label, fontsize=7, color="white",
            bbox=dict(facecolor=color, alpha=0.7, pad=1, edgecolor="none"),
        )

    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=10)


def plot_size_bucket_distribution(
    bucket_counts: dict,
    ax=None,
    title: str = "Size Buckets",
    target_line: int | None = None,
):
    """Horizontal bar chart of bbox counts per size bucket, color-coded."""
    ax = _ensure_ax(ax)
    buckets = BUCKET_ORDER
    counts = [bucket_counts.get(b, 0) for b in buckets]
    colors = [BUCKET_COLORS[b] for b in buckets]
    y_pos = np.arange(len(buckets))

    bars = ax.barh(y_pos, counts, color=colors, edgecolor="white", height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([b.capitalize() for b in buckets])
    ax.set_xlabel("Count")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.invert_yaxis()

    for bar, count in zip(bars, counts):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                str(count), va="center", fontsize=9)

    if target_line is not None:
        ax.axvline(target_line, color="#888", linestyle="--", linewidth=1, label=f"Target ({target_line})")
        ax.legend(fontsize=8)


def plot_spatial_heatmap(
    region_counts: dict,
    ax=None,
    title: str = "Spatial Distribution",
):
    """3x3 annotated heatmap of bbox center locations."""
    ax = _ensure_ax(ax)
    grid = np.zeros((3, 3))
    for i, row in enumerate(REGION_LABELS):
        for j, _ in enumerate(row):
            region_key = REGION_ORDER[i * 3 + j]
            grid[i, j] = region_counts.get(region_key, 0)

    im = ax.imshow(grid, cmap="YlOrRd", aspect="equal")
    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 1, 2])
    ax.set_xticklabels(["Left", "Center", "Right"], fontsize=8)
    ax.set_yticklabels(["Top", "Middle", "Bottom"], fontsize=8)
    ax.set_title(title, fontsize=11, fontweight="bold")

    for i in range(3):
        for j in range(3):
            val = int(grid[i, j])
            color = "white" if val > grid.max() * 0.6 else "black"
            ax.text(j, i, str(val), ha="center", va="center", fontsize=12, fontweight="bold", color=color)

    plt.colorbar(im, ax=ax, shrink=0.8)


def plot_annotations_per_image(histogram: dict, ax=None):
    """Bar chart of annotation count distribution with mean line."""
    ax = _ensure_ax(ax)
    keys = sorted(histogram.keys())
    counts = [histogram[k] for k in keys]

    total_images = sum(counts)
    total_annots = sum(k * c for k, c in zip(keys, counts))
    mean_val = total_annots / total_images if total_images else 0

    ax.bar([str(k) for k in keys], counts, color="#3498db", edgecolor="white")
    ax.axvline(mean_val, color="#e74c3c", linestyle="--", linewidth=1.5, label=f"Mean ({mean_val:.1f})")
    ax.set_xlabel("Annotations per image")
    ax.set_ylabel("Image count")
    ax.set_title("Annotations per Image", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)


def plot_generation_tasks(tasks: list, ax=None):
    """Horizontal bars: tasks sorted by priority, colored by priority level."""
    ax = _ensure_ax(ax)
    if not tasks:
        ax.text(0.5, 0.5, "No tasks", ha="center", va="center", transform=ax.transAxes)
        return

    # Sort by priority descending
    sorted_tasks = sorted(tasks, key=lambda t: t.get("priority", 0) if isinstance(t, dict) else t.priority, reverse=True)

    labels = []
    widths = []
    priorities = []
    for t in sorted_tasks:
        if isinstance(t, dict):
            labels.append(t.get("rationale", t.get("task_id", ""))[:40])
            widths.append(t.get("num_images", 0))
            priorities.append(t.get("priority", 0))
        else:
            labels.append(t.rationale[:40] if t.rationale else t.task_id)
            widths.append(t.num_images)
            priorities.append(t.priority)

    y_pos = np.arange(len(labels))
    cmap = plt.cm.RdYlGn
    colors = [cmap(p) for p in priorities]

    bars = ax.barh(y_pos, widths, color=colors, edgecolor="white", height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Images to generate")
    ax.set_title("Generation Tasks (by priority)", fontsize=11, fontweight="bold")
    ax.invert_yaxis()

    for bar, w, p in zip(bars, widths, priorities):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f"{w} (p={p:.1f})", va="center", fontsize=8)


def plot_training_curve(metrics_history: list[dict], ax=None):
    """Dual y-axis: box_loss (left, descending) and mAP50 (right, ascending)."""
    ax = _ensure_ax(ax)
    epochs = [m["epoch"] for m in metrics_history]
    losses = [m["box_loss"] for m in metrics_history]
    maps = [m["mAP50"] for m in metrics_history]

    color_loss = "#e74c3c"
    color_map = "#2ecc71"

    ax.plot(epochs, losses, color=color_loss, linewidth=2, label="Box Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Box Loss", color=color_loss)
    ax.tick_params(axis="y", labelcolor=color_loss)

    ax2 = ax.twinx()
    ax2.plot(epochs, maps, color=color_map, linewidth=2, label="mAP50")
    ax2.set_ylabel("mAP50", color=color_map)
    ax2.tick_params(axis="y", labelcolor=color_map)
    ax2.set_ylim(0, 1)

    ax.set_title("Training Curves", fontsize=11, fontweight="bold")

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="center right", fontsize=8)


def plot_active_learning_progress(iterations: list[dict], ax=None):
    """Line chart: mAP50 vs iteration with delta annotations."""
    ax = _ensure_ax(ax)
    iters = [it["iteration"] for it in iterations]
    maps = [it["map50"] for it in iterations]

    ax.plot(iters, maps, "o-", color="#3498db", linewidth=2, markersize=8)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("mAP50")
    ax.set_title("Active Learning Progress", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    for it in iterations:
        added = it.get("records_added", 0)
        improvement = it.get("improvement", 0)
        if added > 0:
            ax.annotate(
                f"+{added} imgs\n+{improvement:.1%}",
                xy=(it["iteration"], it["map50"]),
                xytext=(0, 15),
                textcoords="offset points",
                ha="center",
                fontsize=7,
                color="#666",
            )


def plot_bucket_recall_comparison(before: dict, after: dict, ax=None):
    """Grouped bar chart: per-bucket recall, before vs after."""
    ax = _ensure_ax(ax)
    buckets = BUCKET_ORDER
    x = np.arange(len(buckets))
    width = 0.35

    vals_before = [before.get(b, 0) for b in buckets]
    vals_after = [after.get(b, 0) for b in buckets]

    ax.bar(x - width / 2, vals_before, width, label="Baseline", color="#e74c3c", alpha=0.8)
    ax.bar(x + width / 2, vals_after, width, label="Augmented", color="#2ecc71", alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels([b.capitalize() for b in buckets])
    ax.set_ylabel("Recall")
    ax.set_ylim(0, 1)
    ax.set_title("Per-Bucket Recall", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)


def plot_region_recall_heatmap(
    region_map: dict,
    ax=None,
    title: str = "Per-Region Recall",
):
    """3x3 heatmap, 0-1 scale with text annotations."""
    ax = _ensure_ax(ax)
    grid = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            region_key = REGION_ORDER[i * 3 + j]
            grid[i, j] = region_map.get(region_key, 0)

    im = ax.imshow(grid, cmap="RdYlGn", vmin=0, vmax=1, aspect="equal")
    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 1, 2])
    ax.set_xticklabels(["Left", "Center", "Right"], fontsize=8)
    ax.set_yticklabels(["Top", "Middle", "Bottom"], fontsize=8)
    ax.set_title(title, fontsize=11, fontweight="bold")

    for i in range(3):
        for j in range(3):
            val = grid[i, j]
            color = "white" if val < 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=11, fontweight="bold", color=color)

    plt.colorbar(im, ax=ax, shrink=0.8)

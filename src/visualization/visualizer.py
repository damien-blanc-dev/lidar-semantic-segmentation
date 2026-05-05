"""
src/visualization/visualizer.py
────────────────────────────────
Visualization utilities for LiDAR point clouds.

Two rendering backends:
  1. Open3D  — interactive 3D viewer (needs a display / GUI)
  2. Matplotlib — static 2D/3D figures, safe in notebooks and headless envs

Key design choices:
  - Functions are stateless and return figures/None, never mutate input clouds.
  - Colors are always driven by CLASS_COLORS to stay consistent across all views.
  - We normalize XYZ to a centered bounding box so the interactive viewer is
    always well-framed regardless of the Lambert-93 absolute coordinates.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from src.data.loader import CLASS_COLORS, CLASS_NAMES, NUM_CLASSES, PointCloud

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  Color helpers
# ─────────────────────────────────────────────────────────────────────────────

def labels_to_colors(labels: np.ndarray) -> np.ndarray:
    """Map integer class labels → (N, 3) float32 RGB colors in [0, 1]."""
    colors = CLASS_COLORS[np.clip(labels, 0, len(CLASS_COLORS) - 1)]
    return colors.astype(np.float32)


def reflectance_to_colors(reflectance: np.ndarray, colormap: str = "plasma") -> np.ndarray:
    """Map scalar reflectance values → (N, 3) float32 RGB using a matplotlib colormap."""
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap(colormap)
    norm = (reflectance - reflectance.min()) / (reflectance.max() - reflectance.min() + 1e-8)
    rgba = cmap(norm)
    return rgba[:, :3].astype(np.float32)


def height_to_colors(xyz: np.ndarray, colormap: str = "viridis") -> np.ndarray:
    """Color points by their Z (height) value — useful for quick sanity checks."""
    return reflectance_to_colors(xyz[:, 2], colormap=colormap)


# ─────────────────────────────────────────────────────────────────────────────
#  Open3D interactive viewer
# ─────────────────────────────────────────────────────────────────────────────

def _to_o3d(xyz: np.ndarray, colors: np.ndarray):
    """Build an Open3D PointCloud from xyz + RGB colors."""
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(np.clip(colors, 0, 1).astype(np.float64))
    return pcd


def _center_xyz(xyz: np.ndarray) -> np.ndarray:
    """Subtract centroid so the cloud is centered at origin for display."""
    return xyz - xyz.mean(axis=0)


def show_pointcloud(
    pc: PointCloud,
    mode: str = "labels",
    window_name: str = "Point Cloud",
    point_size: float = 2.0,
) -> None:
    """Open an interactive Open3D viewer.

    Parameters
    ----------
    pc          : PointCloud to display
    mode        : "labels"     — color by semantic class (requires pc.has_labels)
                  "reflectance"  — color by laser reflectance
                  "height"     — color by Z coordinate
    window_name : title bar string
    point_size  : render point size in pixels
    """
    import open3d as o3d

    xyz = _center_xyz(pc.xyz)

    if mode == "labels":
        if not pc.has_labels:
            logger.warning("No labels found, falling back to height coloring.")
            mode = "height"
        else:
            colors = labels_to_colors(pc.labels)
    if mode == "reflectance":
        if pc.reflectance is None:
            logger.warning("No reflectance field, falling back to height coloring.")
            mode = "height"
        else:
            colors = intensity_to_colors(pc.reflectance)
    if mode == "height":
        colors = height_to_colors(xyz)

    o3d_pc = _to_o3d(xyz, colors)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name, width=1280, height=720)
    vis.add_geometry(o3d_pc)

    opt = vis.get_render_option()
    opt.point_size = point_size
    opt.background_color = np.array([0.05, 0.05, 0.05])  # near-black background

    vis.run()
    vis.destroy_window()


def show_comparison(
    pc: PointCloud,
    predicted_labels: np.ndarray,
    window_name: str = "GT vs Prediction",
) -> None:
    """Side-by-side Open3D view: ground truth (left) vs. predictions (right).

    The predicted cloud is offset by the bounding box width so both clouds
    appear next to each other in the same window.
    """
    import open3d as o3d

    if not pc.has_labels:
        raise ValueError("PointCloud must have ground-truth labels for comparison.")

    xyz = _center_xyz(pc.xyz)
    x_offset = (xyz[:, 0].max() - xyz[:, 0].min()) * 1.1

    gt_colors = labels_to_colors(pc.labels)
    pred_colors = labels_to_colors(predicted_labels)

    gt_pcd = _to_o3d(xyz, gt_colors)
    pred_pcd = _to_o3d(xyz + np.array([x_offset, 0, 0]), pred_colors)

    o3d.visualization.draw_geometries(
        [gt_pcd, pred_pcd],
        window_name=window_name,
        width=1600, height=720,
        point_show_normal=False,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Matplotlib static figures (notebook-friendly)
# ─────────────────────────────────────────────────────────────────────────────

def plot_class_distribution(
    pc: PointCloud,
    save_path: Optional[str | Path] = None,
    figsize: tuple = (10, 5),
):
    """Horizontal bar chart of points per semantic class.

    This is the first thing to look at on a new dataset — mirrors what you'd
    do with a class histogram in CT segmentation to spot imbalance early.
    """
    import matplotlib.pyplot as plt

    if not pc.has_labels:
        raise ValueError("PointCloud must have labels.")

    counts = np.bincount(pc.labels, minlength=len(CLASS_NAMES))
    percentages = counts / counts.sum() * 100

    fig, ax = plt.subplots(figsize=figsize)
    y_pos = np.arange(len(CLASS_NAMES))

    bars = ax.barh(
        y_pos,
        percentages,
        color=[CLASS_COLORS[i] for i in range(len(CLASS_NAMES))],
        edgecolor="white",
        linewidth=0.5,
    )

    # Annotate with point counts
    for bar, count, pct in zip(bars, counts, percentages):
        ax.text(
            pct + 0.3, bar.get_y() + bar.get_height() / 2,
            f"{count:,} ({pct:.1f}%)",
            va="center", ha="left", fontsize=9, color="white",
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(CLASS_NAMES, fontsize=10)
    ax.set_xlabel("Proportion of points (%)", fontsize=11)
    ax.set_title(
        f"Class distribution — {pc.num_points:,} points"
        + (f"\n{pc.path.name}" if pc.path else ""),
        fontsize=12, fontweight="bold",
    )
    ax.set_facecolor("#1a1a2e")
    fig.patch.set_facecolor("#1a1a2e")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.title.set_color("white")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["bottom", "left"]].set_color("#444")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        logger.info(f"Saved class distribution to {save_path}")
    return fig


def plot_topdown(
    pc: PointCloud,
    mode: str = "labels",
    subsample: int = 200_000,
    point_size: float = 0.5,
    save_path: Optional[str | Path] = None,
    figsize: tuple = (14, 8),
):
    """Top-down (bird's eye) 2D scatter plot of the point cloud.

    Subsampling is applied for speed — the visual is representative even at
    200k points for a scene of several million.

    Parameters
    ----------
    mode      : "labels" | "reflectance" | "height"
    subsample : max number of points to render
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    xyz = _center_xyz(pc.xyz)
    n = len(xyz)

    if n > subsample:
        idx = np.random.choice(n, subsample, replace=False)
        xyz_plot = xyz[idx]
        labels_plot = pc.labels[idx] if pc.has_labels else None
        reflectance_plot = pc.reflectance[idx] if pc.reflectance is not None else None
    else:
        xyz_plot = xyz
        labels_plot = pc.labels if pc.has_labels else None
        reflectance_plot = pc.reflectance

    if mode == "labels" and labels_plot is not None:
        colors = labels_to_colors(labels_plot)
        legend_handles = [
            Patch(color=CLASS_COLORS[i], label=CLASS_NAMES[i])
            for i in range(len(CLASS_NAMES))
            if np.any(labels_plot == i)
        ]
    elif mode == "reflectance" and reflectance_plot is not None:
        colors = reflectance_to_colors(reflectance_plot)
        legend_handles = []
    else:
        colors = height_to_colors(xyz_plot)
        legend_handles = []

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(
        xyz_plot[:, 0], xyz_plot[:, 1],
        c=colors, s=point_size, linewidths=0,
    )

    if legend_handles:
        ax.legend(
            handles=legend_handles, loc="upper right",
            framealpha=0.3, labelcolor="white",
            facecolor="#1a1a2e", edgecolor="#444",
            fontsize=9, markerscale=2,
        )

    ax.set_aspect("equal")
    ax.set_xlabel("X (m)", fontsize=10, color="white")
    ax.set_ylabel("Y (m)", fontsize=10, color="white")
    ax.set_title(
        f"Top-down view — {mode}"
        + (f" | {pc.path.name}" if pc.path else ""),
        fontsize=12, fontweight="bold", color="white",
    )
    ax.set_facecolor("#0d0d1a")
    fig.patch.set_facecolor("#0d0d1a")
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#333")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        logger.info(f"Saved top-down view to {save_path}")
    return fig


def plot_class_samples(
    pc: PointCloud,
    n_cols: int = 3,
    points_per_class: int = 5_000,
    save_path: Optional[str | Path] = None,
):
    """Grid of top-down crops, one per semantic class.

    Useful for quickly verifying that label remapping is correct — the same
    sanity check you'd run on a medical segmentation dataset before training.
    """
    import matplotlib.pyplot as plt

    if not pc.has_labels:
        raise ValueError("PointCloud must have labels.")

    present_classes = np.unique(pc.labels)
    n_rows = int(np.ceil(len(present_classes) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = np.array(axes).flatten()

    xyz_c = _center_xyz(pc.xyz)

    for ax_idx, cls_id in enumerate(present_classes):
        mask = pc.labels == cls_id
        cls_xyz = xyz_c[mask]

        if len(cls_xyz) > points_per_class:
            idx = np.random.choice(len(cls_xyz), points_per_class, replace=False)
            cls_xyz = cls_xyz[idx]

        axes[ax_idx].scatter(
            cls_xyz[:, 0], cls_xyz[:, 1],
            c=[CLASS_COLORS[cls_id]], s=0.5, linewidths=0,
        )
        axes[ax_idx].set_title(
            f"{CLASS_NAMES[cls_id]}\n({mask.sum():,} pts)",
            fontsize=9, color="white",
        )
        axes[ax_idx].set_aspect("equal")
        axes[ax_idx].set_facecolor("#0d0d1a")
        axes[ax_idx].tick_params(colors="white", labelsize=6)
        axes[ax_idx].spines[:].set_color("#333")

    for ax in axes[len(present_classes):]:
        ax.set_visible(False)

    fig.patch.set_facecolor("#0d0d1a")
    fig.suptitle("Per-class point samples (top-down)", fontsize=13,
                 fontweight="bold", color="white", y=1.01)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        logger.info(f"Saved class samples to {save_path}")
    return fig

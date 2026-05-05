"""
app/demo.py
────────────
Streamlit interactive demo for LiDAR semantic segmentation.

Launch with:
    streamlit run app/demo.py

Features:
  - Upload a preprocessed .npy file OR select a pre-loaded scan
  - Run inference with the trained PointNet++ model
  - Interactive top-down visualization colored by semantic class
  - Per-class metrics and distribution chart
  - Download predictions as CSV
"""

import sys
from pathlib import Path

# Project root on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import torch

from src.data.loader import CLASS_NAMES, CLASS_COLORS, NUM_CLASSES, PointCloud
from src.models.pointnet2 import PointNet2
from src.visualization.visualizer import plot_topdown, plot_class_distribution

# ─────────────────────────────────────────────────────────────────────────────
#  Page config
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="LiDAR Semantic Segmentation",
    page_icon="🌆",
    layout="wide",
    initial_sidebar_state="expanded",
)

CHECKPOINT_PATH = Path("outputs/checkpoints/pointnet2_ssg_pl3d/best.pth")
PROCESSED_DIR   = Path("data/processed")

AVAILABLE_SCANS = {
    stem: PROCESSED_DIR / stem
    for stem in ["Lille1_1", "Lille1_2", "Lille2", "Paris"]
    if (PROCESSED_DIR / stem / "points.npy").exists()
}

# ─────────────────────────────────────────────────────────────────────────────
#  Model loading (cached)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def load_model():
    if not CHECKPOINT_PATH.exists():
        return None, None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    model = PointNet2(in_channels=8, num_classes=NUM_CLASSES, dropout=0.0)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()
    return model, ckpt


@st.cache_data
def load_scan(scan_name: str):
    scan_dir = PROCESSED_DIR / scan_name
    points = np.load(scan_dir / "points.npy")
    labels = np.load(scan_dir / "labels.npy") if (scan_dir / "labels.npy").exists() else None
    return points, labels


# ─────────────────────────────────────────────────────────────────────────────
#  Inference
# ─────────────────────────────────────────────────────────────────────────────

def infer_block_sample(
    points: np.ndarray,
    model,
    device,
    n_blocks: int = 200,
    num_points: int = 4096,
    block_size: float = 4.0,
) -> np.ndarray:
    """Quick inference on a random sample of blocks (for demo speed).

    Returns per-point predictions for the sampled points only.
    """
    xy   = points[:, :2]
    N    = len(points)
    half = block_size / 2.0

    x_min, x_max = xy[:, 0].min(), xy[:, 0].max()
    y_min, y_max = xy[:, 1].min(), xy[:, 1].max()

    all_feats  = []
    all_orig   = []
    vote_counts = np.zeros((N, NUM_CLASSES), dtype=np.int32)

    for _ in range(n_blocks):
        cx = np.random.uniform(x_min + half, x_max - half)
        cy = np.random.uniform(y_min + half, y_max - half)
        mask = (
            (xy[:, 0] >= cx - half) & (xy[:, 0] < cx + half) &
            (xy[:, 1] >= cy - half) & (xy[:, 1] < cy + half)
        )
        m = mask.sum()
        if m < 64:
            continue

        block_pts = points[mask]
        chosen = np.random.choice(m, num_points, replace=(m < num_points))
        block = block_pts[chosen]
        orig_idx = np.where(mask)[0][chosen]

        z_ground = np.percentile(block[:, 2], 5)
        height   = (block[:, 2] - z_ground).clip(min=0).astype(np.float32)
        x_norm   = ((block[:, 0] - cx) / half).astype(np.float32)
        y_norm   = ((block[:, 1] - cy) / half).astype(np.float32)

        feat = np.stack([
            x_norm, y_norm, block[:, 2],
            height, block[:, 3],
            block[:, 4], block[:, 5], block[:, 6],
        ], axis=1).astype(np.float32)

        all_feats.append(feat)
        all_orig.append(orig_idx)

        if len(all_feats) >= 16:
            _flush(all_feats, all_orig, model, device, vote_counts)

    if all_feats:
        _flush(all_feats, all_orig, model, device, vote_counts)

    pred = vote_counts.argmax(axis=1).astype(np.int32)
    pred[vote_counts.sum(axis=1) == 0] = -1  # unvisited points
    return pred


def _flush(feats, orig_idxs, model, device, vote_counts):
    feat_t = torch.from_numpy(np.stack(feats)).to(device)
    with torch.no_grad():
        logits = model(feat_t)
        preds  = logits.argmax(dim=-1).cpu().numpy()
    for pred, oi in zip(preds, orig_idxs):
        np.add.at(vote_counts, (oi,), np.eye(NUM_CLASSES, dtype=np.int32)[pred])
    feats.clear()
    orig_idxs.clear()


# ─────────────────────────────────────────────────────────────────────────────
#  UI helpers
# ─────────────────────────────────────────────────────────────────────────────

def legend_html() -> str:
    items = []
    for i, (name, color) in enumerate(zip(CLASS_NAMES, CLASS_COLORS)):
        r, g, b = (color * 255).astype(int)
        items.append(
            f'<span style="background:rgb({r},{g},{b});'
            f'display:inline-block;width:14px;height:14px;'
            f'border-radius:2px;margin-right:5px;vertical-align:middle"></span>'
            f'{i}: {name}'
        )
    cols = [items[:5], items[5:]]
    html = '<div style="display:flex;gap:30px;flex-wrap:wrap;font-size:13px">'
    for col in cols:
        html += '<div>' + '<br>'.join(col) + '</div>'
    html += '</div>'
    return html


# ─────────────────────────────────────────────────────────────────────────────
#  Main app
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # ── Header ───────────────────────────────────────────────────────────────
    st.title("🌆 LiDAR Point Cloud Semantic Segmentation")
    st.markdown(
        "**PointNet++ SSG** trained on [Paris-Lille-3D](http://npm3d.fr/paris-lille-3d) "
        "— 64.42% mIoU on validation set (Paris scan)."
    )
    st.markdown("---")

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Settings")

        model, ckpt = load_model()
        if model is None:
            st.error(f"Checkpoint not found:\n`{CHECKPOINT_PATH}`\nRun training first.")
            st.stop()

        st.success(
            f"Model loaded ✓  \n"
            f"Best mIoU: **{ckpt['miou']*100:.2f}%**  \n"
            f"Epoch: {ckpt['epoch']}"
        )
        device = next(model.parameters()).device
        st.caption(f"Device: `{device}`")

        st.markdown("---")
        st.subheader("Scan selection")

        if not AVAILABLE_SCANS:
            st.error("No preprocessed scans found in `data/processed/`.")
            st.stop()

        scan_name = st.selectbox("Select scan", list(AVAILABLE_SCANS.keys()))

        st.markdown("---")
        st.subheader("Inference settings")
        n_blocks   = st.slider("Blocks to infer (more = slower but denser)", 50, 500, 200, 50)
        show_gt    = st.checkbox("Show ground truth", value=True)
        col_mode   = st.radio("Color by", ["labels", "height", "reflectance"])

        st.markdown("---")
        st.markdown(legend_html(), unsafe_allow_html=True)

    # ── Load scan ─────────────────────────────────────────────────────────────
    with st.spinner(f"Loading {scan_name} ..."):
        points, gt_labels = load_scan(scan_name)

    st.markdown(f"**Scan:** `{scan_name}` — {len(points):,} points after downsampling")

    # ── Run inference ─────────────────────────────────────────────────────────
    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        run = st.button("▶ Run inference", type="primary", use_container_width=True)

    if "pred_labels" not in st.session_state or st.session_state.get("last_scan") != scan_name:
        st.session_state.pred_labels = None

    if run:
        with st.spinner(f"Inferring {n_blocks} blocks on {device} ..."):
            pred = infer_block_sample(
                points, model, device,
                n_blocks=n_blocks, num_points=4096,
            )
            st.session_state.pred_labels = pred
            st.session_state.last_scan   = scan_name

    pred_labels = st.session_state.pred_labels

    # ── Visualizations ────────────────────────────────────────────────────────
    n_cols = 2 if (show_gt and gt_labels is not None) else 1
    cols   = st.columns(n_cols)

    subsample = 300_000

    if show_gt and gt_labels is not None:
        with cols[0]:
            st.subheader("Ground truth")
            pc_gt = PointCloud(xyz=points[:, :3], labels=gt_labels,
                               reflectance=points[:, 3] if col_mode == "reflectance" else None)
            fig = plot_topdown(pc_gt, mode="labels" if col_mode != "reflectance" else "reflectance",
                               subsample=subsample, figsize=(7, 6))
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

    if pred_labels is not None:
        vis_col = cols[1] if (show_gt and gt_labels is not None) else cols[0]
        with vis_col:
            st.subheader("Predictions")
            visible = pred_labels >= 0
            pc_pred = PointCloud(
                xyz=points[visible, :3],
                labels=pred_labels[visible],
                reflectance=points[visible, 3] if col_mode == "reflectance" else None,
            )
            fig = plot_topdown(pc_pred, mode="labels" if col_mode != "reflectance" else "reflectance",
                               subsample=min(subsample, visible.sum()), figsize=(7, 6))
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
    else:
        with cols[-1]:
            st.subheader("Predictions")
            st.info("Click **▶ Run inference** to see predictions.")

    # ── Metrics ───────────────────────────────────────────────────────────────
    if pred_labels is not None and gt_labels is not None:
        st.markdown("---")
        st.subheader("📊 Metrics on sampled blocks")

        from src.training.metrics import MetricTracker
        visible = pred_labels >= 0
        tracker = MetricTracker(NUM_CLASSES, CLASS_NAMES, ignore_index=0)
        tracker.update(pred_labels[visible], gt_labels[visible])
        m = tracker.compute()

        mcol1, mcol2, mcol3 = st.columns(3)
        mcol1.metric("mIoU", f"{m['miou']*100:.2f}%")
        mcol2.metric("Overall Accuracy", f"{m['overall_acc']*100:.2f}%")
        mcol3.metric("Points inferred", f"{visible.sum():,}")

        # Per-class table
        rows = []
        for name in CLASS_NAMES[1:]:  # skip unclassified
            iou = m["per_class_iou"].get(name, float("nan"))
            acc = m["per_class_acc"].get(name, float("nan"))
            rows.append({
                "Class": name,
                "IoU (%)": f"{iou*100:.1f}" if not np.isnan(iou) else "—",
                "Accuracy (%)": f"{acc*100:.1f}" if not np.isnan(acc) else "—",
            })
        st.dataframe(rows, use_container_width=True, hide_index=True)

        # Distribution chart
        st.subheader("Predicted class distribution")
        visible_mask = pred_labels >= 0
        pc_dist = PointCloud(xyz=points[visible_mask, :3], labels=pred_labels[visible_mask])
        fig = plot_class_distribution(pc_dist, figsize=(10, 4))
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        # Download predictions
        st.markdown("---")
        st.subheader("💾 Export")
        pred_csv = "\n".join([
            "point_idx,pred_class,pred_name",
            *[f"{i},{p},{CLASS_NAMES[p] if p >= 0 else 'unvisited'}"
              for i, p in enumerate(pred_labels)]
        ])
        st.download_button(
            "Download predictions (CSV)",
            data=pred_csv,
            file_name=f"{scan_name}_predictions.csv",
            mime="text/csv",
        )

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.caption(
        "Damien Blanc — PhD in AI for 3D Imaging | "
        "[GitHub](https://github.com/your-username/lidar-pointcloud-semantic-segmentation)"
    )


if __name__ == "__main__":
    main()

# LiDAR Point Cloud Semantic Segmentation

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.6-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Open3D](https://img.shields.io/badge/Open3D-0.19-4285F4?style=flat-square)
![mIoU](https://img.shields.io/badge/mIoU-64.42%25-brightgreen?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

**Semantic segmentation of outdoor urban LiDAR point clouds using deep learning.**  
Dataset: [Paris-Lille-3D](http://npm3d.fr/paris-lille-3d) · Model: PointNet++ SSG · **mIoU: 64.42%**

</div>

---

## Context

This project applies deep learning–based semantic segmentation to large-scale outdoor LiDAR point clouds, using the **Paris-Lille-3D** benchmark dataset. It demonstrates the transfer of 3D segmentation skills from volumetric medical imaging (CT, organoids) to the outdoor LiDAR domain.

The core challenge is the same in both domains: **learning meaningful geometric representations from unstructured 3D data**. What changes is the scale (meters vs. millimeters), the sensor (LiDAR vs. CT scanner), and the absence of a regular grid.

Key challenges specific to outdoor LiDAR:
- **Scale heterogeneity**: objects range from small bollards (cm) to building facades (hundreds of meters)
- **Class imbalance**: ground and buildings dominate (~76%); pedestrians represent 0.1% of points
- **Sensor noise**: variable point density depending on scan distance and surface reflectance
- **No grid structure**: unlike voxels or images, raw point clouds are orderless and irregular

---

## Results

Trained for **34 epochs** on a single RTX 3090 (~9 hours). Early stopping triggered at epoch 34 (best at epoch 19).

| Model | mIoU | OA | Epochs | GPU |
|-------|------|-----|--------|-----|
| PointNet++ SSG (ours) | **64.42%** | **93.16%** | 34 | RTX 3090 |
| PointNet++ (published) | ~63% | — | — | — |
| KPConv (SOTA) | ~76% | — | — | — |

### Per-class IoU

| Class | IoU | Accuracy | Notes |
|-------|-----|----------|-------|
| ground | 97.0% | 98.0% | Dominant class, near-perfect |
| building | 87.4% | 88.3% | Vertical planar surfaces well captured |
| vegetation | 87.6% | 97.8% | Distinctive normal distribution |
| car | 94.5% | 95.6% | Compact regular shape, easiest to learn |
| barrier | 56.0% | 73.3% | Confused with ground at edges |
| pedestrian | 54.6% | 72.2% | Only 0.1% of points — weighted loss critical |
| bollard | 37.9% | 64.6% | 0.0% of points — remarkable given scarcity |
| pole/sign | 32.6% | 89.1% | Thin elongated objects, few points per instance |
| trash can | 17.8% | 31.3% | Rarest class, hardest to segment |

---

## Project Structure

```
lidar-semantic-segmentation/
├── data/
│   ├── raw/                        # Original Paris-Lille-3D .ply files
│   └── processed/                  # Preprocessed .npy arrays (post voxel downsampling)
│       ├── Lille1_1/{points,labels,stats}.npy
│       ├── Lille1_2/
│       ├── Lille2/
│       └── Paris/
├── src/
│   ├── data/
│   │   ├── loader.py               # PLY / LAS loader with class remapping
│   │   └── dataset.py              # PyTorch Dataset — block cropping on-the-fly
│   ├── preprocessing/
│   │   └── pipeline.py             # Voxel downsampling, normal estimation, features
│   ├── models/
│   │   └── pointnet2.py            # PointNet++ SSG — pure PyTorch, no dependencies
│   ├── training/
│   │   ├── trainer.py              # Training loop, weighted CE loss, early stopping
│   │   └── metrics.py              # mIoU, per-class IoU, confusion matrix
│   └── visualization/
│       └── visualizer.py           # Open3D + Matplotlib visualizations
├── scripts/
│   ├── download_data.py            # Dataset download helper
│   ├── preprocess.py               # Run full preprocessing pipeline
│   ├── train.py                    # Launch training
│   └── inference.py                # Sliding-window inference + figure export
├── app/
│   └── demo.py                     # Streamlit interactive demo
├── configs/
│   └── default.yaml
├── outputs/
│   ├── checkpoints/
│   └── figures/
└── main.py                         # Quick entry point (load + visualize)
```

---

## Methodology

### 1. Data Loading
The Paris-Lille-3D `training_10_classes` split provides 4 annotated PLY files
with fields `x, y, z, reflectance (uchar), class (int)`.

10 coarse semantic classes:

| ID | Class | Color |
|----|-------|-------|
| 0 | Unclassified | grey |
| 1 | Ground | sandy brown |
| 2 | Building | light grey |
| 3 | Pole / Road sign | gold |
| 4 | Bollard | orange |
| 5 | Trash can | chocolate |
| 6 | Barrier | saddle brown |
| 7 | Pedestrian | orange-red |
| 8 | Car | royal blue |
| 9 | Vegetation | forest green |

### 2. Preprocessing Pipeline

```
Raw PLY (30M pts / scan)
    ↓  Voxel downsampling (0.05 m)      →  ~12M pts — uniform density
    ↓  Normal estimation (r=0.3 m)      →  local surface orientation
    ↓  Feature assembly                 →  [x, y, z, reflectance, nx, ny, nz]
    ↓  Save as .npy                     →  data/processed/<scan>/
```

Block cropping (4 m × 4 m spatial blocks, 4096 pts each) is done on-the-fly
in the Dataset class, following the same pattern as sliding-window inference
on CT volumes.

Final feature vector per point — 8 dimensions:

| Index | Feature | Description |
|-------|---------|-------------|
| 0 | x_norm | X normalized within block [-1, 1] |
| 1 | y_norm | Y normalized within block [-1, 1] |
| 2 | z | Absolute Z (Lambert-93) |
| 3 | height | Z - Z_block_5th_percentile (height above local ground) |
| 4 | reflectance | Laser return intensity [0, 1] |
| 5-7 | nx, ny, nz | Surface normal unit vector |

### 3. Model: PointNet++ SSG

Encoder–decoder architecture with skip connections, analogous to a 3D U-Net on point clouds:

```
Input: (B, 4096, 8)
│
├── SA1: 4096→1024 pts  r=0.20  k=32  →  64  features   (fine details: poles, edges)
├── SA2: 1024→256  pts  r=0.40  k=32  →  128 features   (medium objects: cars, pedestrians)
├── SA3:  256→64   pts  r=0.80  k=32  →  256 features   (large structures: buildings)
└── SA4:   64→1    pt   global         →  512 features   (global context)
│
├── FP3:   1→ 64  pts   [512+256 → 256]
├── FP2:  64→256  pts   [256+128 → 128]
├── FP1: 256→1024 pts   [128+64  → 128]
└── FP0: 1024→4096 pts  [128+8   → 128]
│
Head: Conv1d(128→128) → Dropout(0.5) → Conv1d(128→10)

Parameters: 972,714
```

### 4. Training

- **Loss**: Weighted cross-entropy — weights inversely proportional to class frequency
  (bollard weight = 4.56×, ground weight = 0.004×)
- **Optimizer**: Adam, lr=0.001, weight decay=1e-4
- **LR schedule**: Cosine annealing
- **Augmentation**: Z-axis rotation, XYZ jitter (σ=0.01 m), random reflectance dropout
- **Early stopping**: patience=15 epochs on val mIoU

### 5. Inference

Sliding-window: 4 m × 4 m blocks with 2 m stride over the full scan.
Each point is predicted independently in each block it belongs to.
Final label = majority vote across all blocks.

---

## Connection to Volumetric Segmentation

| CT / Microscopy domain | LiDAR domain |
|------------------------|--------------|
| 3D voxel grid | Irregular point cloud |
| Hounsfield units | Reflectance + height above ground |
| 3D U-Net | PointNet++ (FPS + ball query instead of strided conv) |
| Sliding window inference | Sliding block inference (4m × 4m) |
| Dice + weighted CE | Weighted CE (inverse class frequency) |
| Isotropic resampling | Voxel downsampling (0.05 m) |
| Gradient-based features | Surface normals (PCA on KNN neighborhood) |

---

## Getting Started

### 1. Clone and install

```bash
git clone https://github.com/your-username/lidar-pointcloud-semantic-segmentation.git
cd lidar-pointcloud-semantic-segmentation
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

### 2. Download the dataset

Register at [http://npm3d.fr/paris-lille-3d](http://npm3d.fr/paris-lille-3d).  
Place the `training_10_classes/` folder at the path in `configs/default.yaml`.

```bash
python scripts/download_data.py --check "D:/your/path/training_10_classes/"
```

### 3. Preprocess

```bash
python scripts/preprocess.py          # ~5 min per scan on modern CPU
```

### 4. Visualize raw data

```bash
python main.py --file "D:/your/path/training_10_classes/Lille1_1.ply" --mode labels
```

### 5. Train

```bash
python scripts/train.py --batch_size 16 --num_workers 4
# Monitor: tensorboard --logdir outputs/logs
```

### 6. Run inference and export figures

```bash
python scripts/inference.py --scan Paris --save_ply
```

### 7. Interactive demo

```bash
streamlit run app/demo.py
```

---

## Technologies

| Category | Tools |
|----------|-------|
| Point cloud I/O | `plyfile`, `open3d`, `laspy` |
| Deep learning | `torch 2.6` (CUDA 12.4) |
| Training monitoring | `tensorboard` |
| Visualization | `open3d`, `matplotlib`, `plotly` |
| Demo | `streamlit` |

---

## License

MIT — see [LICENSE](LICENSE).

---

## Author

**Damien Blanc** — PhD in AI for 3D Imaging  
Specialization: volumetric segmentation (CT, microscopy) → transferring to outdoor LiDAR  
[LinkedIn](https://linkedin.com/in/your-profile) · [GitHub](https://github.com/your-username)

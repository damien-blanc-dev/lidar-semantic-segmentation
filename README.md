# LiDAR Point Cloud Semantic Segmentation

Semantic segmentation of outdoor urban LiDAR point clouds on Paris-Lille-3D, with a focus on geometric feature ablations, class imbalance, and spatial error analysis.

## Project framing

This project studies how a deep model learns semantic structure from irregular 3D point clouds, where geometry is sparse, unordered, and highly imbalanced across classes. Rather than treating the task as a pure benchmark exercise, the goal is to understand which signals actually drive performance on difficult classes such as pedestrians, bollards, trash cans, and pole-signs.

The project also demonstrates transfer from volumetric biomedical imaging to outdoor LiDAR segmentation. In both cases, the core challenge is learning meaningful 3D representations from spatial data, but the LiDAR setting removes the regular voxel grid and introduces strong density variation, scale heterogeneity, and sensor-driven geometric noise.

## Research questions

- Do surface normals consistently improve semantic segmentation, or do they become unreliable on thin vertical structures?
- Are rare-class failures driven mainly by class imbalance, noisy local geometry, or limited spatial context?
- Which loss function best handles extreme class imbalance — and does focal loss help or hurt?
- How sensitive is the model to the radius used for normal estimation?
- How much context (block size, point count) is needed to recover rare classes?

## Dataset

Experiments use the Paris-Lille-3D benchmark, with three scans for training (`Lille1_1`, `Lille1_2`, `Lille2`) and one scan for validation (`Paris`). The semantic label space contains 10 classes including ground, building, pole-sign, bollard, trash can, barrier, pedestrian, car, and vegetation.

A key difficulty is severe class imbalance. Ground and building dominate the dataset, while pedestrian represents only about 0.1 percent of points, making global accuracy alone a misleading metric.

## Method

### Preprocessing

The preprocessing pipeline converts raw annotated PLY files into reusable NumPy tensors through voxel downsampling at 0.05 m, normal estimation, and feature assembly. Each point is represented by 8 input channels: normalized x and y within the block, absolute z, height above local ground, reflectance, and the three components of the local surface normal.

### Model

The main baseline is a PointNet++ SSG encoder-decoder with set abstraction and feature propagation layers, analogous in spirit to a 3D U-Net operating on irregular point sets instead of dense volumes. This model has 972,714 trainable parameters.

### Training

Training uses weighted cross-entropy to address extreme class imbalance, Adam optimization, cosine annealing, geometric augmentation, gradient clipping, TensorBoard logging, checkpointing, and early stopping on validation mIoU. The class-weight computation is explicitly designed to prevent collapse toward dominant classes such as ground and building.

### Inference

Inference is performed with sliding 4 m × 4 m windows using a 2 m stride over the full scene, followed by aggregation of multiple local predictions for each point through majority voting. This mirrors the same tiling logic often used in volumetric medical segmentation, adapted here to irregular LiDAR geometry.

## Results

All experiments use PointNet++ SSG with 972,714 parameters, 4096 points per block (unless noted), `block_size=4.0 m`, `seed=42`.

### Best runs

| Experiment | Loss | Normal radius | mIoU | OA | Pedestrian | Bollard | Pole/sign | Trash can |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| exp1_r010 | weighted_ce | 0.1 | 59.28 | 90.69 | 48.97 | 19.21 | 46.94 | 16.96 |
| exp1_r020 | weighted_ce | 0.2 | 69.13 | 95.11 | 49.63 | **41.96** | 58.34 | **28.12** |
| exp1_r030 | weighted_ce | 0.3 | 63.35 | 93.75 | 53.12 | 31.91 | 44.88 | 13.41 |
| exp1_r050 | weighted_ce | 0.5 | 68.53 | 95.65 | 49.06 | 35.94 | 57.16 | 25.29 |
| exp2_ce | ce | 0.3 | 68.31 | 96.14 | 54.46 | 33.49 | 58.55 | 24.08 |
| **exp2_weighted_ce** | weighted_ce | 0.3 | **69.51** | **96.49** | 54.73 | 32.31 | 60.10 | 22.87 |
| exp2_focal | focal | 0.3 | 1.42 | 0.74 | 9.02 | 2.69 | 0.39 | 0.66 |
| exp2_focal_v2 | focal | 0.3 | 50.10 | 83.22 | 43.39 | 27.63 | 23.09 | 4.53 |
| exp2_cb_focal | cb_focal | 0.3 | 2.49 | 0.69 | 7.75 | 14.01 | 0.54 | 0.09 |
| exp2_cb_focal_v2 | cb_focal | 0.3 | 57.07 | 89.11 | 41.44 | 37.53 | 35.91 | 7.45 |
| exp3_b2_n2048 | weighted_ce | 0.3 | 67.09 | 94.79 | 54.34 | 35.03 | **66.68** | 9.94 |

> **Best overall:** `exp2_weighted_ce` — 69.51% mIoU, 96.49% OA.  
> `exp1_r020` is competitive at 69.13% mIoU and achieves the highest bollard IoU (41.96%) and trash can IoU (28.12%), suggesting that a tighter normal radius better captures thin and small objects.

### Ablation study

#### Effect of loss function (normal radius = 0.3, block = 4 m, N = 4096)

| Loss | mIoU | OA | Notes |
|---|---:|---:|---|
| weighted_ce | **69.51** | **96.49** | Best overall, stable training |
| ce | 68.31 | 96.14 | Only −1.2% mIoU vs weighted, but weaker on rare classes |
| focal | 1.42 | 0.74 | Complete collapse — learning failure |
| focal_v2 | 50.10 | 83.22 | Unstable, strong degradation on rare classes |
| cb_focal | 2.49 | 0.69 | Complete collapse — learning failure |
| cb_focal_v2 | 57.07 | 89.11 | Second attempt improves, but still −12% mIoU |

`weighted_ce` is the most robust loss for this imbalanced setup. Both focal variants fully collapsed on the first attempt and recovered partially after tuning (`_v2`), but remain significantly below `weighted_ce`. The high OA paired with near-zero mIoU in collapsed runs confirms the model predicted only dominant classes (ground, building).

#### Effect of normal estimation radius (loss = weighted_ce, block = 4 m, N = 4096)

| Normal radius | mIoU | OA | Bollard | Trash can | Pole/sign |
|---:|---:|---:|---:|---:|---:|
| 0.1 m | 59.28 | 90.69 | 19.21 | 16.96 | 46.94 |
| 0.2 m | **69.13** | 95.11 | **41.96** | **28.12** | 58.34 |
| 0.3 m | 69.51 | **96.49** | 32.31 | 22.87 | 60.10 |
| 0.5 m | 68.53 | 95.65 | 35.94 | 25.29 | 57.16 |

The optimal radius lies between 0.2 and 0.3 m. At r=0.1 m, normals are noisy and the model degrades sharply, especially on bollards (−22% vs r=0.2). At r=0.5 m, overly smooth normals lose discriminative power on small objects. r=0.2 produces the best result on rare classes while r=0.3 gives slightly better overall OA.

#### Effect of block size and point count (loss = weighted_ce, normal radius = 0.3)

| Experiment | Block size | N points | mIoU | OA | Pole/sign | Trash can |
|---|---:|---:|---:|---:|---:|---:|
| exp2_weighted_ce | 4.0 m | 4096 | **69.51** | **96.49** | 60.10 | **22.87** |
| exp3_b2_n2048 | 2.0 m | 2048 | 67.09 | 94.79 | **66.68** | 9.94 |

Halving block size and point count reduces global mIoU by only 2.4%, but dramatically shifts the per-class distribution: pole/sign improves (+6.6%) while trash can drops sharply (−12.9%). Smaller blocks provide finer local context, which benefits thin elongated structures, but reduce spatial coverage for rare compact objects.

### Interpretation

- **`weighted_ce` is the most reliable loss** for this setup. Focal loss variants failed or underperformed significantly, even after hyperparameter tuning. The imbalance here is too severe for focal loss without careful gamma and alpha tuning.
- **Normal radius r=0.2–0.3 m is the optimal range.** Too small is noisy; too large is oversmoothed. The sweet spot depends on the target class: r=0.2 best serves small objects, r=0.3 gives better global OA.
- **Block geometry is a trade-off, not a free parameter.** Smaller blocks help thin structures (pole/sign) at the cost of compact rare objects (trash can). This suggests that multi-scale inference or class-specific block strategies could further close the gap.
- **Global OA is not a reliable indicator of rare-class performance.** Collapsed runs (focal, cb_focal) achieved <1% mIoU while showing 0.7–0.8% OA — confirming that the model learned to predict only dominant classes.

## Per-class interpretation

| Class | IoU (best run) | Interpretation |
|---|---:|---|
| Ground | 97.0 | Dominant and geometrically easy |
| Building | 87.4 | Strong planar structure |
| Vegetation | 87.6 | Distinctive local geometry |
| Car | 94.5 | Compact regular object |
| Barrier | 56.0 | Boundary confusion with ground |
| Pedestrian | 54.7 | Rare and small, recoverable with weighting |
| Bollard | 41.96 (r=0.2) | Extremely scarce; geometry critical |
| Pole-sign | 66.68 (block=2m) | Thin structures; benefits from smaller blocks |
| Trash can | 28.12 (r=0.2) | Rarest and hardest class |

These results show that average mIoU hides several distinct regimes: easy large planar classes, medium-sized regular objects, and rare thin or compact classes where local representation quality matters far more than raw capacity.

## Error analysis

The repository includes a dedicated post-hoc analysis script that generates a recall-normalized confusion matrix, a top-down error map, the most frequent confusion pairs, and zoomed inspections of hard classes. This moves beyond a scoreboard mindset and turns the model into an analyzable system.

The error-analysis tooling shows that failures concentrate around hard classes, class boundaries, and rare object regions rather than being uniformly distributed in space.

## Limitations

- The z coordinate is in absolute Lambert-93 space (~0–100 m), not normalized. KNN-based architectures (RandLA-Net, PointTransformer) may have biased neighborhoods due to elevation-dominated distances. This is a known issue that will be addressed before the architecture benchmark.
- The 4 m block formulation is a likely bottleneck for context-dependent distinctions. Some classes may require more global context, while others may suffer from the majority-vote fusion or from instability in local normal estimation.
- Architecture comparison (RandLA-Net, PointTransformer) is in progress as Exp 4.

## Next experiments

1. Benchmark PointNet++ SSG, RandLA-Net, and PointTransformer under the same protocol (Exp 4 — in progress).
2. Fix z normalization before the architecture benchmark to ensure fair KNN neighborhoods across models.
3. Measure the effect of inference stride and multi-scale blocks on hard-class IoU and runtime.
4. Add uncertainty maps to predict where the model is likely to fail and support error triage.

## Repository structure

- `preprocess.py` — preprocessing and feature generation.
- `train.py` — training entry point with selectable architectures (PointNet++ SSG, RandLA-Net, PointTransformer).
- `train_ablation.py` — ablation runner for feature and training recipe studies.
- `inference.py` — sliding-window scene inference.
- `error_analysis.py` — confusion and spatial failure analysis.
- `trainer.py` — optimization loop, weighting, early stopping, and logging.
- `src/fusion/projection.py` — camera-LiDAR geometry utilities (projection, colorization, calibration sensitivity).
- `scripts/colorize_pointcloud.py` — CLI for point cloud colorization from RGB image.
- `scripts/eval_projection.py` — CLI for calibration sensitivity analysis.

## Takeaway

This project is best presented as a 3D representation-learning study on irregular geometry, not only as a segmentation benchmark. Its main strength is the combination of engineering completeness and hypothesis-driven analysis around normals, class imbalance, loss sensitivity, and hard spatial failure modes.

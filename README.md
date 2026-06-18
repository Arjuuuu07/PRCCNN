# PRCCN: Projection Row-Column Convolutional Network

> **Status: Active experiment — results are preliminary and the architecture is still evolving.**

---

## Overview

PRCCN is a lightweight image classification network built around a projection-based feature extraction idea. Instead of applying convolution directly over 2D pixel grids, PRCCN first converts the image into 1D row and column signals using full-span convolutions, then applies further convolution over those projected signals.

Standard CNNs also extract higher-level patterns — shape, texture, edges — but do so through local 2D patch extraction. PRCCN asks whether convolving over global 1D row and column projections can capture the same higher-level information from a different perspective. Importantly, this idea is not restricted to 2D images; any structured input with meaningful row and column axes can be treated the same way.

This is a hypothesis under active investigation. The experiments here are early steps in validating it.

---

## Architecture

### Projection stage

**RowExtractor** — `Conv2d` with kernel `(1 × W)`: sweeps the full width of each row, producing a 1D signal that globally summarizes each row.

**ColExtractor** — `Conv2d` with kernel `(H × 1)`: sweeps the full height of each column, producing a 1D signal that globally summarizes each column.

The row and column signals are stacked into a `(B, C, 2, W)` tensor — a compact dual-axis structural signature of the image — and passed into a standard conv-BN-ReLU processing stack followed by `AdaptiveAvgPool` and a fully connected head.

### Two configurations tested

The configurations differ in the technique applied to process the projected signals after extraction:

| Config | Downsampling | Projection channels | FC head | Params |
|---|---|---|---|---|
| v1 | MaxPool | 8 | [64, 32] | 71,050 |
| v2 | Stride-based | 16 | [128, 64] | 80,586 |

---

## Baselines

Two standard CNN baselines, one per dataset. Both use the same training setup as PRCCN.

**BaselineCNN** (Fashion-MNIST comparison)
4 conv blocks (16→32→64→128, 3×3 kernels, MaxPool), `AdaptiveAvgPool(4,4)`, FC [64, 32] — **231,178 params**

**ComparableCNN** (KMNIST comparison)
4 conv blocks (32→64→128→64, stride 2), `AdaptiveAvgPool(1,1)`, FC [128, 64] — **184,266 params**

---

## Experiments & Results

Two separate experiments on two datasets. Comparisons are not fully controlled across datasets — different baseline architectures and fold counts were used. This is acknowledged as a limitation.

---

### Experiment 1 — Fashion-MNIST

PRCCN v1 config (MaxPool downsampling, 8 projection channels) vs BaselineCNN.

| Model | Params | CV Folds | Mean AUC | Std AUC | Mean Val Acc | Std Acc |
|---|---|---|---|---|---|---|
| PRCCN | 71,050 | 5 | 0.9939 | 0.0003 | 90.40% | 0.36% |
| BaselineCNN | 231,178 | 3 | 0.9964 | 0.0002 | 93.06% | 0.20% |

**Observation:** BaselineCNN outperforms PRCCN on Fashion-MNIST by ~0.0025 AUC and ~2.7% accuracy, while using 3.3× more parameters. The comparison is not fully controlled — different fold counts (5 vs 3) and different FC head sizes. The v1 config with MaxPool downsampling and narrower projection channels may be under-capacity for this dataset. Running the v2 config on Fashion-MNIST is the most important missing experiment.

---

### Experiment 2 — KMNIST

PRCCN v2 config (stride-based downsampling, 16 projection channels) vs ComparableCNN.

| Model | Params | CV Folds | Mean AUC | Std AUC | Mean Val Acc | Std Acc |
|---|---|---|---|---|---|---|
| PRCCN | 80,586 | 5 | 0.9991 | 0.0001 | 96.62% | 0.15% |
| ComparableCNN | 184,266 | 5 | 0.9987 | 0.0001 | 96.17% | 0.31% |

**Observation:** PRCCN exceeds the ComparableCNN on KMNIST on both AUC (+0.0004) and accuracy (+0.45%) while using 2.3× fewer parameters. PRCCN also shows tighter variance across folds on both metrics (Std Acc 0.15% vs 0.31%), indicating more stable training. The ComparableCNN shows volatile early epochs — AUC as low as 0.9385 in fold 5 epoch 1 — while PRCCN converges cleanly from epoch 1 across all folds.

---

## What the Results Tell Us So Far

**Suggests:**
- The projection approach is competitive with and in some settings outperforms standard CNNs at significantly lower parameter counts
- The processing technique applied after projection matters considerably — stride-based downsampling with wider projection channels (v2) substantially improves over MaxPool with narrower channels (v1)
- PRCCN trains more stably with lower variance across folds and epochs than the CNN baselines

**Still open:**
- Fashion-MNIST comparison is not fully controlled — v2 config vs BaselineCNN on Fashion-MNIST is pending
- Whether the 1D projections genuinely encode higher-level structure or whether this is an architectural efficiency coincidence
- Behavior on RGB datasets or inputs with more complex spatial structure

---

## Training Setup

All experiments use the same protocol:

- Optimizer: Adam, lr = 1e-3
- Scheduler: ReduceLROnPlateau (patience=3, factor=0.5)
- Loss: CrossEntropyLoss
- Dropout: 0.1
- Image size: 128×128, grayscale
- Normalization: per-split mean/std computed on training data only (no leakage)
- Metrics: macro-averaged AUROC and accuracy
- CV: stratified k-fold, seed 42

---




---

## Next Steps

- Run PRCCN v2 config on Fashion-MNIST with matched 5-fold CV and BaselineCNN comparison
- Run BaselineCNN on KMNIST for a complete cross-dataset picture
- Activation visualization — what do RowExtractor and ColExtractor actually learn?
- Extend to RGB inputs (CIFAR-10)
- Explore attention over row/col projections as a richer abstraction mechanism

---

## Datasets

- **Fashion-MNIST** — Xiao et al., 2017. [GitHub](https://github.com/zalandoresearch/fashion-mnist)
- **KMNIST** — Clanuwat et al., 2018. [GitHub](https://github.com/rois-codh/kmnist)

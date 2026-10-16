# 🧠 Optimised AI for Low-Cost Brain Tumor MRI Classification

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.11%2B-orange?logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-green)
![Dataset](https://img.shields.io/badge/Dataset-Kaggle-20BEFF?logo=kaggle)
![Models](https://img.shields.io/badge/Models-CNN%20%7C%20MLP%20%7C%20EfficientNet%20%7C%20ResNet18%20%7C%20ResNet50-red)

A comparative study of five deep learning architectures for multiclass brain tumor classification from MRI scans. Models are benchmarked not only on **predictive performance** (accuracy, F1-score) but also on **computational efficiency** (training time, inference time, parameter count) — directly addressing the low-cost deployment constraint.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Models](#models)
- [Preprocessing & Augmentation](#preprocessing--augmentation)
- [Results](#results)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Limitations](#limitations)
- [References](#references)

---

## Overview

Brain tumor detection and classification from MRI is a high-stakes clinical task where both **accuracy** and **speed** matter. This project evaluates five architectures across four tumor categories to identify which model best balances diagnostic performance with computational cost — a critical consideration for deployment in low-resource clinical environments.

The pipeline covers:

- Class-balanced data loading with `WeightedRandomSampler`
- Data augmentation (horizontal flip, rotation, color jitter)
- Training with per-epoch loss tracking across all 5 models
- Comprehensive evaluation: Accuracy, Precision, Recall, F1-score, Confusion Matrix
- Training and inference time measurement per model
- Parameter count comparison as a proxy for deployment cost
- Single-image inference for each model
- Comparative loss curve visualisation across all algorithms

---

## Dataset

**Brain Tumor MRI Dataset** — Masoud Nickparvar (Kaggle)

| Property | Value |
|---|---|
| Source | [Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) |
| Classes | Glioma, Meningioma, No Tumor, Pituitary |
| Format | JPEG images, organised by class folder |
| Split | Pre-divided into `Training/` and `Testing/` |

> **Note:** The dataset is not included in this repository. See [Usage](#usage) for setup instructions.

---

## Models

Five architectures are trained and evaluated on the same dataset under identical conditions:

| Model | Type | Pretrained | Parameters |
|---|---|---|---|
| **CNN** | Custom 2-conv + 3-FC | ✗ | ~90K |
| **MLP** | 4-layer fully connected | ✗ | ~3.8M |
| **EfficientNet-B0** | Compound scaling CNN | ✓ ImageNet | ~5.3M |
| **ResNet18** | Residual network | ✓ ImageNet | ~11.2M |
| **ResNet50** | Deep residual network | ✓ ImageNet | ~25.6M |

### Architecture Highlights

**CNN** — Lightweight custom network. Two convolutional layers (3→12→24 filters, 5×5 kernel) with MaxPooling, followed by three fully connected layers (120 → 84 → 4). Designed for fast training on limited hardware.

**MLP** — Fully connected baseline. Input images are flattened to 3072 features (3×32×32), passed through three hidden layers (1024 → 512 → 256 → 4) with ReLU activations.

**EfficientNet-B0** — Pretrained compound-scaled architecture. Final classification layer replaced with a 4-class linear head. Adam optimiser.

**ResNet18 / ResNet50** — Pretrained residual networks. Final fully connected layer replaced with a 4-class head. SGD with momentum.

---

## Preprocessing & Augmentation

All images undergo the following transforms at training time:

| Step | Detail |
|---|---|
| Resize | 32 × 32 px |
| Random horizontal flip | p = 0.5 |
| Random rotation | ±15° |
| Color jitter | brightness, contrast, saturation ±0.2; hue ±0.1 |
| Normalisation | mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5) |

**Class rebalancing:** `WeightedRandomSampler` assigns inverse-frequency weights per class, compensating for imbalance during training without discarding any samples.

---

## Results

> Results reflect each model trained for 40 epochs (MLP: 50 epochs), batch size 12, on the Kaggle Brain Tumor MRI dataset.

### Performance Metrics

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| CNN | — | — | — | — |
| MLP | — | — | — | — |
| EfficientNet-B0 | — | — | — | — |
| ResNet18 | — | — | — | — |
| ResNet50 | — | — | — | — |

### Efficiency Metrics

| Model | Train Time (s) | Inference Time (s) | Parameters |
|---|---|---|---|
| CNN | — | — | ~90K |
| MLP | — | — | ~3.8M |
| EfficientNet-B0 | — | — | ~5.3M |
| ResNet18 | — | — | ~11.2M |
| ResNet50 | — | — | ~25.6M |

> Fill these tables after running the notebook. Timing values are printed at the end of each model's training and evaluation blocks.

---

## Project Structure

```
brain-tumor-classification/
├── brain_tumor_classification.py    # Full pipeline: all 5 models
├── requirements.txt                 # Python dependencies
├── .gitignore                       # Excludes data, checkpoints, cache
└── README.md
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/iMalcolmS/brain-tumor-classification.git
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the dataset

Download from [Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) and upload to Google Drive:

```
MyDrive/Brain Tumor/
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
└── Testing/
    ├── glioma/
    ├── meningioma/
    ├── notumor/
    └── pituitary/
```

---

## Usage

### Running on Google Colab

1. Upload `brain_tumor_classification.py` to your Colab session
2. Mount Google Drive when prompted
3. Verify the dataset paths:
   ```python
   train_path = '/content/drive/MyDrive/Brain Tumor/Training'
   test_path  = '/content/drive/MyDrive/Brain Tumor/Testing'
   ```
4. Run the script — each model trains, evaluates, and saves results sequentially

### Outputs

| File | Description |
|---|---|
| `CNN_trained_model.pth` | Saved CNN weights |
| `MLP_trained_model.pth` | Saved MLP weights |
| `EfficientNet_trained_model.pth` | Saved EfficientNet weights |
| `ResNet18_trained_model.pth` | Saved ResNet18 weights |
| `ResNet50_trained_model.pth` | Saved ResNet50 weights |
| `training_loss_<model>.png` | Per-model loss curve |
| `comparison_training_loss.png` | Combined loss comparison plot |

### Single-Image Inference

Each model includes a single-image prediction block. Replace the path with any MRI image:

```python
image_path = '/content/drive/MyDrive/your_mri_image.jpg'
```

The predicted class (Glioma / Meningioma / No Tumor / Pituitary) is printed alongside the image.

---

## Limitations

- **Input resolution:** All models use 32×32 images. Pretrained models (EfficientNet, ResNet) were designed for 224×224 — using a smaller resolution limits the feature extraction capacity of these architectures.
- **Google Colab dependency:** Paths are hard-coded to `/content/drive/MyDrive/`. Adapt for local execution.
- **No cross-validation:** A single train/test split is used. K-fold cross-validation would yield more robust metric estimates.
- **Training variable scoping:** Some training loops share variable names across models. Ensure cells are executed in order to avoid stale variable references.

---

## References

- Nickparvar, M. (2021). *Brain Tumor MRI Dataset.* Kaggle.
- Tan, M. & Le, Q. (2019). *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.* ICML.
- He, K. et al. (2016). *Deep Residual Learning for Image Recognition.* CVPR.
- [efficientnet-pytorch](https://github.com/lukemelas/EfficientNet-PyTorch) — Luke Melas-Kyriazi.

---

## Author

**Malcolm** — MSc Artificial Intelligence (Healthcare), University of West London  
[LinkedIn](https://www.linkedin.com/in/malcolm-sanha-138918269/) · [GitHub](https://github.com/iMalcolmS)

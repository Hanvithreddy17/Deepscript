# DeepScript: Ancient Indian Script Identification System

DeepScript is a deep-learning-based image classification system designed to identify selected ancient Indian scripts from visual inscription images.

> **Important Scope Clarification:** DeepScript focuses strictly on **Script Recognition** (classifying which script class/symbol family appears in an image). It is **NOT** an Optical Character Recognition (OCR), text transcription, transliteration, or semantic translation tool.

---

## 1. Project Overview

Ancient Indian inscriptions engraved on stone, copper plates, pottery, and palm leaves represent a vital cultural and historical heritage. Identifying these historical scripts manually requires specialized paleographic and epigraphic expertise. DeepScript automates the visual identification of ancient Indian script classes using modern vision architectures, transfer learning, and metric feature embeddings.

### Problem Statement
- **Substrate Degradation:** Severe physical weathering, surface erosion, and irregular stone textures create substantial visual noise.
- **Visual Similarity:** Close evolutionary relationships between regional Indian scripts lead to subtle inter-class visual boundaries.
- **Epigraphic Data Constraints:** Historical inscriptions exhibit inherent class imbalances and scarce high-quality labeled samples.

### System Solution
DeepScript combines a pretrained Vision Transformer (ViT-B/16) backbone, a normalized metric projection head, and a two-stage transfer learning strategy to achieve robust representation extraction from inscription imagery.

---

## 2. System Architecture & ML Pipeline

```
[ Inscription Image (RGBA / Grayscale / RGB) ]
                      │
                      ▼
[ Image Preprocessing: Resize 224×224 (Bicubic) + ImageNet Normalization ]
                      │
                      ▼
[ Pretrained Vision Transformer (ViT-B/16) Backbone ]
                      │  (768-dimensional features)
                      ▼
[ Metric Embedding Projector (MLP: 768 → 512 → 256 + L2-Normalization) ]
                      │  (256-dimensional unit-norm embedding, ||z||₂ = 1.0)
                      ▼
[ Classification Head (Cosine Similarity Head / Linear Head / Prototypical Head) ]
                      │
                      ▼
[ Script Prediction (Top-1 Class, Top-k Candidates & Confidence Scores) ]
```

---

## 3. Dataset & 62-Class Organization

- **Total Valid Images:** 6,792 inscription character samples.
- **Script Classes:** 62 distinct script symbol/character classes (including vowels, consonants, conjuncts, and numerals such as `a`, `aa`, `ka`, `kha`, `ga`, `ta`, `da`, `la`, `ya`, `zero`, etc.).
- **Dynamic Discovery & Sorting:** Automated class directory scanning with deterministic alphabetical label indexing ($0 \le y \le 61$).
- **File Integrity:** Zero corrupted or unreadable image files detected across the entire 6,792 sample corpus.
- **Color Space Standardization:** Automatic conversion of all input formats (RGBA, Grayscale, Palette) to 3-channel standard RGB.

---

## 4. Dataset Partitioning & Zero-Leakage Protocol

To ensure rigorous scientific evaluation and prevent data leakage:
- **Stratified Partitioning:** Stratified 70/15/15 split maintaining exact class balance across all splits:
  - **Training Set (70%):** 4,751 images (69.95%)
  - **Validation Set (15%):** 1,021 images (15.03%)
  - **Held-Out Test Set (15%):** 1,020 images (15.02%)
- **Deterministic Random Seed:** Fixed `random_seed = 42` for exact reproducibility.
- **Verified Disjointness:** Formally verified zero data leakage between subsets:
  $$\text{Train} \cap \text{Val} = \emptyset, \quad \text{Train} \cap \text{Test} = \emptyset, \quad \text{Val} \cap \text{Test} = \emptyset$$

---

## 5. Image Preprocessing & Augmentation

### Preprocessing Pipeline
1. **Format Standardization:** PIL image loading with safe RGB conversion.
2. **Geometric Standardization:** Bicubic interpolation resizing to fixed dimensions $(224, 224)$.
3. **Tensor Conversion:** Conversion to `torch.float32` tensors in range $[0.0, 1.0]$.
4. **ImageNet Normalization:** Channel-wise standardization using ImageNet statistics:
   - Mean: $\mu = [0.485, 0.456, 0.406]$
   - Standard Deviation: $\sigma = [0.229, 0.224, 0.225]$
5. **Tensor Validation:** Validated shape `[3, 224, 224]`, float32 precision, and finite values ($\forall x \in \text{tensor}, x \notin \{\text{NaN}, \pm\infty\}$).

### Domain-Tailored Augmentation
Training transforms apply safe, non-destructive augmentations preserving stroke topology:
- **Random Affine Transformations:** Subtle rotation ($\pm 8^\circ$), translation ($\pm 4\%$), and scale variation ($0.95 - 1.05\times$).
- **Color Jitter:** Subtle brightness and contrast variations ($\pm 10\%$) simulating variable lighting and surface shadows.

---

## 6. Model Architecture & Components

### Pretrained Vision Transformer Backbone (ViT-B/16)
- **Patch Size:** $16 \times 16$ pixels ($14 \times 14 = 196$ patches per image).
- **Encoder Blocks:** 12 Transformer encoder layers with multi-head self-attention.
- **Hidden Feature Dimension:** 768 dimensions.
- **Pretrained Weights:** ImageNet-1K pretrained weights for transferable visual feature extraction.

### Metric Embedding Projector
- **Architecture:** Two-layer MLP with LayerNorm and intermediate GELU non-linearity:
  $$\text{LayerNorm}(768) \to \text{Linear}(768 \to 512) \to \text{GELU} \to \text{Dropout}(0.1) \to \text{Linear}(512 \to 256)$$
- **L2 Normalization:** Output embeddings are projected onto a 256-dimensional unit hypersphere ($\|\mathbf{z}\|_2 = 1.0$).

### Classification Heads
1. **Cosine Similarity Head (Default Production Head):** Computes normalized cosine similarities between embedding $\mathbf{z}$ and learnable class weight vectors $\mathbf{w}_c$ with a learnable scale parameter $s$:
   $$\text{logits}_c = s \cdot \frac{\mathbf{z} \cdot \mathbf{w}_c}{\|\mathbf{z}\|_2 \|\mathbf{w}_c\|_2}$$
2. **Linear Classification Head:** Standard linear layer mapping $\mathbb{R}^{256} \to \mathbb{R}^{62}$.
3. **Prototypical Head:** Computes class centroids for episodic metric-based distance computation.

---

## 7. Transfer Learning & Training Strategy

DeepScript employs a structured two-stage transfer learning strategy:

```
┌────────────────────────────────────────────────────────┐
│ STAGE 1: HEAD WARMUP                                   │
│ • ViT Backbone: FROZEN (85.8M parameters)              │
│ • Projector + Head: TRAINABLE (804.8K parameters)      │
│ • Head LR: 1e-3 | Backbone LR: 0.0                     │
│ • Optimizer: AdamW | Scheduler: CosineAnnealingLR      │
└────────────────────────────────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│ STAGE 2: FINE-TUNING                                   │
│ • ViT Backbone: Unfreeze Last 2 Blocks (Layers 10 & 11)│
│ • Trainable Params: 14.98M parameters                  │
│ • Differential LR: Backbone LR = 2e-5, Head LR = 2e-4  │
│ • Optimizer: AdamW | Scheduler: CosineAnnealingLR      │
└────────────────────────────────────────────────────────┘
```

---

## 8. Actual Observed Training & Test Results

### 8.1 Multi-Epoch Supervised Training Progress

| Epoch | Stage | Train Loss | Train Acc (Top-1) | Val Loss | Val Acc (Top-1) | Val Acc (Top-3) | Head LR | Duration |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | Stage 1: Warmup | 1.6299 | 55.61% | 0.7376 | 77.08% | 94.32% | $1.0 \times 10^{-3}$ | 1,759.0s |
| **2** | Stage 1: Warmup | 0.4221 | 86.55% | 0.4403 | 85.70% | 97.55% | $5.0 \times 10^{-4}$ | 2,627.5s |
| **3** | Stage 2: Fine-Tuning | 0.2317 | 92.53% | 0.3236 | 89.32% | 98.63% | $2.0 \times 10^{-4}$ | 2,669.6s |
| **4** | Stage 2: Fine-Tuning | **0.1373** | **96.00%** | **0.2591** | **92.26%** | **98.73%** | $1.0 \times 10^{-4}$ | 1,824.5s |

### 8.2 Held-Out Test Set Evaluation (Checkpoint: `checkpoints/best_vit_model.pth`)

Evaluation was conducted on the completely isolated test split ($N = 1,020$ samples):

| Metric | Full Training Result | Smoke-Test Baseline | Uniform Random Baseline (62 Classes) |
| :--- | :---: | :---: | :---: |
| **Evaluated Test Samples** | **1,020 images** | 1,020 images | — |
| **Cross-Entropy Test Loss** | **0.2944** | 3.9030 | $\approx 4.1271$ |
| **Top-1 Test Accuracy** | **89.71%** | 14.41% | $\approx 1.61\%$ ($1/62$) |
| **Top-3 Test Accuracy** | **98.33%** | 28.53% | $\approx 4.84\%$ ($3/62$) |
| **Macro Precision** | **90.46%** | 8.58% | — |
| **Macro Recall** | **89.60%** | 13.72% | — |
| **Macro F1-Score** | **89.61%** | 7.81% | — |
| **Weighted F1-Score** | **89.69%** | 8.20% | — |

---

## 9. Current Implementation Status

### COMPLETED & VERIFIED
- [x] Dataset loading, validation, and dynamic 62-class discovery.
- [x] Stratified 70/15/15 train/val/test data partitioning with verified zero data leakage.
- [x] Image preprocessing pipeline (RGB conversion, bicubic 224×224 resizing, ImageNet normalization).
- [x] Domain-safe image augmentations (subtle affine transformations, color jitter).
- [x] Vision Transformer (ViT-B/16) backbone integration.
- [x] 768-dimensional feature extraction with frozen backbone parameter verification.
- [x] MLP embedding projector (768 $\to$ 512 $\to$ 256) with L2 unit-sphere normalization.
- [x] Classification heads (Cosine Similarity Head, Linear Head, Prototypical Head).
- [x] Two-stage transfer learning pipeline (Stage 1 warmup + Stage 2 differential fine-tuning).
- [x] Supervised training engine with validation model selection (`training/trainer.py`, `training/train.py`).
- [x] **Full Multi-Epoch Model Training** executed across all 4,751 training images.
- [x] **High-Performance Trained Model Checkpoint** saved to `checkpoints/best_vit_model.pth` (Val Acc: 92.26%).
- [x] **Held-Out Test Evaluation Benchmark** (89.71% Top-1, 98.33% Top-3, 89.61% Macro F1).
- [x] Experiment artifacts generated (training curves plot, confusion matrix heatmap, metrics JSON, report CSV).
- [x] Comprehensive test suites (`verify_preprocessing.py`, `verify_models.py`, `test_feature_extraction.py`).
- [x] **FastAPI Backend Service:** Production REST API endpoints (`/predict`, `/health`, `/classes`) serving real-time model inference.
- [x] **End-to-End Live Integration:** Seamless Vite proxy and React frontend connected to live PyTorch Vision Transformer inference.
- [x] **62-Class Epigraphic & Phonetic Dossier:** Rich character mapping connecting model predictions to paleographic context.

### PENDING NEXT MILESTONES
- [ ] **Few-Shot Learning Training:** Episodic N-way K-shot training using Prototypical Networks against the trained ViT baseline.
- [ ] **Docker Containerization:** Multi-stage Docker packaging for cloud deployment.

---

## 10. Reproduction & Verification Guide

### Prerequisites
Install dependencies from `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 1. Run Verification Test Suites
```bash
python verify_preprocessing.py
python verify_models.py
python test_feature_extraction.py
```

### 2. Run Full Multi-Epoch Training Pipeline
Executes the two-stage transfer learning pipeline:
```bash
python training/train.py --config configs/training_config.yaml
```

### 3. Run Test Set Evaluation & Generate Artifacts
Evaluates the saved model checkpoint on the 1,020 isolated test samples and generates plots:
```bash
python evaluation/evaluate.py --checkpoint checkpoints/best_vit_model.pth --batch_size 32
```

---

## 11. Project Directory Structure

```
Deepscript/
├── README.md                     # Central project documentation
├── requirements.txt              # Project dependencies
├── preprocessing_comparison.png  # Preprocessing visual comparison
├── configs/                      # Experiment and training configurations
│   ├── training_config.yaml      # Master training configuration
│   └── README.md
├── results/                      # Observed training and test evaluation results
│   ├── training/
│   │   ├── training_history.csv  # Training & validation telemetry
│   │   ├── training_metadata.json# Experiment hyperparameters and duration
│   │   └── training_curves.png   # Multi-panel loss/acc/LR plot
│   ├── evaluation/
│   │   ├── metrics.json          # Test metrics summary
│   │   ├── classification_report.csv # Per-class metrics
│   │   └── confusion_matrix.png  # 62-class normalized heatmap
│   ├── training_results.md       # Training results breakdown
│   ├── test_results.md           # Test results breakdown
│   └── README.md
├── dataset/                      # Inscription dataset images and metadata
├── preprocessing/                # Dataset loaders, transforms, and stratified split logic
│   ├── dataset.py                # AncientScriptDataset class
│   ├── transforms.py             # Preprocessing & augmentation pipelines
│   ├── splits.py                 # Stratified split & leakage verification functions
│   └── visualize.py              # Visual comparison utilities
├── models/                       # Model definitions and architectures
│   ├── backbone.py               # ViTFeatureExtractor and EmbeddingProjector
│   ├── classifier.py             # Prototypical, Cosine, and Linear heads
│   └── deepscript_model.py       # Unified DeepScriptModel interface
├── training/                     # Training routines and trainer classes
│   ├── train.py                  # Training CLI entrypoint
│   └── trainer.py                # Supervised Trainer implementation
├── evaluation/                   # Test set evaluation suite
│   └── evaluate.py               # Evaluation CLI & metrics computation
├── backend/                      # FastAPI inference service (planned)
├── frontend/                     # React user interface prototype
├── docs/                         # Detailed ML documentation
│   ├── architecture.md           # DeepScript model architecture guide
│   ├── training.md               # Training procedure and hyperparameters
│   ├── evaluation.md             # Test benchmark and metrics analysis
│   └── experiment_log.md         # Experiment history and evolution
├── verify_preprocessing.py       # Dataset verification test suite
├── verify_models.py              # Model architecture verification test suite
└── test_feature_extraction.py    # End-to-end feature extraction test
```

---

## 12. Current Limitations

- **Recognition Scope:** DeepScript identifies script classes only; it does not perform character segmentation, word bounding box extraction, OCR, or text translation.
- **Physical Media Noise:** Severe surface deterioration or damaged inscriptions may introduce visual artifacts impacting classification confidence.

---

## 13. License & Contributors

- **Project:** DeepScript Research Project
- **Lead Developer:** Hanvith Reddy

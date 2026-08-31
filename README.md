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
                      ↓
[ Image Preprocessing: Resize 224x224 (Bicubic) + ImageNet Normalization ]
                      ↓
[ Pretrained Vision Transformer (ViT-B/16) Backbone ]
                      ↓ (768-dimensional features)
[ Metric Embedding Projector (MLP: 768 → 512 → 256 + L2 Normalization) ]
                      ↓ (256-dimensional unit-norm embedding)
[ Classification Head (Cosine Similarity Head / Linear Head / Prototypical Head) ]
                      ↓
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

To ensure rigorous evaluation and prevent data leakage:
- **Stratified Partitioning:** Stratified 70/15/15 split maintaining exact class balance across all splits.
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
- **Random Affine Transformations:** Slight rotation ($\pm 10^\circ$), translation ($\pm 4\%$), and scale variation ($0.95 - 1.05\times$).
- **Color Jitter:** Subtle brightness and contrast variations ($\pm 10\%$) simulating variable lighting and surface shadows.

---

## 6. Model Architecture & Components

### Pretrained Vision Transformer Backbone (ViT-B/16)
- **Patch Size:** $16 \times 16$ pixels ($14 \times 14 = 196$ patches per image).
- **Encoder Blocks:** 12 Transformer encoder layers with multi-head self-attention.
- **Hidden Feature Dimension:** 768 dimensions.
- **Pretrained Weights:** ImageNet-1K pretrained weights for transferable visual feature extraction.

### Metric Embedding Projector
- **Architecture:** Two-layer MLP with intermediate non-linearity:
  $$\text{Linear}(768 \to 512) \to \text{ReLU} \to \text{Dropout}(0.1) \to \text{Linear}(512 \to 256)$$
- **L2 Normalization:** Output embeddings are projected onto a 256-dimensional unit hypersphere ($\|\mathbf{z}\|_2 = 1.0$).

### Classification Heads
1. **Cosine Similarity Head (Default):** Computes normalized cosine similarities between embedding $\mathbf{z}$ and learnable class weight vectors $\mathbf{w}_c$ with a learnable scale parameter $s$:
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
                           ↓
┌────────────────────────────────────────────────────────┐
│ STAGE 2: FINE-TUNING                                   │
│ • ViT Backbone: Unfreeze Last 2 Blocks (Layers 10 & 11)│
│ • Trainable Params: 14.8M parameters                   │
│ • Differential LR: Backbone LR = 2e-5, Head LR = 2e-4  │
│ • Optimizer: AdamW | Scheduler: CosineAnnealingLR      │
└────────────────────────────────────────────────────────┘
```

---

## 8. Verified Preliminary Smoke-Test Results

> [!NOTE]
> The metrics below represent **PRELIMINARY SMOKE-TEST RESULTS** from a rapid execution run designed to verify pipeline correctness, gradient flow, checkpoint serialization, and evaluation metrics. These are **NOT** final converged model results.

### Training & Validation Smoke-Test Progress
| Phase | Stage | Train Loss | Train Acc (Top-1) | Val Loss | Val Acc (Top-1) | Val Top-3 Acc | Learning Rate |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Stage 1** | Head Warmup | 4.3909 | 2.08% | 4.1200 | 2.08% | — | $1.0 \times 10^{-3}$ |
| **Stage 2** | Fine-Tuning | 3.9108 | 9.38% | 3.0524 | 10.42% | 62.50% | $2.0 \times 10^{-5}$ |

### Held-Out Test Set Evaluation (Checkpoint: `checkpoints/best_vit_model.pth`)
Evaluation was conducted on the completely isolated test split ($N = 1,020$ samples):

| Metric | Observed Smoke-Test Result |
| :--- | :--- |
| **Evaluated Test Samples** | 1,020 images (32 mini-batches) |
| **Cross-Entropy Test Loss** | 3.9030 |
| **Top-1 Test Accuracy** | **14.41%** |
| **Top-3 Test Accuracy** | **28.53%** |
| **Macro Precision** | 8.58% |
| **Macro Recall** | 13.72% |
| **Macro F1-Score** | 7.81% |
| **Weighted F1-Score** | 8.20% |

---

## 9. Current Implementation Status

### COMPLETED
- [x] Dataset loading, validation, and dynamic 62-class discovery.
- [x] Stratified 70/15/15 train/val/test data partitioning with zero data leakage.
- [x] Image preprocessing pipeline (RGB conversion, bicubic 224x224 resizing, ImageNet normalization).
- [x] Domain-safe image augmentations (subtle affine transformations, color jitter).
- [x] Vision Transformer (ViT-B/16) backbone integration.
- [x] 768-dimensional feature extraction with frozen backbone parameter verification.
- [x] MLP embedding projector (768 $\to$ 512 $\to$ 256) with L2 unit-sphere normalization.
- [x] Classification heads (Cosine Similarity Head, Linear Head, Prototypical Head).
- [x] Two-stage transfer learning pipeline (Stage 1 warmup + Stage 2 differential fine-tuning).
- [x] Supervised training engine with validation tracking and best checkpoint saving (`training/trainer.py`).
- [x] Test set evaluation suite (`evaluation/evaluate.py`).
- [x] Comprehensive programmatic verification test suites (`verify_preprocessing.py`, `verify_models.py`, `test_feature_extraction.py`).
- [x] Minimal frontend prototype for script visualization.

### PRELIMINARY / VERIFIED
- [x] Smoke-test training run successfully executed and verified on real data.
- [x] Smoke-test model checkpoint saved to `checkpoints/best_vit_model.pth`.
- [x] Test evaluation executed on isolated 1,020 test samples (Top-1: 14.41%, Top-3: 28.53%).

### PENDING
- [ ] **Few-Shot Learning Training:** Episodic N-way K-shot training using Prototypical Networks.
- [ ] **Full-Scale Model Training:** Multi-epoch GPU training across all 62 classes to achieve optimal classification convergence.
- [ ] **Hyperparameter Optimization:** Systematic exploration of learning rates, warmup schedules, and projector architectures.
- [ ] **FastAPI Backend Service:** Production REST API endpoints (`/predict`, `/health`) serving model inference.
- [ ] **End-to-End Integration:** Direct connection between frontend UI and backend inference service.

---

## 10. Reproduction & Verification Guide

### Prerequisites
Install dependencies from `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 1. Run Dataset & Preprocessing Verification
Verifies dataset discovery, 62 classes, label mappings, preprocessing shapes, stratified splits, and zero data leakage:
```bash
python verify_preprocessing.py
```

### 2. Run Model Architecture Verification
Verifies ViT feature extraction, L2 projector normalization, classification heads, prototypical centroid calculations, forward pass API, real data batch inference, and parameter freezing:
```bash
python verify_models.py
```

### 3. Run Feature Extraction Test
Verifies end-to-end pipeline on a real image from the dataset through pretrained ViT-B/16:
```bash
python test_feature_extraction.py
```

### 4. Run Smoke-Test Training Pipeline
Executes the two-stage transfer learning smoke test (Stage 1 Warmup + Stage 2 Fine-Tuning):
```bash
python training/train.py --warmup_epochs 2 --finetune_epochs 2 --batch_size 32 --unfreeze_blocks 2 --max_train_batches 3 --max_val_batches 2
```

### 5. Run Test Set Evaluation
Evaluates the saved model checkpoint on the 1,020 isolated test samples:
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
├── results/                      # Observed training and test evaluation results
├── dataset/                      # Inscription dataset images and metadata
├── preprocessing/                # Dataset loaders, transforms, and stratified split logic
│   ├── dataset.py                # AncientScriptDataset class
│   ├── transforms.py             # Preprocessing & augmentation pipelines
│   ├── splits.py                 # Stratified split & leakage verification functions
│   └── visualize.py              # Visual comparison utilities
├── models/                       # Model definitions and architectures
│   ├── backbone.py               # ViTFeatureExtractor backbone wrapper
│   ├── classifier.py             # EmbeddingProjector and classification heads
│   └── deepscript_model.py       # Unified DeepScriptModel interface
├── training/                     # Training routines and trainer classes
│   ├── train.py                  # Training CLI entrypoint
│   └── trainer.py                # Supervised Trainer implementation
├── evaluation/                   # Test set evaluation suite
│   └── evaluate.py               # Evaluation CLI & metrics computation
├── backend/                      # FastAPI inference service
├── frontend/                     # React user interface prototype
├── notebooks/                    # Exploratory analysis notebooks
├── docs/                         # Additional documentation
├── verify_preprocessing.py       # Dataset verification test suite
├── verify_models.py              # Model architecture verification test suite
└── test_feature_extraction.py    # End-to-end feature extraction test
```

---

## 12. Current Limitations

- **Recognition Scope:** DeepScript identifies script classes only; it does not perform character segmentation, word bounding box extraction, OCR, or text translation.
- **Smoke-Test Optimization:** Current accuracy metrics reflect a preliminary smoke test; full convergence requires multi-epoch training on dedicated GPU hardware.
- **Physical Media Noise:** Severe surface deterioration or damaged inscriptions may introduce visual artifacts impacting classification confidence.

---

## 13. License & Contributors

- **Project:** DeepScript Research Project
- **Lead Developer:** Hanvith Reddy

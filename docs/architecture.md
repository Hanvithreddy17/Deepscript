# DeepScript Model Architecture & System Design

This document details the deep learning vision architecture, feature extraction pipeline, metric embedding projector, and classification heads implemented in DeepScript.

---

## 1. System Pipeline Overview

```
[ Input Inscription Image: Grayscale / RGBA / RGB ]
                          │
                          ▼
[ Image Preprocessing: Safe RGB Conversion + 224×224 Bicubic Resize + ImageNet Normalization ]
                          │
                          ▼
[ Vision Transformer (ViT-B/16) Feature Backbone ]
                          │  (768-dimensional dense representations)
                          ▼
[ Metric Embedding Projector (MLP: 768 → 512 → 256 + L2-Normalization) ]
                          │  (256-dimensional unit-norm hypersphere embedding, ||z||₂ = 1.0)
                          ▼
[ Modular Classification Head (Cosine Similarity Head / Linear Head / Prototypical Head) ]
                          │
                          ▼
[ Script Class Logits, Top-1 Prediction, and Top-k Confidence Distributions ]
```

---

## 2. Component Specifications

### 2.1 Pretrained Vision Transformer Backbone (`ViT-B/16`)
- **Model Family:** Vision Transformer (Dosovitskiy et al., ICLR 2021).
- **Patch Resolution:** $16 \times 16$ pixels ($14 \times 14 = 196$ non-overlapping image patches for a $224 \times 224$ input).
- **Encoder Blocks:** 12 Transformer encoder layers with multi-head self-attention (12 attention heads per layer).
- **Hidden Dimensionality:** $D = 768$.
- **Pretrained Weights:** ImageNet-1K (`torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1`).
- **Selective Layer Freezing:**
  - Layers 0–9 remain frozen during transfer learning to retain low- and mid-level visual feature primitives.
  - Layers 10–11 are unfrozen in Stage 2 fine-tuning to adapt high-level attention maps to inscription stroke geometries.

### 2.2 Metric Embedding Projector
- **Structure:** Two-layer Multi-Layer Perceptron (MLP) with LayerNorm, GELU activation, and dropout:
  $$\mathbf{h} = \text{Dropout}(\text{GELU}(\mathbf{W}_1 \text{LayerNorm}(\mathbf{f}) + \mathbf{b}_1), p = 0.1)$$
  $$\mathbf{z}_{\text{raw}} = \mathbf{W}_2 \mathbf{h} + \mathbf{b}_2$$
  $$\mathbf{z} = \frac{\mathbf{z}_{\text{raw}}}{\|\mathbf{z}_{\text{raw}}\|_2}$$
- **Layer Dimensions:** $\mathbf{W}_1 \in \mathbb{R}^{512 \times 768}$, $\mathbf{W}_2 \in \mathbb{R}^{256 \times 512}$.
- **Hyperspherical Normalization:** Projecting onto the unit sphere ($\|\mathbf{z}\|_2 = 1.0$) ensures that distance metrics operate on angular similarity, mitigating gradient variances caused by feature magnitude fluctuations.

### 2.3 Classification Heads

#### 1. Cosine Similarity Head (Default Production Head)
Computes scaled angular similarity between the unit-normalized image embedding $\mathbf{z}$ and learnable unit-normalized class weight vectors $\mathbf{w}_c$:
$$\text{logits}_c = s \cdot \frac{\mathbf{z} \cdot \mathbf{w}_c}{\|\mathbf{z}\|_2 \|\mathbf{w}_c\|_2 + \epsilon}$$
- **Scale Factor:** Learnable scalar parameter $s$ (initialized to $s = 16.0$) to scale the bounded cosine range $[-1.0, 1.0]$ for standard cross-entropy loss.
- **Advantage:** Prevents dominant class gradient collapse in long-tailed or fine-grained script datasets.

#### 2. Linear Classification Head
Standard linear projection layer mapping $\mathbb{R}^{256} \to \mathbb{R}^{62}$ with optional dropout regularization.

#### 3. Prototypical Head
Computes Euclidean distances between query embeddings $\mathbf{z}_q$ and class centroid prototypes $\mathbf{c}_k$:
$$d(\mathbf{z}_q, \mathbf{c}_k) = \|\mathbf{z}_q - \mathbf{c}_k\|_2^2$$
$$P(y = k \mid \mathbf{x}_q) = \frac{\exp(-d(\mathbf{z}_q, \mathbf{c}_k) / \tau)}{\sum_{j=1}^K \exp(-d(\mathbf{z}_q, \mathbf{c}_j) / \tau)}$$

---

## 3. Parameter Summary

| Component | Total Parameters | Trainable in Stage 1 | Trainable in Stage 2 |
| :--- | :---: | :---: | :---: |
| **ViT Backbone (Layers 0–9)** | 71,621,376 | 0 | 0 |
| **ViT Backbone (Layers 10–11)** | 14,177,280 | 0 | 14,177,280 |
| **Embedding Projector (MLP)** | 525,120 | 525,120 | 525,120 |
| **Cosine Head (62 Classes)** | 279,745 | 279,745 | 279,745 |
| **Total System** | **86,603,521** | **804,865** | **14,982,145** |

---

## 4. Inference Backend & Serving Architecture

```
[ Inscription Image (File / Base64 Data URL) ]
                      │
                      ▼
[ FastAPI POST /predict Endpoint ]
                      │
                      ▼
[ Validation & Normalization Pipeline: PIL RGB → Resize(224×224) → ImageNet Norm ]
                      │
                      ▼
[ Pre-warmed DeepScriptModel(ViT-B/16 + CosineHead) Forward Pass ]
                      │
                      ▼
[ Softmax Normalization + Top-k Ranking (k=5) ]
                      │
                      ▼
[ JSON Output: Predicted Script, Confidence %, Top-5 Distribution, Latency (ms) ]
```

- **Async Lifespan Management:** Initializes `best_vit_model.pth` once at server boot, assigns to optimal device (`CUDA` or multi-threaded CPU), and runs an initial dummy forward pass to eliminate first-request latency spikes.
- **REST Contract:**
  - `POST /predict`: Upload image file (`multipart/form-data`) $\to$ script class, confidence, candidates.
  - `GET /health`: Server health, device name, and model checkpoint metadata.
  - `GET /classes`: Full alphabetical array of all 62 supported script classes.


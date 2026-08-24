# DeepScript Model Architecture (`models/`)

This directory contains the Vision Transformer (ViT) backbone, metric embedding projector, and few-shot classification modules for the DeepScript ancient Indian script identification system.

---

## Architecture Overview

```
Input Image [B, 3, 224, 224]
          │
          ▼
 [ ViT Feature Extractor ]       ──► Raw Features [B, 768] (ViT-B/16, Swin-T)
          │
          ▼
 [ Embedding Projector ]         ──► L2-Normalized Metric Embeddings [B, 256]
          │
          ▼
 [ Classification / Metric Head ] ──► Logits [B, 62] & Confidence Probabilities
```

---

## Core Modules

### 1. `ViTFeatureExtractor` (`models/backbone.py`)
- **Vision Transformer Engine**: Supports `vit_b_16`, `vit_b_32`, `vit_l_16`, `swin_t`, and `resnet50` via `torchvision.models`.
- **Feature Extraction**: Strips standard 1000-class ImageNet heads to output rich spatial-attention representations ($d=768$).
- **Freezing & Fine-Tuning**: Provides `freeze()`, `unfreeze()`, and `unfreeze_last_n_blocks(n)` for fine-grained transfer learning.

### 2. `EmbeddingProjector` (`models/backbone.py`)
- **Metric Projection**: Projects high-dimensional backbone features into a compact metric embedding space ($768 \rightarrow 256$ or $512$).
- **Architectures**: Multi-Layer Perceptron (`MLP`: Linear $\rightarrow$ LayerNorm $\rightarrow$ GELU $\rightarrow$ Dropout $\rightarrow$ Linear) or Linear probe.
- **Normalization**: Enforces $L_2$ unit-sphere normalization ($\|e\|_2 = 1.0$) for stable distance and angular metric comparisons.

### 3. Classification Heads (`models/classifier.py`)
- **`PrototypicalHead`**:
  - Implements Prototypical Networks for few-shot episodic classification.
  - Computes class centroid prototypes: $\mathbf{c}_k = \frac{1}{|S_k|} \sum_{x \in S_k} f(x)$.
  - Calculates Euclidean and Cosine distances to prototypes with episodic cross-entropy loss.
- **`CosineSimilarityHead`**:
  - Normalized weight vectors $W \in \mathbb{R}^{C \times D}$ computing angular similarity logits $s_i = \frac{1}{\tau} \frac{W_i \cdot x}{\|W_i\| \|x\|}$.
  - Learnable temperature scaling factor ($\tau$) for calibrated confidence scores.
- **`LinearClassificationHead`**:
  - Standard linear probe with LayerNorm and Dropout for baseline multi-class supervised learning.

### 4. `DeepScriptModel` (`models/deepscript_model.py`)
- Integrated PyTorch `nn.Module` combining backbone, projector, and classification head.
- Provides unified methods: `forward()`, `extract_features()`, `predict()`, and `get_parameter_summary()`.

---

## Quick Usage Examples

### Instantiating Model with Factory

```python
from models import get_deepscript_model

# 1. Cosine similarity model for 62 script classes
model = get_deepscript_model(
    backbone_name="vit_b_16",
    num_classes=62,
    embedding_dim=256,
    head_type="cosine",
    pretrained=True,
    freeze_backbone=False,
)

# 2. Forward pass with batch of images
import torch
images = torch.randn(4, 3, 224, 224)
logits, embeddings = model(images, return_embeddings=True)
print(f"Logits shape: {logits.shape}")         # [4, 62]
print(f"Embeddings shape: {embeddings.shape}") # [4, 256]

# 3. Model inference with top-k predictions
results = model.predict(images, top_k=3)
print(f"Predicted class: {results['predicted_class']}")
print(f"Confidence score: {results['confidence']}")
```

### Prototypical Few-Shot Metric Evaluation

```python
from models import PrototypicalHead

proto_head = PrototypicalHead(metric="euclidean")

# Support set (e.g., 5 classes, 5 shots each = 25 samples)
support_embeddings = torch.randn(25, 256)
support_labels = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4])

# Compute centroids
prototypes, unique_classes = proto_head.compute_prototypes(support_embeddings, support_labels)

# Query set (10 query samples)
query_embeddings = torch.randn(10, 256)
query_labels = torch.tensor([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])

loss, accuracy = proto_head.compute_episodic_loss(
    query_embeddings, query_labels, prototypes, unique_classes
)
print(f"Episodic Loss: {loss.item():.4f}, Accuracy: {accuracy.item() * 100:.2f}%")
```

---

## Model Verification

To run the automated verification test suite:

```bash
python verify_models.py
```

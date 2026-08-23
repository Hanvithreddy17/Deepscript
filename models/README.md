# Models Directory

## Directory Purpose

This directory will store deep learning model architecture definitions, backbone feature extractors, few-shot classification heads, and model initialization logic.

> **Status Notice:** Model code and pretrained weight loading functions are not yet implemented. The architecture documented below represents the planned conceptual design.

---

## Planned Model Architecture

The proposed DeepScript model architecture combines a modern Vision Transformer (ViT) backbone with a metric-learning / few-shot classifier:

1. **Pretrained Vision Transformer (ViT) Backbone:**
   Leverages self-attention mechanisms pre-trained on large-scale visual data to extract rich spatial and contextual visual features from inscription images.

2. **Feature Extraction & Embedding Generation:**
   Maps preprocessed image inputs into compact, high-dimensional feature embedding vectors that capture fine-grained glyph shapes and script patterns.

3. **Few-Shot Classification Layer:**
   A metric-learning classification head (such as a **Prototypical Network**) that evaluates the distance between query image embeddings and class prototypes derived from support samples.

---

## Conceptual Overview

```
Input Image ──► [ ViT Backbone ] ──► [ Feature Embedding ] ──► [ Few-Shot Metric Layer ] ──► Class Prediction
                                                                       ▲
                                   Support Prototypes ─────────────────┘
```

- **Vision Transformer (ViT):** Functions as the visual representation engine, encoding complex stroke structures and text layout patterns.
- **Few-Shot Metric Layer:** Evaluates visual similarity between query embeddings and target script prototypes, allowing accurate classification even when training samples per script class are severely limited.

# Preprocessing Directory

## Directory Purpose

This directory will contain Python modules and utilities responsible for image loading, cleaning, standardization, transformation, and dataset augmentation pipelines prior to model input.

> **Status Notice:** Preprocessing scripts have not yet been implemented. This document outlines the planned design and specifications for the image pipeline.

---

## Planned Preprocessing Pipeline

The image preprocessing workflow will consist of the following standard stages:

1. **Image Loading:** Secure loading of inscription images from disk or memory buffers using OpenCV and Pillow.
2. **RGB Conversion:** Standardization of multi-channel formats (Grayscale, RGBA, BGR) to uniform 3-channel RGB.
3. **Resizing:** Scaling images to uniform spatial dimensions compatible with Vision Transformer (ViT) input requirements (e.g., $224 \times 224$ or $384 \times 384$ pixels).
4. **Normalization:** Applying mean and standard deviation normalization aligned with ImageNet or ViT pretraining standards.
5. **Noise Reduction (Optional):** Selective filtering or denoising to mitigate background stone texture where appropriate.
6. **Contrast Enhancement:** Adaptive contrast handling (e.g., CLAHE) for inscriptions on faded or weathered media.
7. **Training Augmentation:** Application of photometric and spatial augmentations (e.g., subtle rotations, slight brightness/contrast variations) to enhance model generalization.

---

## Historical Integrity Guidelines

> **Crucial Rule:** Data augmentations applied during training **must not distort historically meaningful script characteristics**. Severe morphological distortions, aggressive skewing, or destructive filtering that alters stroke geometry or structural glyph features could lead to false feature learning and must be strictly avoided.

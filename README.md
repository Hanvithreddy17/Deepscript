# DeepScript

DeepScript is an image-classification system designed to identify selected ancient Indian scripts from inscription images.

> **Important Note:** DeepScript is focused strictly on **Script Recognition** (classifying which script appears in an image). It is **NOT** an Optical Character Recognition (OCR), transcription, translation, or semantic interpretation tool.

---

## Project Overview

Ancient Indian inscriptions, engraved on stone, copper plates, and palm leaves, feature diverse scripts that evolved across historical periods. Identifying these scripts manually requires specialized epigraphic expertise. DeepScript leverages deep learning and modern vision architectures to automate script identification from visual inscription data.

## Problem Statement

Identifying ancient Indian scripts from physical inscriptions presents several challenges:
- High variability in physical media (decayed stone, uneven surfaces, weathering).
- Limited available labeled datasets for many historical scripts.
- Visual ambiguities across script evolutions.

An automated script classification system provides epigraphists, historians, and researchers with a reliable visual baseline tool for script identification.

## Objective

Develop a deep-learning-based image classification system capable of identifying selected ancient Indian scripts from inscription images using Vision Transformers (ViT) and Few-Shot Learning methodologies.

## Proposed Methodology

The methodology follows a modular pipeline designed for high feature extraction capacity and strong performance under limited sample sizes:

1. **Inscription Image Input:** Input high-resolution or cropped inscription image.
2. **Image Preprocessing:** Clean, resize, normalize, and handle contrast/noise.
3. **Vision Transformer (ViT):** Extract high-level visual features using a pretrained ViT backbone.
4. **Feature Embedding:** Map image representations into a normalized vector embedding space.
5. **Few-Shot Classification:** Compare feature embeddings against reference class representations (e.g., via Prototypical Networks or distance-based metric learning).
6. **Script Prediction & Score:** Output the predicted script identity alongside a similarity/confidence score.

## Core Pipeline

```
[ Inscription Image ]
         ↓
[ Image Preprocessing ]
         ↓
[ Vision Transformer (ViT) Backbone ]
         ↓
[ Feature Embedding ]
         ↓
[ Few-Shot Classification ]
         ↓
[ Script Prediction ]
         ↓
[ Similarity / Confidence Score ]
```

## Technology Stack

- **Language:** Python
- **Deep Learning Framework:** PyTorch, torchvision, `timm`, Hugging Face Transformers
- **Computer Vision & Image Processing:** OpenCV, Pillow
- **Data & Numerical Analysis:** NumPy, Pandas, Scikit-learn
- **Visualization:** Matplotlib
- **Backend Services:** FastAPI, Uvicorn
- **Development Environment:** VS Code / Antigravity
- **Version Control:** Git / GitHub

## Current ML Progress

### COMPLETED:
- Dataset pipeline (class detection, stratified splitting, dynamic discovery)
- Image preprocessing (standardization, bicubic resizing to 224x224, ImageNet normalization)
- Pretrained ViT-B/16 loading (ImageNet weights via torchvision)
- Feature extraction (frozen backbone extracting 768-d embeddings)

### NOT YET IMPLEMENTED:
- Transfer learning / fine-tuning
- Few-Shot classification
- Prototypical Networks
- Final classification
- Evaluation

## Planned Development Phases

1. **Dataset Research & Collection:** Identify and curate initial target script classes.
2. **Dataset Cleaning & Organization:** Standardize labels, partition splits, and assess quality.
3. **Image Preprocessing:** Implement image normalization, resizing, and domain-appropriate augmentations.
4. **ViT Feature Extraction:** Set up and evaluate pretrained Vision Transformer backbones.
5. **Few-Shot Classification:** Integrate metric learning / few-shot classification modules.
6. **Model Training & Experimentation:** Execute episodic training and hyperparameter tuning.
7. **Evaluation:** Compute accuracy, per-class metrics, precision/recall, and confusion matrices.
8. **FastAPI Backend Integration:** Build RESTful API serving model inference (`POST /predict`).
9. **Frontend Integration:** Build user interface for image uploads and prediction viewing.
10. **Final Testing & Documentation:** Perform end-to-end testing, error analysis, and final documentation.

## Project Structure

```
deepscript/
├── README.md             # Project overview & system documentation
├── .gitignore            # Git exclusion rules
├── requirements.txt      # Python dependencies
├── dataset/              # Dataset guidelines, metadata, and references
├── preprocessing/        # Image preprocessing pipelines and utilities
├── models/               # ViT backbone & few-shot architecture definitions
├── training/             # Training routines and experiment scripts
├── evaluation/           # Metrics computation and evaluation reports
├── backend/              # FastAPI service and REST endpoints
├── frontend/             # User interface application
├── notebooks/            # Exploratory analysis and experiment notebooks
└── docs/                 # Extended specifications and research notes
```

## Evaluation Plan

Evaluation will measure script classification efficacy using:
- **Overall Accuracy**
- **Precision, Recall, F1-Score**
- **Per-Class Metrics** (critical due to inherent class imbalance in epigraphic data)
- **Confusion Matrix Analysis** to identify misclassifications between visually similar scripts

## Limitations

- **Script Recognition Only:** Does not perform character segmentation, OCR, transliteration, or translation.
- **Data Availability:** Efficacy is constrained by the availability and quality of historical inscription images.
- **Physical Wear:** Extreme physical erosion or surface damage on artifacts may degrade feature extraction quality.

## Future Scope

- Expansion to additional regional and historical Indian scripts.
- Integration of character-level segmentation or OCR pipelines.
- Mobile/edge deployment for field research use by epigraphists.

## Team

- DeepScript Project Team

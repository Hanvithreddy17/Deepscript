# DeepScript: Ancient Indian Script Identification System

DeepScript is an end-to-end deep learning system designed to identify ancient Indian script families from whole-inscription photographs (stone walls, cave rock faces, pillars, slabs, and copper charters).

> **Scope Clarification:** DeepScript performs **Script Family Identification** (classifying which script family an inscription belongs to). It is **NOT** an Optical Character Recognition (OCR), transcription, transliteration, or translation tool.

---

## 1. Project Overview

Ancient Indian inscriptions engraved across temple walls, caves, pillars, and copper charters represent millennia of cultural and historical heritage. Identifying historical scripts requires specialized epigraphic paleography expertise. DeepScript automates whole-inscription script identification using a fine-tuned Vision Transformer (ViT-B/16), metric feature embeddings, and cosine similarity classification.

### Supported 5-Class MVP Script Families
1. **Ashokan Brahmi** (`brahmi`) — 3rd Century BCE monumental imperial script.
2. **Gupta Script** (`gupta`) — 4th–8th Century CE late northern Brahmi with wedge headmarks.
3. **Kharosthi** (`kharosthi`) — 4th Century BCE – 3rd Century CE Gandharan right-to-left script.
4. **Grantha Script** (`grantha`) — 6th–14th Century CE South Indian rounded script for Sanskrit.
5. **Kadamba Script** (`kadamba`) — 5th–6th Century CE box-headed Western Brahmi of the Deccan.

---

## 2. System Architecture & ML Pipeline

```
[ Whole Inscription Image (Stone Wall / Cave / Pillar / Copper Plate) ]
                                 │
                                 ▼
[ Image Preprocessing: Resize 224×224 (Bicubic) + ImageNet Normalization ]
                                 │
                                 ▼
[ Pretrained Vision Transformer (ViT-B/16) Backbone ]
                                 │  (768-dimensional latent representations)
                                 ▼
[ Metric Embedding Projector (MLP: 768 → 512 → 256 + L2-Normalization) ]
                                 │  (256-dimensional unit-norm embedding, ||z||₂ = 1.0)
                                 ▼
[ Cosine Similarity Classification Head (5 Script Classes) ]
                                 │
                                 ▼
[ Script Family Prediction + Probability Distribution + Paleographic Dossier ]
```

---

## 3. Dataset & Provenance

- **Target Corpus:** `dataset/whole_inscriptions/`
- **Total Verified Images:** 49 whole-inscription specimens.
  - **Brahmi:** 10 images
  - **Gupta:** 10 images
  - **Kharosthi:** 10 images
  - **Grantha:** 10 images
  - **Kadamba:** 9 images
- **Metadata Provenance:** Full archaeological provenance, site records, and Wikimedia Commons source URLs catalogued in [`dataset/metadata.csv`](dataset/metadata.csv).
- **Format Standardization:** Automatic RGB conversion, bicubic interpolation, and ImageNet standardization.

---

## 4. Transfer Learning & Training Strategy

Training uses a two-stage transfer learning strategy:

1. **Stage 1 (Head Warmup - 3 Epochs):**
   - ViT backbone frozen (85.8M parameters).
   - Embedding projector & cosine head trained (AdamW, Head LR = $10^{-3}$, CosineAnnealingLR).
2. **Stage 2 (Fine-Tuning - 3 Epochs):**
   - Unfreezes last 2 Transformer encoder blocks (Layers 10 & 11).
   - Differential learning rates: Backbone LR = $2 \times 10^{-5}$, Head LR = $2 \times 10^{-4}$.

---

## 5. Evaluation Results

### Held-Out Test Set (Isolated 15% Split, $N = 6$ samples)
- **Top-1 Accuracy:** 100.00%
- **Top-3 Accuracy:** 100.00%
- **Test Cross-Entropy Loss:** 0.0348
- **Macro Precision / Recall / F1:** 1.0000 / 1.0000 / 1.0000

> **Small Dataset Disclaimer:** Because the current curated dataset contains 49 whole-inscription images across 5 classes (with 6 isolated test samples), these metrics validate that the ViT metric learning pipeline and cosine similarity head are working end-to-end. They do not imply universal real-world generalization across uncurated field images.

---

## 6. System Components

### 1. Primary Inference Engine: ViT-B/16
- **Module:** [`backend/vit_inference.py`](backend/vit_inference.py)
- **Checkpoint:** [`checkpoints/whole_inscription_vit/best_vit_model.pth`](checkpoints/whole_inscription_vit/best_vit_model.pth)
- **Execution:** Pure neural inference on CPU (~160–280 ms latency per image).

### 2. Optional Fallback Engine: Hugging Face VLM
- **Module:** [`backend/hf_vlm_engine.py`](backend/hf_vlm_engine.py)
- **Purpose:** Secondary multimodal vision-language paleographic feature analysis. Accessible via `?engine=vlm`.

### 3. FastAPI REST Backend
- **Endpoints:**
  - `GET /health` — Operational status, active model, and class list.
  - `GET /classes` — 5 supported script classes with historical metadata.
  - `POST /predict` — Classify uploaded whole-inscription image.
  - `POST /api/hf-config` — Configure optional Hugging Face VLM token/model.

### 4. React + Vite Frontend
- **Interactive Epigraphic Studio:** Specimen selector, custom upload, real-time image filters (high contrast, invert, monochrome), classification dashboard, and historical dossiers.

---

## 7. Quickstart & Execution Guide

### Prerequisites
Install Python dependencies:
```bash
pip install -r requirements.txt
```

### 1. Run Backend Server
```bash
python -m backend.main
# Server runs on http://localhost:8000
```

### 2. Run Frontend Web App
```bash
cd frontend
npm install
npm run dev
# Web app runs on http://localhost:3000
```

### 3. Run Test Suite
```bash
python backend/test_backend.py
python test_end_to_end.py
```

### 4. Re-evaluate Checkpoint
```bash
python evaluation/evaluate.py --checkpoint checkpoints/whole_inscription_vit/best_vit_model.pth --dataset_root dataset/whole_inscriptions --batch_size 8 --results_dir results/whole_inscription_vit --seed 42
```

---

## 8. Directory Structure

```
Deepscript/
├── backend/                      # FastAPI service & inference engines
│   ├── main.py                   # REST endpoints (/health, /classes, /predict)
│   ├── vit_inference.py          # Primary ViT-B/16 whole-inscription classifier
│   ├── hf_vlm_engine.py          # Optional Hugging Face VLM fallback
│   └── test_backend.py           # Backend integration test suite
├── checkpoints/                  # Trained PyTorch model checkpoints
│   └── whole_inscription_vit/    # 5-class whole-inscription ViT weights
├── dataset/                      # Dataset files & provenance
│   ├── metadata.csv              # Full epigraphic metadata & source URLs
│   └── whole_inscriptions/       # Verified whole-inscription images
│       ├── brahmi/
│       ├── grantha/
│       ├── gupta/
│       ├── kadamba/
│       └── kharosthi/
├── frontend/                     # React + Vite web dashboard
│   ├── src/
│   │   ├── components/           # UI components (Hero, Navbar, PredictionResult, etc.)
│   │   ├── data/                 # Sample specimens and script metadata
│   │   └── services/             # Backend API client
│   └── package.json
├── models/                       # ViT backbone, projector & classification heads
├── preprocessing/                # Image transforms & stratified splitting
├── training/                     # Two-stage supervised trainer
├── evaluation/                   # Test evaluation & metric reporting
└── tools/
    └── validate_dataset.py       # Dataset audit & validation utility
```

---

## 9. License & Attribution

- **Project:** DeepScript Research Project
- **Lead Developer:** Hanvith Reddy
- **Dataset Images:** Wikimedia Commons under CC BY-SA & Public Domain licenses (see `dataset/metadata.csv`).

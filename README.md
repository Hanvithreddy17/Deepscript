# DeepScript: Ancient Indian Script Identification System

DeepScript is an end-to-end deep learning system designed to identify ancient Indian script families from whole-inscription photographs (stone walls, cave rock faces, pillars, slabs, and copper charters).

> **Scope Clarification:** DeepScript performs **Script Family Identification** (classifying which script family an inscription belongs to). It is **NOT** an Optical Character Recognition (OCR), transcription, transliteration, or translation tool.

---

## 1. Project Overview

Ancient Indian inscriptions engraved across temple walls, caves, pillars, and copper charters represent millennia of cultural and historical heritage. Identifying historical scripts requires specialized epigraphic paleography expertise. DeepScript automates whole-inscription script identification using a fine-tuned Vision Transformer (ViT-B/16), metric feature embeddings, and cosine similarity classification.

### Supported 6 Epigraphic Script Families
1. **Ashokan Brahmi** (`brahmi`) — c. 3rd Century BCE monumental imperial Prakrit script of Emperor Ashoka.
2. **Tamil-Brahmi** (`tamil_brahmi`) — c. 3rd Century BCE – 5th Century CE early Sangam cavern script adapted for Old Tamil phonology.
3. **Gupta Script** (`gupta`) — c. 4th–8th Century CE late northern Brahmi / Siddhamātṛkā with wedge headmarks.
4. **Kharosthi** (`kharosthi`) — c. 4th Century BCE – 3rd Century CE Gandharan right-to-left sister script.
5. **Grantha Script** (`grantha`) — c. 6th–14th Century CE South Indian rounded script for classical Sanskrit.
6. **Kadamba Script** (`kadamba`) — c. 5th–6th Century CE box-headed Western Brahmi of the Deccan.

---

## 2. System Architecture & ML Pipeline

```
[ Whole Inscription Image (Stone Wall / Cave / Pillar / Copper Plate) ]
                                 │
                                 ▼
[ Input Validator: Blank/Degenerate Image Rejection + Integrity Checks ]
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
[ Cosine Similarity Classification Head (6 Script Classes) ]
                                 │
                                 ▼
[ Script Family Prediction + Probability Distribution + Epigraphic Dossier ]
```

---

## 3. Dataset & Provenance

- **Target Corpus:** `dataset/whole_inscriptions/`
- **Total Verified Images:** 61 whole-inscription specimens.
  - **Ashokan Brahmi (`brahmi`):** 10 images (Mauryan Rock & Pillar Edicts: Ramgarh, Kangra, Erragudi, Sarnath, Lumbini, Vidisha, Nashik).
  - **Tamil-Brahmi (`tamil_brahmi`):** 12 images (Sangam Cavern & Boulder Inscriptions: Mangulam, Sittannavasal, Arittapatti, Jambai, Pugalur, Alagarmalai, Edakal, Kongarpuliyankulam, Karungalakkudi, Kilavalavu, Aiyarmalai).
  - **Gupta (`gupta`):** 10 images (Samudragupta Eran, Chandragupta II Mathura/Sanchi, Bhitari, Allahabad Prashasti).
  - **Kharosthi (`kharosthi`):** 10 images (Shahbazgarhi, Mansehra, Niya wooden tablets, Bimaran reliquary).
  - **Grantha (`grantha`):** 10 images (Mandagapattu, Thalavanur, Velvikudi grant, Kurumathur).
  - **Kadamba (`kadamba`):** 9 images (Halmidi, Talagunda pillar & door jambs, Malavalli, Banavasi).
- **Provenance & Licences:**
  - Tamil-Brahmi stone inscriptions: *Tamil Brahmi Stone Inscription (TBSI)* dataset by Sukumar M / Tamil Nadu Department of Archaeology (TNDA), hosted on Mendeley Data (DOI: [10.17632/2spz2js9jp.1](https://data.mendeley.com/datasets/2spz2js9jp/1)), licensed under **CC BY 4.0**.
  - Brahmi, Gupta, Kharosthi, Grantha, Kadamba: Curated from Wikimedia Commons under Creative Commons (CC BY-SA 4.0 / CC BY-SA 3.0 / CC BY-SA 2.0 / Public Domain). Full per-image provenance recorded in [`dataset/metadata.csv`](dataset/metadata.csv).
- **Format Standardization:** Automatic RGB conversion, bicubic interpolation, and ImageNet standardization.

---

## 4. Transfer Learning & Training Strategy

Training uses a two-stage transfer learning strategy (`configs/training_config_6class.yaml`):

1. **Stage 1 (Head Warmup - 4 Epochs):**
   - ViT backbone frozen (85.8M parameters).
   - Embedding projector & cosine head trained (AdamW, Head LR = $10^{-3}$, CosineAnnealingLR).
2. **Stage 2 (Fine-Tuning - 4 Epochs):**
   - Unfreezes last 2 Transformer encoder blocks (Layers 10 & 11, 14.9M trainable parameters).
   - Differential learning rates: Backbone LR = $2 \times 10^{-5}$, Head LR = $2 \times 10^{-4}$.

---

## 5. Evaluation Results

### Held-Out Test Set (Isolated Partition, $N = 8$ samples)
- **Top-1 Accuracy:** 100.00%
- **Top-3 Accuracy:** 100.00%
- **Test Cross-Entropy Loss:** 0.0424
- **Macro Precision / Recall / F1:** 1.0000 / 1.0000 / 1.0000
- **Ashokan Brahmi vs. Tamil-Brahmi:** 0 cross-class confusion (Precision: 100.0%, Recall: 100.0% for both classes).

> **Epigraphic & Dataset Limitations:** While the model achieves high metric separation on the curated corpus ($N = 61$), whole-inscription recognition in open-world archaeological settings is subject to rock weathering, erosion, biogrowth, and poor lighting. Photos must exhibit visible glyph stroke relief. Sub-variety identification (e.g., exact regional Pandya vs. Chera cavern sub-phase) is not claimed beyond family-level classification.

---

## 6. System Components

### 1. Primary Inference Engine: ViT-B/16
- **Module:** [`backend/vit_inference.py`](backend/vit_inference.py)
- **Active 6-Class Checkpoint:** [`checkpoints/six_class_vit/best_vit_model.pth`](checkpoints/six_class_vit/best_vit_model.pth)
- **Preserved 5-Class Fallback Checkpoint:** [`checkpoints/whole_inscription_vit/best_vit_model.pth`](checkpoints/whole_inscription_vit/best_vit_model.pth)
- **Execution:** Pure neural inference on CPU (~180–260 ms latency per image).

### 2. Optional Fallback Engine: Hugging Face VLM
- **Module:** [`backend/hf_vlm_engine.py`](backend/hf_vlm_engine.py)
- **Purpose:** Multimodal vision-language paleographic feature analysis. Accessible via `?engine=vlm`.

### 3. FastAPI REST Backend
- **Endpoints:**
  - `GET /health` — Operational status, active model, and class list.
  - `GET /classes` — 6 supported script classes with historical epigraphic dossiers.
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

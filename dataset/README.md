# DeepScript Inscription Dataset

## 1. Project Objective & Purpose

The objective of **DeepScript** is to automatically identify and classify the **SCRIPT FAMILY** directly from photographic captures and rubbings of **WHOLE ancient Indian inscriptions**, especially those found on:
- Stone temple walls and bas-reliefs
- Monolithic pillars and edicts
- Natural rock surfaces and cavern beds
- Copper charters and stone slabs
- Archaeological monument faces

The pipeline utilizes a Vision Transformer backbone (**ViT-B/16**) coupled with metric embedding and few-shot / prototypical learning to generalize across varied stone weathering, uneven natural lighting, and erosion.

---

## 2. Dataset Architecture & Directory Structure

The dataset directory is partitioned strictly into whole-inscription training corpora and auxiliary character references:

```text
dataset/
├── whole_inscriptions/
│   ├── brahmi/            # Whole rock, pillar, and slab edicts in Ashokan / Early Brahmi
│   ├── gupta/             # Late Northern Brahmi / Gupta script inscriptions (4th–6th c. CE)
│   ├── kharosthi/         # Gandharan right-to-left inscriptions on slabs, schist, and bark
│   ├── grantha/           # Pallava / Chola temple wall inscriptions in Grantha script
│   ├── kadamba/           # Box-headed Deccan stone inscriptions (Kadamba / Early Chalukya)
│   └── tamil_brahmi/      # Cavern bed and rock-face cave inscriptions in Tamil-Brahmi
│
├── character_reference/
│   └── brahmi/            # Isolated character glyph references (used solely for metric prototyping)
│
├── metadata.csv           # Master provenance, license, split, and verification registry
└── README.md              # Dataset specifications and governance policies
```

---

## 3. Whole-Inscriptions vs. Character-Reference Data

| Characteristic | `whole_inscriptions/` | `character_reference/` |
| :--- | :--- | :--- |
| **Data Type** | Full field photographs, rubbings, and large multi-line panels | Isolated single character crops / glyph drawings |
| **Model Role** | Primary input for whole-inscription ViT-B/16 classification | Secondary reference data for few-shot metric spaces |
| **Visual Context** | Contains full stone texture, line flow, margins, and natural erosion | Cropped, segmented, or synthetic isolated glyphs |
| **Training Policy** | Directly evaluated on whole-inscription inference | **Never** mixed into whole-inscription training splits |

---

## 4. Supported Script Families

1. **Ashokan Brahmi:** 3rd Century BCE monumental script characterized by clean geometric forms (circle *ma*, cross *ka*, inverted-V *ga*).
2. **Gupta Script:** 4th–6th Century CE Northern epigraphy distinguished by solid triangular wedge headmarks and ornate conjuncts.
3. **Kharosthi:** 3rd Century BCE – 3rd Century CE Northwestern cursive script written right-to-left with distinctive descenders.
4. **Grantha:** 7th–13th Century CE South Indian script with flowing, ornate double loops and circular vowel marks.
5. **Kadamba Script:** 5th–7th Century CE Deccan script featuring hollow rectangular box-head topmarks.
6. **Tamil-Brahmi:** 3rd Century BCE – 2nd Century CE Southern rock and cave-bed script with specialized Dravidian phonetic signs.

---

## 5. Strict Labeling & Provenance Rules

1. **No Guessed Labels:** Never assign a script label based merely on visual similarity. A specimen must only be assigned a target class when verified by documented epigraphic citations.
2. **Mandatory Provenance:** Every sample entered into the dataset must be registered in `dataset/metadata.csv` with:
   - `source`: Academic publication, archaeological corpus (e.g., ASI *Epigraphia Indica*, *CII*, *SII*), or museum repository.
   - `source_url`: Persistent link or DOI to the archive.
   - `site`: Monument, cave, temple, or geographic provenance.
   - `verified`: Boolean (`TRUE`/`FALSE`) indicating ground-truth verification.
3. **Training Quarantine for Unverified Images:** Any image marked `verified = FALSE` or `script = UNKNOWN` is quarantined and **MUST NOT** be used for model training or validation.

---

## 6. Data Partitioning & Leakage Prevention Policy

1. **Disjoint Monument Splits:** Images originating from the same physical monument, pillar, or cave site must belong to the **same split** (train, validation, or test) to avoid spatial data leakage.
2. **Duplicate Prevention:** Inked rubbings, cropped sub-regions, or alternate camera angles of the same inscription must never be split across train and test sets.
3. **Fixed Stratification:** Standard partitioning targets 70% Train, 15% Validation, and 15% Test across all verified script families.

---

## 7. Image Quality & Preprocessing Guidelines

1. **Preserve Originals:** Store raw, uncompressed master images without lossy downsampling.
2. **No Ad-Hoc Squashing:** Do not artificially force non-square inscriptions into square aspect ratios.
3. **Minimum Quality:** Whole-inscription images should maintain sufficient resolution for character stroke visibility (minimum recommended resolution: $800 \times 600$ px).

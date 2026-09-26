# Whole-Inscription Dataset Provenance

## Objective

The DeepScript architecture is transitioning from isolated-character classification to an end-to-end whole-inscription pipeline:

$$\text{Whole Inscription Image} \longrightarrow \text{ViT-B/16 Backbone} \longrightarrow \text{Metric Embedding} \longrightarrow \text{Few-Shot Learning} \longrightarrow \text{Script Family Prediction}$$

Target script families include:
1. **Ashokan Brahmi**
2. **Gupta**
3. **Kharosthi**
4. **Grantha**
5. **Kadamba**
6. **Tamil-Brahmi**

Before training or fine-tuning vision transformers and few-shot metric spaces, establishing authentic, verifiable ground-truth labels is strictly required. Visual similarity (e.g., *"this looks like Brahmi"*) does not constitute ground truth and leads to catastrophic label noise. This document details the provenance audit and metadata verification for all whole-inscription image collections in the repository.

---

## Collections Investigated

All 237 whole-inscription images in the local repository were audited across 6 primary collections:

| Collection | Images | Script Label | Evidence | Confidence |
| :--- | :---: | :---: | :--- | :---: |
| **`OriginalImage*.jpg`** | 85 | `UNKNOWN` | MATLAB script [`noisefinal.m`](file:///c:/Users/hanvi/OneDrive/Desktop/Deepscript/Deepscript/dataset/noisefinal.m) references `shilalekhfeature.xlsx` / `Final Image DB\Temp`. General stone inscription category (*shilalekh*), but no script class or monument catalog documented. | `UNKNOWN` |
| **`I_*.JPG` / `II.JPG`** | 67 | `UNKNOWN` | MATLAB script [`noisem.m`](file:///c:/Users/hanvi/OneDrive/Desktop/Deepscript/Deepscript/dataset/noisem.m) references `G:\PHD\Final PHD Data\Final Database\I_2(1).JPG`. EXIF tags confirm Nikon D7200 field photography (Nov 2021). No script or site index in local repository. | `UNKNOWN` |
| **`I01.jpg` – `I44.jpg`** | 44 | `UNKNOWN` | Unannotated numbered photography / rubbing scans (43 JPEG, 1 GIF: `I25.gif`). No associated metadata or catalog text files exist. | `UNKNOWN` |
| **`T01.jpg` – `T27.jpg`** | 27 | `UNKNOWN` | High-resolution inked estampages and temple wall rubbings. No documentation mapping prefix `T` to a verified epigraph or script. | `UNKNOWN` |
| **`N01.jpg` – `N09.jpg`** | 11 | `UNKNOWN` | Pillar and rock edict photographs (Canon EOS Rebel T2i, Feb 2016). No site or script metadata provided. Includes 1 corrupted file (`N03.JPG`). | `UNKNOWN` |
| **`J01.jpg` – `J03.jpg`** | 3 | `UNKNOWN` | Unannotated cave / rock face inscription photographs. No class index or ground-truth documentation. | `UNKNOWN` |

---

## Verified Labels

* **Total Verified Images:** **0**
* **Verified Script Classes:** None currently verified from local repository files.

No image in the whole-inscription collections currently possesses an unambiguous, documented ground-truth script label in the repository. In accordance with strict data curation standards, no labels have been inferred or guessed.

---

## Unknown Images

All **237 images** across the investigated collections are cataloged as `UNKNOWN` pending authoritative provenance documentation:
- [`OriginalImage.jpg`](file:///c:/Users/hanvi/OneDrive/Desktop/Deepscript/Deepscript/dataset/OriginalImage.jpg) to `OriginalImage84.jpg` (85 files)
- [`II.JPG`](file:///c:/Users/hanvi/OneDrive/Desktop/Deepscript/Deepscript/dataset/II.JPG), [`I_1.JPG`](file:///c:/Users/hanvi/OneDrive/Desktop/Deepscript/Deepscript/dataset/I_1.JPG) to `I_46(3).JPG` (67 files)
- [`I01.jpg`](file:///c:/Users/hanvi/OneDrive/Desktop/Deepscript/Deepscript/dataset/I01.jpg) to `I44.jpg`, `I-19.JPG`, `I-8(4).JPG` (44 files)
- [`T01.jpg`](file:///c:/Users/hanvi/OneDrive/Desktop/Deepscript/Deepscript/dataset/T01.jpg) to `T27.jpg` (27 files)
- [`N01.jpg`](file:///c:/Users/hanvi/OneDrive/Desktop/Deepscript/Deepscript/dataset/N01.jpg) to `N09.jpg`, `N1.jpg`, `N2.jpg`, `No4.JPG` (11 files)
- [`J01.jpg`](file:///c:/Users/hanvi/OneDrive/Desktop/Deepscript/Deepscript/dataset/J01.jpg) to `J03.jpg` (3 files)

*(Machine-readable tracking for all 237 images is provided in [`dataset/whole_inscription_metadata.csv`](file:///c:/Users/hanvi/OneDrive/Desktop/Deepscript/Deepscript/dataset/whole_inscription_metadata.csv)).*

---

## Sources & Evidence Audited

1. **Local MATLAB Scripts ([`noisefinal.m`](file:///c:/Users/hanvi/OneDrive/Desktop/Deepscript/Deepscript/dataset/noisefinal.m), [`noisem.m`](file:///c:/Users/hanvi/OneDrive/Desktop/Deepscript/Deepscript/dataset/noisem.m))**:
   - Confirms images originate from an academic research project (*"Final PHD Data / Final Database / Shilalekh Feature Extraction"*).
   - Confirms algorithmic noise testing and filtering was performed on these stone edicts.
   - Does not contain script class indices or monument-level cross-references.
2. **Repository Documentation ([`dataset/README.md`](file:///c:/Users/hanvi/OneDrive/Desktop/Deepscript/Deepscript/dataset/README.md), [`README.md`](file:///c:/Users/hanvi/OneDrive/Desktop/Deepscript/Deepscript/README.md))**:
   - Outlines general project architecture and target classes; contains no per-image catalog.
3. **Image EXIF Metadata**:
   - Contains camera hardware specifications (Nikon D7200, Canon Rebel T2i) and software tags (`Picasa`, `gd-jpeg`); no textual script or site annotations.

---

## Dataset Problems Identified

1. **Missing Ground-Truth Labels:** 100% of whole-inscription images lack explicit script metadata in local project files.
2. **Exact Duplicate Images:** 
   - `OriginalImage38.jpg` == `OriginalImage55.jpg`
   - `OriginalImage39.jpg` == `OriginalImage58.jpg`
   - `OriginalImage40.jpg` == `OriginalImage64.jpg`
   - `OriginalImage50.jpg` == `OriginalImage84.jpg`
3. **Synthetic Derivative Files:** 170 files in `dataset/` (`HFI*.jpg` and `VFI*.jpg`) are horizontally and vertically filtered duplicates of `OriginalImage*.jpg` and do not represent independent inscriptions.
4. **Corrupted File:** [`dataset/N03.JPG`](file:///c:/Users/hanvi/OneDrive/Desktop/Deepscript/Deepscript/dataset/N03.JPG) contains a truncated JPEG byte stream (incomplete transfer).
5. **Heterogeneous Formats & Scales:**
   - File formats include JPEG and GIF (`I25.gif`).
   - Image resolutions vary from low-resolution crops ($272 \times 185$) to ultra-high-resolution photographic captures ($5475 \times 3644$).
   - `OriginalImage` series has been uniformly resized/squashed to square $1024 \times 1024$, altering original glyph aspect ratios.
6. **Unrelated Data in Root:** 3,446 modern Thai line crops (`a001`–`a140`) and `.gt.txt` transcription files reside directly in `dataset/` root.

---

## Recommended Labeling Plan

To construct a reliable whole-inscription training and few-shot evaluation set:

1. **Source Identification & Author Contact / Catalog Matching:**
   - Locate the original PhD thesis / research catalog (`shilalekhfeature.xlsx` or source index) that produced the `OriginalImage` and `I_` database.
   - Match high-resolution images (`I_1` – `I_46`, `T01` – `T27`, `N01` – `N09`) against published archaeological epigraphy corpora (e.g., Archaeological Survey of India *Epigraphia Indica*, *Corpus Inscriptionum Indicarum*, temple estampage collections).
2. **Metadata Population in [`dataset/whole_inscription_metadata.csv`](file:///c:/Users/hanvi/OneDrive/Desktop/Deepscript/Deepscript/dataset/whole_inscription_metadata.csv):**
   - Populate `script`, `location`, `period`, `source`, and `evidence` fields exclusively with documented historical references.
   - Update `status` to `VERIFIED` only when an authoritative publication or monument record confirms the script.
3. **Dataset Structuring & Hygiene:**
   - Segregate verified images into canonical script folders or clean manifest splits.
   - Exclude synthetic derivative duplicates (`HFI`/`VFI`) and repair/remove corrupted files (`N03.JPG`).

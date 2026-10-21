# DeepScript — Non-Stone Dataset Audit Metadata Report
**Substrate Classification, Modality Flagging, and Provenance Gap Analysis**

- **Date**: October 21, 2026
- **Project Root**: `C:\Users\hanvi\OneDrive\Desktop\Deepscript\Deepscript`
- **Audit Target File**: [`dataset/metadata.csv`](file:///c:/Users/hanvi/OneDrive/Desktop/Deepscript/Deepscript/dataset/metadata.csv) (74 total records)
- **Output Audit Registry**: [`dataset/metadata_stone_only.csv`](file:///c:/Users/hanvi/OneDrive/Desktop/Deepscript/Deepscript/dataset/metadata_stone_only.csv) (74 records, 21 fields)
- **Verified Checkpoint**: `checkpoints/six_class_vit/best_vit_model.pth`
- **SHA-256 Checksum**: `6efa33c8a11389f64b8fa5040a7de6c4cf8042fb5359889ca4a23f5653fdaa5c` (100% Intact)

---

## 1. Executive Summary

In accordance with Step 1 of the dataset expansion plan, an exhaustive audit of all 74 records in `dataset/metadata.csv` was conducted. A separate, non-destructive audit file—[`dataset/metadata_stone_only.csv`](file:///c:/Users/hanvi/OneDrive/Desktop/Deepscript/Deepscript/dataset/metadata_stone_only.csv)—has been generated. 

### Key Audit Findings
1. **Preservation of Original Data**: All 11 original metadata columns and all 74 original rows are completely preserved without alteration. The active production metadata file [`dataset/metadata.csv`](file:///c:/Users/hanvi/OneDrive/Desktop/Deepscript/Deepscript/dataset/metadata.csv) was left 100% untouched.
2. **Substrate Separation**: Of the 74 records, **60 are stone inscriptions (81.1%)** and **14 are non-stone artifacts or reproductions (18.9%)**.
3. **Flagged Non-Stone Examples (14 items)**:
   - **5 Copper Plates** (Grantha: Velvikudi Grants x2, Dharmeshwara; Gupta: Arang, Raipur).
   - **6 Wooden Tablets / Plates** (Kharosthi: Niya Silk Road administrative timber records).
   - **1 Terracotta Clay Seal** (Gupta: Nalanda monastic clay sealing of Vishnugupta).
   - **1 Steatite Reliquary Casket** (Kharosthi: Bimaran Stupa portable carved vessel).
   - **1 Paper Rubbing Facsimile** (Brahmi: Paderia / Lumbini Pillar ink stamping reproduction).
4. **Rows Requiring Epigraphic Review (4 items)**:
   - 4 stone inscriptions lack specific archaeological site or monument provenance (1 Brahmi, 2 Grantha, 1 Kadamba) and have been explicitly flagged with `audit_status = 'needs_review'`.
5. **Verified Clean Stone Corpus (56 items)**:
   - Exactly 56 items possess verified monumental stone substrate and definitive archaeological site and monument provenance.

---

## 2. Quantitative Distribution by Script Class and Physical Medium

The cross-tabulation below details the physical medium distribution across all six script families:

| Script Class | Total Records | Stone (Granite / Sandstone / Basalt) | Copper (Plates / Charters) | Wood (Timber Tablets / Plates) | Terracotta (Clay Seals) | Steatite (Portable Vessels) | Paper (Ink Rubbings) | Verified Stone | Flagged Non-Stone | Needs Review |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`brahmi`** | 10 | 9 | 0 | 0 | 0 | 0 | 1 | 8 | 1 | 1 |
| **`grantha`** | 10 | 7 | 3 | 0 | 0 | 0 | 0 | 5 | 3 | 2 |
| **`gupta`** | 10 | 7 | 2 | 0 | 1 | 0 | 0 | 7 | 3 | 0 |
| **`kadamba`** | 9 | 9 | 0 | 0 | 0 | 0 | 0 | 8 | 0 | 1 |
| **`kharosthi`** | 10 | 3 | 0 | 6 | 0 | 1 | 0 | 3 | 7 | 0 |
| **`tamil_brahmi`** | 25 | 25 | 0 | 0 | 0 | 0 | 0 | 25 | 0 | 0 |
| **Total** | **74** | **60** | **5** | **6** | **1** | **1** | **1** | **56** | **14** | **4** |

```
+─────────────────────────────────────────────────────────────────────────────+
|                     Physical Medium Breakdown (N=74)                        |
+─────────────────────────────────────────────────────────────────────────────+
| Stone (Monumental Rock, Pillar, Slab, Bed, Ledge) : 60 (81.1%)              |
| Wood (Silk Road Timber Charters)                  :  6  (8.1%)              |
| Copper (Engraved Royal Land Charters)             :  5  (6.8%)              |
| Terracotta (Monastic Stamped Seal)                :  1  (1.4%)              |
| Steatite (Portable Reliquary Vessel)              :  1  (1.4%)              |
| Paper (2D Ink Rubbing Facsimile)                  :  1  (1.4%)              |
+─────────────────────────────────────────────────────────────────────────────+
```

---

## 3. Detailed Inventory of Flagged Non-Stone Records (14 Items)

These 14 items exhibit domain mismatches with the whole-stone classifier target and must be excluded from whole-stone training and evaluation:

| Row Index | Script Class | Image File Path | Physical Medium | Non-Stone Flag | Inscription Type | Physical Monument / Artifact Identity | Reason for Flagging |
| :---: | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **09** | `brahmi` | `dataset/whole_inscriptions/brahmi/Paderia Pillar.png` | `paper` | `paper_rubbing` | `pillar_inscription` | Lumbini Rummindei Pillar of Ashoka | 2D ink rubbing / paper facsimile reproduction; not a primary stone surface capture. |
| **12** | `grantha` | `dataset/whole_inscriptions/grantha/8th Century Grantha and Vatteluttu Scripts Velvikudi Grant in Sanskrit and Tamil.jpg` | `copper` | `copper_plate` | `copper_plate` | Velvikudi Copper Plate Grant (Full Set) | Engraved copper plate charter (metal substrate); high specular reflectance. |
| **13** | `grantha` | `dataset/whole_inscriptions/grantha/8th Century Grantha script Sanskrit language Velvikudi Grant, LINES 8-14.jpg` | `copper` | `copper_plate` | `copper_plate` | Velvikudi Copper Plate Grant (Detail Lines 8-14) | Detail macro view of engraved copper plate charter. |
| **16** | `grantha` | `dataset/whole_inscriptions/grantha/Dharmeshwara Temple Plates.jpg` | `copper` | `copper_plate` | `copper_plate` | Dharmeshwara Temple Copper Plate Charter | Multi-plate copper charter bound by bronze ring. |
| **20** | `gupta` | `dataset/whole_inscriptions/gupta/6th-century Arang copper plate inscription, Hinduism, king Jayaraja, Chhattisgarh, Sanskrit.jpg` | `copper` | `copper_plate` | `copper_plate` | Arang Copper Plate Grant of King Jayaraja | 6th c. Sharabhapuriya royal copper plate grant. |
| **21** | `gupta` | `dataset/whole_inscriptions/gupta/6th-century Raipur copper plate inscription, Hinduism, king Sudevaraja, Chhattisgarh, Sanskrit.jpg` | `copper` | `copper_plate` | `copper_plate` | Raipur Copper Plate Grant of King Sudevaraja | 6th c. royal copper plate grant. |
| **28** | `gupta` | `dataset/whole_inscriptions/gupta/Nalanda clay seal of Vishnugupta.jpg` | `terracotta` | `clay_seal` | `clay_seal` | Nalanda Clay Sealing of Vishnugupta | Monastic terracotta sealing (baked clay impression); non-monumental portable object. |
| **40** | `kharosthi` | `dataset/whole_inscriptions/kharosthi/Document on Wooden Tablet written in Kharosthi Script (cropped).jpg` | `wood` | `wooden_tablet` | `wooden_tablet` | Niya Inscribed Wooden Administrative Tablet | Inscribed timber tablet from Niya oasis along Silk Road; lacks stone texture. |
| **43** | `kharosthi` | `dataset/whole_inscriptions/kharosthi/Kharoshti script on a wooden plate, National Museum, New Delhi 01.jpg` | `wood` | `wooden_plate` | `wooden_plate` | Niya Inscribed Wooden Wedge Plate 01 | Wooden wedge document excavated by Aurel Stein. |
| **44** | `kharosthi` | `dataset/whole_inscriptions/kharosthi/Kharoshti script on a wooden plate, National Museum, New Delhi 02.jpg` | `wood` | `wooden_plate` | `wooden_plate` | Niya Inscribed Wooden Wedge Plate 02 | Wooden wedge document obverse face. |
| **45** | `kharosthi` | `dataset/whole_inscriptions/kharosthi/Kharoshti script on a wooden plate, National Museum, New Delhi.jpg` | `wood` | `wooden_plate` | `wooden_plate` | Niya Rectangular Inscribed Wooden Plate | Rectangular timber plate charter. |
| **46** | `kharosthi` | `dataset/whole_inscriptions/kharosthi/Niya BLH170 OR821114231424 W.jpg` | `wood` | `wooden_tablet` | `wooden_document` | Niya Wooden Timber Record BLH170 | Inscribed timber communication record (British Library Stein collection). |
| **47** | `kharosthi` | `dataset/whole_inscriptions/kharosthi/Niya wooden Karosthi tablets.jpg` | `wood` | `wooden_tablet` | `wooden_tablets` | Niya Paired Double-Wooden Tablets | Paired timber tablets with clay seal socket. |
| **48** | `kharosthi` | `dataset/whole_inscriptions/kharosthi/Reliquary casket with Kharoshti characters, Bimran Stupa, housed at the British Museum London.jpg` | `steatite` | `reliquary_casket` | `steatite_reliquary` | Bimaran Stupa Steatite Reliquary Casket | Miniature carved steatite Buddhist relic bowl/casket (portable museum object). |

---

## 4. Rows Requiring Epigraphic & Provenance Review (4 Items)

Four rows represent authentic stone epigraphs but lack definitive site or monument provenance. Rather than inventing information or inferring labels from filenames alone, these records have been flagged with `audit_status = 'needs_review'` and `label_confidence = 'needs_review'`:

### 1. Row 03 — `dataset/whole_inscriptions/brahmi/024 Brahmi Inscription On-Site.jpg`
- **Script**: `brahmi` | **Partition**: `val`
- **Original Site Field**: `""` (Empty string in source CSV)
- **Original Notes**: *"On-site Brahmi rock inscription field capture; specific archaeological site unstated in filename."*
- **Audit Field Values**:
  - `physical_medium`: `stone`
  - `archaeological_site_normalized`: `unknown`
  - `physical_monument`: `unknown`
  - `audit_status`: `needs_review`
- **Manual Review Action Required**: Cross-reference the visual rock features, character style, and Wikimedia upload history to identify the exact rock edict or cavern findspot.

### 2. Row 14 — `dataset/whole_inscriptions/grantha/9th century Tamil Grantha script Sanskrit Chera era font.jpg`
- **Script**: `grantha` | **Partition**: `train`
- **Original Site Field**: `Chera Kingdom, Kerala / Tamil Nadu`
- **Original Notes**: *"9th-century Chera dynasty Sanskrit inscription in Grantha script."*
- **Audit Field Values**:
  - `physical_medium`: `stone`
  - `archaeological_site_normalized`: `unknown (regional: Chera Kingdom)`
  - `physical_monument`: `unknown`
  - `audit_status`: `needs_review`
- **Manual Review Action Required**: The filename suggests a typographic or style illustration; verify whether this is a photograph of an authentic in-situ Chera stone inscription or a secondary book engraving, and identify the specific monument.

### 3. Row 15 — `dataset/whole_inscriptions/grantha/Chola-q.jpg`
- **Script**: `grantha` | **Partition**: `train`
- **Original Site Field**: `Tamil Nadu` (Broad state name only)
- **Original Notes**: *"Chola era temple stone inscription featuring Grantha and Tamil characters."*
- **Audit Field Values**:
  - `physical_medium`: `stone`
  - `archaeological_site_normalized`: `unknown (regional: Tamil Nadu)`
  - `physical_monument`: `unknown`
  - `audit_status`: `needs_review`
- **Manual Review Action Required**: Image depicts temple ashlar masonry; locate the specific temple (e.g. Thanjavur, Gangaikondacholapuram, or Tribhuvanam) from the inscription's text lines to establish monument identity.

### 4. Row 33 — `dataset/whole_inscriptions/kadamba/Kadamba inscription from Kerala.jpg`
- **Script**: `kadamba` | **Partition**: `test`
- **Original Site Field**: `Northern Kerala / Karnataka border`
- **Original Notes**: *"Kadamba dynasty epigraph on stone slab."*
- **Audit Field Values**:
  - `physical_medium`: `stone`
  - `archaeological_site_normalized`: `unknown (regional: Northern Kerala / Karnataka border)`
  - `physical_monument`: `unknown`
  - `audit_status`: `needs_review`
- **Manual Review Action Required**: This single item is currently the entire `kadamba` test split! Epigraphical research is required to determine the exact temple or village findspot on the border.

---

## 5. Schema Specification for `dataset/metadata_stone_only.csv`

The newly created file [`dataset/metadata_stone_only.csv`](file:///c:/Users/hanvi/OneDrive/Desktop/Deepscript/Deepscript/dataset/metadata_stone_only.csv) contains 21 columns:

```
[ Original 11 Columns (Preserved 100% Unchanged) ]
1.  image_path                      : Path to image asset in repository
2.  script                          : Canonical script family label (brahmi, grantha, gupta, kadamba, kharosthi, tamil_brahmi)
3.  source                          : Data source provenance (Wikimedia Commons, Mendeley Data, etc.)
4.  source_url                      : Original persistent URL or citation
5.  site                            : Original archaeological site string
6.  inscription_type                : Structural typology (cave_inscription, stone_slab, pillar_edict, copper_plate, etc.)
7.  image_type                      : Image framing category (whole_inscription, field_photograph)
8.  license                         : Copyright and licensing terms (CC BY-SA 4.0, Public Domain, etc.)
9.  split                           : Original split allocation (train, val, test)
10. verified                        : Boolean verification flag from original intake
11. notes                           : Epigraphical descriptive notes

[ Additional 10 Audit Columns ]
12. physical_medium                 : Material substrate ('stone', 'copper', 'wood', 'terracotta', 'steatite', 'paper')
13. image_modality                  : Photographic representation ('field_photograph', 'museum_photograph', 'rubbing_facsimile')
14. is_stone                        : Boolean flag (True for monumental stone; False for metal, wood, clay, paper, reliquary)
15. non_stone_flag                  : Explicit defect flag ('none', 'paper_rubbing', 'copper_plate', 'wooden_tablet', 'wooden_plate', 'clay_seal', 'reliquary_casket')
16. authenticity_status             : Epigraphic authenticity ('authentic_ancient_epigraph', 'reproduction_rubbing', 'needs_review')
17. archaeological_site_normalized : Normalized findspot name or 'unknown' / 'unknown (regional: ...)'
18. physical_monument               : Specific physical monument identifier (e.g. 'Sittanavasal Stone Bed #1') or 'unknown'
19. label_confidence                : Ground truth confidence ('high', 'medium', 'needs_review')
20. audit_status                    : Overall audit classification ('verified_stone', 'flagged_non_stone', 'needs_review')
21. audit_notes                     : Technical rationale, monument linkage, or review instructions
```

---

## 6. Safety & Integrity Sign-Off

In accordance with strict audit requirements:
- **Production Code Unchanged**: `backend/vit_inference.py`, `models/`, and `preprocessing/` were not modified.
- **Original Metadata Unchanged**: [`dataset/metadata.csv`](file:///c:/Users/hanvi/OneDrive/Desktop/Deepscript/Deepscript/dataset/metadata.csv) has zero git diff.
- **Image Files Unchanged**: Zero images were moved, renamed, downloaded, preprocessed, or deleted.
- **Model Checkpoint Intact**:
  - Target: `checkpoints/six_class_vit/best_vit_model.pth`
  - Pre-audit SHA-256: `6efa33c8a11389f64b8fa5040a7de6c4cf8042fb5359889ca4a23f5653fdaa5c`
  - Post-audit SHA-256: `6efa33c8a11389f64b8fa5040a7de6c4cf8042fb5359889ca4a23f5653fdaa5c`
  - **Match Status: 100% Intact & Untouched**

# DeepScript Inscription Localization Prototype Report

## 1. Executive Summary & Experimental Disclaimer

This experiment implements and evaluates a **non-destructive inscription localization prototype** for DeepScript.

> [!WARNING]
> **EXPERIMENTAL PROTOTYPE — NOT AUTOMATIC LOCALIZATION**
> - All bounding boxes in this study were **manually specified** based on epigraphic inspection of the physical monument (Bed #1 at Ezhadippattam cavern, Sittanavasal).
> - **Automatic inscription localization has NOT been implemented or claimed.**
> - This prototype demonstrates what the downstream ViT-B/16 classifier perceives when provided an isolated inscription region versus an uncropped whole-monument photograph.

**Strict Preservation Guarantees Maintained:**
- **Active Model**: `checkpoints/six_class_vit/best_vit_model.pth` evaluated strictly in read-only mode (SHA-256: `6EFA33C8A11389F64B8FA5040A7DE6C4CF8042FB5359889CA4A23F5653FDAA5C`).
- **Zero Retraining**: No weights, projection matrices, or classifiers were trained or fine-tuned.
- **Dataset Integrity**: All original photographs, metadata annotations, and split definitions remain completely untouched.
- **Production Isolation**: This experiment is completely isolated under `experiments/inscription_localization_prototype/` and `results/inscription_localization_prototype/`. Production inference (`backend/vit_inference.py`) is unmodified.

---

## 2. Localization Pipeline Workflow

```
[Raw Photograph (e.g. 4000x3000, AR 1.33)]
                 │
                 ├──► [Manual Experimental Bounding Box: (x1, y1, x2, y2)]
                 │                    │
                 │                    ▼
                 │       [Extracted Inscription Crop (e.g. 3600x450, AR 8.0)]
                 │                    │
                 ▼                    ▼
     [Full Image Inference]   [Crop Image Inference]
     - Direct 224x224         - Direct 224x224
     - Aspect-Preserving      - Aspect-Preserving
       Tiled Aggregation        Tiled Aggregation (5 tiles)
                 │                    │
                 └──────────┬─────────┘
                            ▼
     [Comparative Evaluation & 3-Panel Visual Diagnostics]
```

For each specimen, the prototype generates a 3-panel visual composite:
1. **Whole Photograph**: Original image with the experimental bounding box highlighted in high-contrast red/magenta.
2. **Extracted Crop**: Isolated inscription ribbon displayed at its native aspect ratio.
3. **Model Input Tensor**: Direct Bicubic $224 \times 224$ squashed representation showing what the ViT backbone receives.

---

## 3. Benchmark Dataset & Bounding Box Specifications

All five benchmark photographs represent the Sangam-era Early Tamil-Brahmi class (`tamil_brahmi`). The four Sittanavasal photographs depict the same physical stone bed from varying angles, while `TBSI_029_original.jpg` serves as a chalked positive control:

| Image File | Original Dimensions | Aspect Ratio | Bounding Box $(x_1, y_1, x_2, y_2)$ | Crop Dimensions | Crop Aspect Ratio | Area % | Epigraphic Rationale & Notes |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`Sittanavasal_Rock_Bed_Inscription.jpg`** | $4608 \times 3456$ | $1.333$ | `(700, 1250, 4550, 1850)` | $3850 \times 600$ | $6.417$ | $14.5\%$ | Isolates the polished headrest ridge running across Bed #1. Excludes dark ceiling and lower bed body with modern graffiti. |
| **`Stone_bed_with_inscription_in_sittanavasal.JPG`** | $4000 \times 3000$ | $1.333$ | `(0, 700, 3600, 1150)` | $3600 \times 450$ | $8.000$ | $13.5\%$ | Higher-angle perspective of Bed #1 headrest. Inscription begins at frame edge (X=0). Excludes modern Tamil graffiti. |
| **`Chithannavasal_cave_inscription_1.JPG`** | $3648 \times 2736$ | $1.333$ | `(500, 360, 3100, 680)` | $2600 \times 320$ | $8.125$ | $8.3\%$ | Distant wide-angle showing two adjacent cavern beds. Cropped strictly to Bed #1 headrest. Significant distance blur. |
| **`Chithannavasal_cave_inscription_2.JPG`** | $3648 \times 2736$ | $1.333$ | `(300, 1450, 3500, 1850)` | $3200 \times 400$ | $8.000$ | $12.8\%$ | Upward-angled perspective of Bed #1 headrest. Upper cave overhang casts shadow over the left half of the inscription. |
| **`TBSI_029_original.jpg`** *(Positive Control)* | $2501 \times 1255$ | $1.993$ | `(100, 180, 2400, 750)` | $2300 \times 570$ | $4.035$ | $41.8\%$ | High-contrast chalk-highlighted Tamil-Brahmi baseline from Mendeley TBSI dataset. Tightens margin around inscribed band. |

---

## 4. Comprehensive Prediction & Score Comparison (All 6 Classes)

All inferences executed with `checkpoints/six_class_vit/best_vit_model.pth` ($s = 15.9930$).

### Summary Comparison Table
| Image File | Ground Truth | Full Image (Direct) | Full Image (Tiled) | Crop (Direct 224x) | Crop (Tiled Aggregated) | Prediction Shift Category |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Sittanavasal_Rock_Bed_Inscription.jpg`** | `tamil_brahmi` | **brahmi** (63.7%, cos 0.256) | **brahmi** (63.7%, cos 0.256) | **brahmi** (77.0%, cos 0.299) | **brahmi** (55.6%, cos 0.235) | `both_incorrect` (Brahmi dominant) |
| **`Stone_bed_with_inscription_in_sittanavasal.JPG`** | `tamil_brahmi` | **kharosthi** (81.7%, cos 0.197) | **kharosthi** (81.7%, cos 0.197) | **brahmi** (72.1%, cos 0.292) | **kharosthi** (70.1%, cos 0.194) | `both_incorrect` (Kharosthi $\to$ Brahmi) |
| **`Chithannavasal_cave_inscription_1.JPG`** | `tamil_brahmi` | **kharosthi** (99.7%, cos 0.400) | **kharosthi** (99.7%, cos 0.400) | **tamil_brahmi** (97.3%, cos 0.419) | **tamil_brahmi** (97.3%, cos 0.416) | **`incorrect_to_correct`** |
| **`Chithannavasal_cave_inscription_2.JPG`** | `tamil_brahmi` | **tamil_brahmi** (55.1%, cos 0.196) | **tamil_brahmi** (55.1%, cos 0.196) | **brahmi** (82.2%, cos 0.410) | **brahmi** (55.2%, cos 0.253) | **`correct_to_incorrect`** ⚠️ |
| **`TBSI_029_original.jpg`** *(Control)* | `tamil_brahmi` | **tamil_brahmi** (99.9%, cos 0.557) | **tamil_brahmi** (99.9%, cos 0.592) | **tamil_brahmi** (98.8%, cos 0.611) | **tamil_brahmi** (99.1%, cos 0.565) | `both_correct` (Robust control) |

---

## 5. Detailed Six-Class Probability and Cosine Metric Distributions

### Specimen 1: `Sittanavasal_Rock_Bed_Inscription.jpg`
- **Full Image (Direct 224x)**:
  - `brahmi`: **63.69%** (cos +0.2555) — *Winner*
  - `tamil_brahmi`: 32.71% (cos +0.2138) — *Runner-up*
  - `kharosthi`: 3.40% (cos +0.0722)
  - `grantha`: 0.17% (cos -0.1144)
  - `gupta`: 0.02% (cos -0.2378)
  - `kadamba`: 0.01% (cos -0.3343)
- **Crop Image (Direct 224x)**:
  - `brahmi`: **76.96%** (cos +0.2986) — *Winner*
  - `tamil_brahmi`: 22.25% (cos +0.2210) — *Runner-up*
  - `grantha`: 0.38% (cos -0.0337)
  - `gupta`: 0.36% (cos -0.0362)
  - `kharosthi`: 0.03% (cos -0.1909)
  - `kadamba`: 0.01% (cos -0.2460)
- **Crop Image (Tiled Aggregated, 5 tiles)**:
  - `brahmi`: **55.59%** (cos +0.2349) — *Winner*
  - `tamil_brahmi`: 42.50% (cos +0.2181) — *Runner-up*
  - `grantha`: 1.25% (cos -0.0026)
  - `kharosthi`: 0.44% (cos -0.0680)
  - `kadamba`: 0.12% (cos -0.1501)
  - `gupta`: 0.11% (cos -0.1532)

---

### Specimen 2: `Stone_bed_with_inscription_in_sittanavasal.JPG`
- **Full Image (Direct 224x)**:
  - `kharosthi`: **81.73%** (cos +0.1971) — *Winner*
  - `brahmi`: 13.39% (cos +0.0840) — *Runner-up*
  - `kadamba`: 3.47% (cos -0.0005)
  - `grantha`: 1.20% (cos -0.0667)
  - `tamil_brahmi`: 0.20% (cos -0.1801)
  - `gupta`: 0.01% (cos -0.3521)
- **Crop Image (Direct 224x)**:
  - `brahmi`: **72.12%** (cos +0.2923) — *Winner*
  - `tamil_brahmi`: 23.56% (cos +0.2224) — *Runner-up*
  - `gupta`: 2.21% (cos +0.0745)
  - `grantha`: 2.06% (cos +0.0701)
  - `kharosthi`: 0.04% (cos -0.1825)
  - `kadamba`: 0.00% (cos -0.3919)

---

### Specimen 3: `Chithannavasal_cave_inscription_1.JPG` (Success Case: False Background Purged)
- **Full Image (Direct 224x)**:
  - `kharosthi`: **99.72%** (cos +0.4003) — *Winner*
  - `brahmi`: 0.13% (cos -0.0151)
  - `kadamba`: 0.07% (cos -0.0537)
  - `grantha`: 0.06% (cos -0.0648)
  - `tamil_brahmi`: 0.02% (cos -0.1313)
  - `gupta`: 0.00% (cos -0.2800)
- **Crop Image (Direct 224x)**:
  - `tamil_brahmi`: **97.28%** (cos +0.4187) — *Winner (Correct)*
  - `gupta`: 1.98% (cos +0.1751)
  - `brahmi`: 0.70% (cos +0.1098)
  - `kharosthi`: 0.01% (cos -0.1310)
  - `grantha`: 0.03% (cos -0.0937)
  - `kadamba`: 0.00% (cos -0.2867)

---

### Specimen 4: `Chithannavasal_cave_inscription_2.JPG` (Critical Flaw: Correct $\to$ Incorrect Flip)
- **Full Image (Direct 224x)**:
  - `tamil_brahmi`: **55.06%** (cos +0.1959) — *Nominal Winner (Weak)*
  - `kharosthi`: 40.73% (cos +0.1771) — *Close Competitor*
  - `brahmi`: 1.97% (cos -0.0122)
  - `grantha`: 1.23% (cos -0.0417)
  - `kadamba`: 0.99% (cos -0.0552)
  - `gupta`: 0.01% (cos -0.3191)
- **Crop Image (Direct 224x)**:
  - `brahmi`: **82.16%** (cos +0.4101) — *Winner (Incorrect)*
  - `tamil_brahmi`: 17.75% (cos +0.3144) — *Runner-up*
  - `grantha`: 0.06% (cos -0.0396)
  - `kharosthi`: 0.02% (cos -0.1100)
  - `gupta`: 0.01% (cos -0.1839)
  - `kadamba`: 0.00% (cos -0.3441)
- **Crop Image (Tiled Aggregated, 5 tiles)**:
  - `brahmi`: **55.22%** (cos +0.2528) — *Winner (Incorrect)*
  - `tamil_brahmi`: 40.93% (cos +0.2341) — *Runner-up*
  - `kharosthi`: 3.42% (cos +0.0788)
  - `grantha`: 0.32% (cos -0.0684)
  - `gupta`: 0.11% (cos -0.1362)
  - `kadamba`: 0.01% (cos -0.3215)

---

### Specimen 5: `TBSI_029_original.jpg` (Baseline Positive Control)
- **Full Image (Direct 224x)**:
  - `tamil_brahmi`: **99.88%** (cos +0.5568) — *Winner*
  - `kharosthi`: 0.10% (cos +0.1247)
  - `brahmi`: 0.02% (cos +0.0157)
- **Crop Image (Direct 224x)**:
  - `tamil_brahmi`: **98.76%** (cos +0.6112) — *Winner*
  - `brahmi`: 1.24% (cos +0.3374)
- **Crop Image (Tiled Aggregated, 5 tiles)**:
  - `tamil_brahmi`: **99.13%** (cos +0.5653) — *Winner*
  - `brahmi`: 0.87% (cos +0.2689)

---

## 6. Critical Investigation: Why Cropping Changed a "Correct" Prediction to "Incorrect" on Inscription 2

A central finding of this prototype is the **`correct_to_incorrect` prediction flip** observed on `Chithannavasal_cave_inscription_2.JPG`:
- On the **whole image**, the active model predicted `tamil_brahmi` ($55.1\%$, cosine $+0.1959$).
- On the **cropped image**, the model predicted Ashokan `brahmi` ($82.2\%$, cosine $+0.4101$), while `tamil_brahmi` dropped to $17.8\%$ (cosine $+0.3144$).

### Why Did This Occur?
1. **The Full Image Prediction was a "Lucky Spurious Classification"**:
   - In the full photograph, the cosine similarity for `tamil_brahmi` was extremely low ($+0.1959$), well below the model's calibrated acceptance floor ($0.280$).
   - The prediction was neck-and-neck with `kharosthi` ($40.73\%$, cosine $+0.1771$), driven entirely by ambient rock contours, shadows, and cave walls. In production, this would trigger open-set rejection / abstention due to tiny classification margin ($\Delta = 0.0188 < 0.040$).
2. **Cropping Exposes the Deep Epigraphic Bias Toward Chalk Contrast**:
   - Isolating the headrest ribbon removes ambient clutter, allowing the ViT to focus entirely on the inscribed strokes.
   - Both `brahmi` and `tamil_brahmi` cosine similarities increase dramatically:
     - Ashokan `brahmi` cosine jumps from $-0.0122 \to +0.4101$ ($\Delta = +0.4223$).
     - `tamil_brahmi` cosine jumps from $+0.1959 \to +0.3144$ ($\Delta = +0.1185$).
   - However, because the authentic Sittanavasal inscription is **unchalked, shallowly incised granite**, its visual features align much more strongly with the unchalked stone edicts representing `brahmi` in the training set than with the high-contrast chalked images representing `tamil_brahmi` (Mendeley TBSI collection).
   - Consequently, **spatial localization revealed the true underlying palaeographic confusion** of the model: without chalk, Tamil-Brahmi incisions on natural rock are predominantly recognized as Ashokan Brahmi.

---

## 7. Conclusions & Next Steps

1. **Spatial Localization is Necessary but Insufficient**:
   - Cropping prevents catastrophic failures caused by background clutter (e.g. converting $99.7\%$ false `kharosthi` to $97.3\%$ true `tamil_brahmi` on Image 1).
   - However, cropping does **not** solve the training domain gap where `tamil_brahmi` is conflated with Ashokan `brahmi` on unchalked stone.
2. **Automation Boundary**:
   - Real-world archaeological photographs contain weathered rock textures, graffiti, and varying lighting that require robust contour or text-line detection algorithms before automated localization can be reliably deployed.

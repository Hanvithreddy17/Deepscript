# Sittanavasal Inscription Preprocessing & Crop Inference Experiment

## 1. Overview & Objective
This non-destructive preprocessing experiment investigates the impact of spatial framing and aspect ratio on model inference for the four Sittanavasal photographs in `dataset/whole_inscriptions/tamil_brahmi/` and the baseline `TBSI_029_original.jpg`.

The experiment evaluates whether isolating the inscribed headrest region of the physical stone bed prior to model preprocessing mitigates glyph feature dilution and distortion caused by standard 224x224 resizing.

**Strict Constraints Maintained:**
- **Active Model**: `checkpoints/six_class_vit/best_vit_model.pth` (read-only, SHA-256 verified untouched: `6EFA33C8A11389F64B8FA5040A7DE6C4CF8042FB5359889CA4A23F5653FDAA5C`).
- **Class Ordering**: `['brahmi', 'grantha', 'gupta', 'kadamba', 'kharosthi', 'tamil_brahmi']` (6 classes, index 0 to 5).
- **Scale Factor**: $s = 15.9930$ (learned cosine inverse temperature).
- **Zero Retraining**: No model parameters or projection weights were trained or fine-tuned.
- **Data Integrity**: Original image files, dataset metadata, and split definitions remain completely unmodified.
- **Monument Identity**: The four Sittanavasal photographs depict the same physical monument (Bed #1 at Ezhadippattam cavern, 1st c. BCE Early Tamil-Brahmi) and are not treated as independent generalization samples.

---

## 2. Preprocessing Architecture & Aggregation Mathematics

### Preprocessing Pipelines
1. **Standard Evaluation Transforms (`get_eval_transforms`)**:
   - `Resize((224, 224), interpolation=InterpolationMode.BICUBIC, antialias=True)`: Force-resizes input directly to square $224 \times 224$.
   - `ToTensor()`: Scales pixel intensities to $[0.0, 1.0]$.
   - `Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])`: ImageNet channel standardization.

2. **Aspect-Preserving Tiling (`extract_aspect_preserving_tiles`)**:
   - Trigger condition: $AR > 1.8$ (wide horizontal) or $AR < 0.55$ (tall vertical).
   - Tile extraction: Square crops of size $H \times H$ stepped across width with 25% overlap (up to `max_tiles=5`).
   - If $AR \in [0.55, 1.8]$: Returns `[image]` (no tiling triggered).

### Score Computation & Aggregation Mechanics
For an input with $K$ tiles and the whole image as index 0 (total $K+1$ tensors):
1. **Per-Tensor Forward Pass**:
   Each image tensor $k \in \{0, \dots, K\}$ is passed through ViT-B/16 and the 256-d bottleneck to produce normalized embedding $\mathbf{x}_{k, \text{norm}}$.
   With class weight vectors $\mathbf{w}_{c, \text{norm}}$ and scale $s = 15.9930$:
   $$\text{cosine}_{k, c} = \mathbf{x}_{k, \text{norm}} \cdot \mathbf{w}_{c, \text{norm}}, \quad \text{logit}_{k, c} = s \cdot \text{cosine}_{k, c}$$
2. **Tile Aggregation**:
   In Modes 2 and 4, logits are averaged across the batch inputs:
   $$\overline{\text{logit}}_c = \frac{1}{K+1} \sum_{k=0}^K \text{logit}_{k, c} = s \cdot \left( \frac{1}{K+1} \sum_{k=0}^K \text{cosine}_{k, c} \right) = s \cdot \overline{\text{cosine}}_c$$
   **Averaging logits is mathematically identical to averaging raw cosine similarities.** It is **not** an average of softmax probabilities.
3. **Reported Metrics**:
   - Cosine Similarity: $\cos_c = \frac{\overline{\text{logit}}_c}{s} = \overline{\text{cosine}}_c$
   - Softmax Probability: $p_c = \frac{\exp(\overline{\text{logit}}_c)}{\sum_j \exp(\overline{\text{logit}}_j)} = \frac{\exp(s \cdot \overline{\text{cosine}}_c)}{\sum_j \exp(s \cdot \overline{\text{cosine}}_j)}$
   - Class Prediction: $\hat{c} = \operatorname{argmax}_c (p_c) \equiv \operatorname{argmax}_c (\overline{\text{logit}}_c) \equiv \operatorname{argmax}_c (\overline{\text{cosine}}_c)$.
   The predicted class is strictly identical whether derived from probabilities, logits, or cosine similarities.

---

## 3. Crop Specifications, Dimensions, and Epigraphic Uncertainty Flags

All bounding boxes were established via high-resolution inspection of the original raw photographs. Because visual appearance alone does not guarantee historical epigraphic boundaries, all uncertain boundaries are explicitly recorded for manual epigraphic review:

| Image File | Native Resolution | Crop Box $(x_1, y_1, x_2, y_2)$ | Crop Dimensions $(W \times H)$ | Area % | Rationale | Uncertainty Flags (For Manual Epigraphic Review) |
| :--- | :---: | :---: | :---: | :---: | :--- | :--- |
| **`Sittanavasal_Rock_Bed_Inscription.jpg`** | $4608 \times 3456$ ($4:3$) | `(700, 1250, 4550, 1850)` | $3850 \times 600$ ($AR=6.42$) | 14.5% | Isolates the polished headrest ridge running across Bed #1 at Ezhadippattam. Excludes upper bedrock and lower bed body containing modern graffiti. | - Left boundary starts at X=700 where initial 'Eruminatu...' glyph appears faint.<br>- Right boundary ends at X=4550 where grooves blend into weathered granite.<br>- Glare along white quartz ridge creates localized highlight blowout. |
| **`Stone_bed_with_inscription_in_sittanavasal.JPG`** | $4000 \times 3000$ ($4:3$) | `(0, 700, 3600, 1150)` | $3600 \times 450$ ($AR=8.00$) | 13.5% | Isolates headrest ridge in a higher-angle, wider perspective. Excludes bed body containing modern graffiti ('M.S. ... N. Mariyappan') and upper ledge. | - Left edge is partially cut off at frame boundary (X=0).<br>- White piece of litter/paper present on right headrest ledge around X=3100-3300.<br>- Lower image sharpness compared to other captures (Laplacian variance 810 vs 2879). |
| **`Chithannavasal_cave_inscription_1.JPG`** | $3648 \times 2736$ ($4:3$) | `(500, 360, 3100, 680)` | $2600 \times 320$ ($AR=8.12$) | 8.3% | Isolates headrest of Bed #1 from an extreme wide-angle photo showing two adjacent beds. Excludes the second bed and cave floor. | - Extreme camera distance: glyphs are only ~20-30 px tall in native resolution.<br>- Severe resolution loss and motion/compression blur along inscription ridge.<br>- White paper litter visible on upper right headrest. |
| **`Chithannavasal_cave_inscription_2.JPG`** | $3648 \times 2736$ ($4:3$) | `(300, 1450, 3500, 1850)` | $3200 \times 400$ ($AR=8.00$) | 12.8% | Isolates headrest of Bed #1 from an upward-angled perspective. Excludes upper cave overhang and lower bed floor with graffiti. | - Perspective incline distortion due to upward camera tilt.<br>- Shadow cast from upper overhang partially darkens left side of ridge.<br>- Gradual contour transition between headrest ridge and bed floor. |
| **`TBSI_029_original.jpg`** | $2501 \times 1255$ ($1.99:1$) | `(100, 180, 2400, 750)` | $2300 \times 570$ ($AR=4.04$) | 41.8% | Tightens bounding box around chalk-highlighted epigraphic ribbon on rock face, removing non-inscribed upper rock margin and lower cave shadow. | - Already an epigraphically cropped baseline from Mendeley TBSI dataset.<br>- Faint chalk markings and weathered stroke tails extend slightly below Y=750.<br>- Serves as positive control demonstrating model behavior on high-contrast chalked script. |

---

## 4. Multi-Mode Empirical Comparison Results

The 4 evaluated preprocessing pathways:
- **Mode 1: Full Image (Direct)**: Whole image resized directly to $224 \times 224$ (1 tensor).
- **Mode 2: Full Image (Tiled)**: Aspect-preserving tiling on raw image (1 tile for $4:3$ images, 3 tiles for TBSI; stacked with whole image).
- **Mode 3: Crop (Direct 224x)**: Crop resized directly to $224 \times 224$ (1 tensor).
- **Mode 4: Crop (Tiled Aggregated)**: Aspect-preserving tiling on crop ($AR > 1.8 \implies 5$ tiles; stacked with whole crop = 6 tensors).

### Summary Comparison Table
| Image File | Ground Truth | Mode 1: Full Image (Direct) | Mode 2: Full Image (Tiled) | Mode 3: Crop (Direct 224x) | Mode 4: Crop (Tiled Aggregated) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`Sittanavasal_Rock_Bed_Inscription.jpg`** | `tamil_brahmi` | **brahmi** (63.7%, cos 0.256) | **brahmi** (63.7%, cos 0.256) | **brahmi** (77.0%, cos 0.299) | **brahmi** (55.6%, cos 0.235) |
| **`Stone_bed_with_inscription_in_sittanavasal.JPG`** | `tamil_brahmi` | **kharosthi** (81.7%, cos 0.197) | **kharosthi** (81.7%, cos 0.197) | **brahmi** (72.1%, cos 0.292) | **kharosthi** (70.1%, cos 0.194) |
| **`Chithannavasal_cave_inscription_1.JPG`** | `tamil_brahmi` | **kharosthi** (99.7%, cos 0.400) | **kharosthi** (99.7%, cos 0.400) | **tamil_brahmi** (97.3%, cos 0.419) | **tamil_brahmi** (97.3%, cos 0.416) |
| **`Chithannavasal_cave_inscription_2.JPG`** | `tamil_brahmi` | **tamil_brahmi** (55.1%, cos 0.196) | **tamil_brahmi** (55.1%, cos 0.196) | **brahmi** (82.2%, cos 0.410) | **brahmi** (55.2%, cos 0.253) |
| **`TBSI_029_original.jpg`** | `tamil_brahmi` | **tamil_brahmi** (99.9%, cos 0.557) | **tamil_brahmi** (99.9%, cos 0.592) | **tamil_brahmi** (98.8%, cos 0.611) | **tamil_brahmi** (99.1%, cos 0.565) |

---

## 5. Audit & Reconciliation of Specific Metrics

### Reconciliation of `Sittanavasal_Rock_Bed_Inscription.jpg`
A potential point of confusion was identified regarding two figures reported in analysis:
1. Direct crop winner: **`brahmi` (77.0%, cos 0.2986)**.
2. The claim that **`tamil_brahmi` probability rises from 22.2% to 42.5%** in tiled crop mode.

**Reconciliation from Raw JSON (`sittanavasal_crop_comparison_report.json`):**
- In **Mode 3 (Direct Crop)**:
  - **Winning class**: `brahmi` $\implies p = 0.7696$ (**77.0%**), $\cos = 0.2986$.
  - **Runner-up class**: `tamil_brahmi` $\implies p = 0.2225$ (**22.2%**), $\cos = 0.2210$.
- In **Mode 4 (Tiled Crop)**:
  - **Winning class**: `brahmi` $\implies p = 0.5559$ (**55.6%**), $\cos = 0.2349$.
  - **Runner-up class**: `tamil_brahmi` $\implies p = 0.4250$ (**42.5%**), $\cos = 0.2181$.

**Finding**: There is no mathematical or numerical inconsistency. The 77.0% figure is the **dominant winning probability** for `brahmi` in direct crop mode, while 22.2% is the **sub-dominant runner-up probability** for `tamil_brahmi` in direct crop mode. When tiled aggregation is applied, `brahmi` drops from 77.0% to 55.6%, while `tamil_brahmi` rises from 22.2% to 42.5%. **Crucially, the predicted class remains `brahmi` in both modes; it does not flip to `tamil_brahmi`.**

---

## 6. What the Experiment Demonstrates vs. What Remains Uncertain

### What the Experiment Actually Demonstrates:
1. **Whole-Image Squashing Distorts Standard Camera Captures**:
   - Because $4:3$ photographs ($AR = 1.333$) fall inside the standard range $[0.55, 1.8]$, the production pipeline never tiles whole images. The narrow inscription is crushed to ~2–5 pixels, causing ambient bedrock and cave features to dominate.
2. **Cropping Successfully Eliminates False Background Attributions**:
   - On `Chithannavasal_cave_inscription_1.JPG`, background floor clutter drove a 99.7% false `kharosthi` prediction. Cropping strictly to Bed #1's headrest shifted the prediction to the true class `tamil_brahmi` at 97.3% confidence (cosine +0.419).
3. **Metric Alignment on High-Contrast Inscriptions is Robust**:
   - On `TBSI_029_original.jpg` (positive control), high-contrast chalked script achieves $\ge 98.8\%$ `tamil_brahmi` and cosine similarities $> 0.55$ across all four modes.

### What Remains Uncertain:
1. **Unchalked Granite vs. Ashokan Brahmi Conflation**:
   - On authentic unchalked Tamil-Brahmi (`Sittanavasal_Rock_Bed_Inscription.jpg` and `Chithannavasal_cave_inscription_2.JPG`), the active model assigns higher cosine similarity to Ashokan `brahmi` (+0.299 and +0.410) than to `tamil_brahmi` (+0.221 and +0.314). The model appears to rely heavily on chalk-filled stroke contrast to differentiate `tamil_brahmi` from Ashokan `brahmi`.
2. **Exact Epigraphic Boundaries**:
   - In three of the four Sittanavasal captures, inscription boundaries fade into natural granite weathering, glare, or modern graffiti. Definitive epigraphic boundaries cannot be established by automated visual inspection alone and require in-situ epigraphic verification.
3. **No Generalization Claim**:
   - Because all four Sittanavasal photographs depict the exact same physical stone bed from different angles, their results represent repeated observations of a single monument, not independent samples of model generalization.

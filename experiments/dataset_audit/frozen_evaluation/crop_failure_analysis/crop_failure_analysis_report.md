# DeepScript — Controlled Failure Analysis: Inscription Cropping vs Full Image

**Experiment Date**: October 21, 2026  
**Target Checkpoint**: `checkpoints/six_class_vit/best_vit_model.pth`  
**Architecture**: ViT-B/16 + MLP Projector (768 $\to$ 512 $\to$ 256) + Learned Cosine Metric Head ($s = 15.9930$)  
**Directory**: `experiments/dataset_audit/frozen_evaluation/crop_failure_analysis/`

---

## 1. Executive Summary

This diagnostic investigation evaluates whether isolating visible carved inscription strokes via cropping resolves the five misclassifications observed in the zero-leakage frozen evaluation benchmark.

### Core Findings
1. **Cropping Successfully Resolved 3 of 5 Failure Cases**:
   - `SET_A_03` (*Sittanavasal Cavern View*): Full-image false `kharosthi` ($40.0\%$, cosine $+0.4003$) was driven entirely by non-inscribed cave floor clutter. Tight cropping flipped prediction to true positive **`tamil_brahmi`** ($41.6\%$, cosine $+0.4161$, a **$+0.5474$ cosine shift**).
   - `SET_B_01` (*TBSI_075 Sittanavasal Archive Photograph*): Full-image false `brahmi` ($35.5\%$, cosine $+0.3554$) was corrected to true positive **`tamil_brahmi`** ($27.4\%$, cosine $+0.2738$) when cropped tightly to the inscribed line.
   - `SET_B_08` (*TBSI_085 Varichiyur Drip Ledge Inscription #1*): Full-image borderline false `brahmi` ($33.9\%$, cosine $+0.3392$ vs. $+0.3358$) was corrected to true positive **`tamil_brahmi`** in both tight ($43.3\%$, cosine $+0.4326$) and context ($45.4\%$, cosine $+0.4538$) framings.
2. **Cropping Failed to Resolve 2 of 5 Failure Cases**:
   - `SET_A_01` (*Sittanavasal Frontal Bed View*): Remained misclassified as `brahmi` across all framings (cosine $+0.2349$ vs $+0.2181$ for Tamil-Brahmi). Unchalked granite texture and identical consonant glyph shapes between Early Tamil-Brahmi and Ashokan Brahmi prevent a class flip.
   - `SET_A_02` (*Sittanavasal High-Angle Wide View*): Remained misclassified (`kharosthi` in tight crop, `brahmi` in context crop) due to severe capture distance and low chisel contrast.
3. **Critical Discovery: Cropping Induces Regressions on Controls**:
   - On Positive Control `SET_A_04` (*Sittanavasal Close-Up Relief*), the uncropped full image was correctly identified as **`tamil_brahmi`** ($19.6\%$). When cropped, the prediction **regressed to `brahmi`** ($25.3\%$ in tight, $26.7\%$ in context).
   - On Positive Control `SET_B_02` (*TBSI_088 Vikkiramangalam Drip Ledge*), the model remained stably correct across all crops ($57.5\%-58.6\%$).

### Key Takeaway
Cropping is **not a universal remedy**. While it eliminates false background attributions in wide-angle cavern scenes, it can simultaneously strip vital macro-context and cause regressions on authentic unchalked rock faces where the model's cosine metric is finely balanced between Early Tamil-Brahmi and Ashokan Brahmi.

---

## 2. Checkpoint Integrity Verification

In strict accordance with audit safety protocols, the active model weights were cryptographically verified before and after the experiment.

| Verification Step | Timestamp | Checkpoint Path | SHA-256 Checksum | Integrity Status |
| :--- | :--- | :--- | :--- | :--- |
| **Pre-Experiment** | `2026-10-21 19:59:01` | `checkpoints/six_class_vit/best_vit_model.pth` | `6efa33c8a11389f64b8fa5040a7de6c4cf8042fb5359889ca4a23f5653fdaa5c` | **MATCH (100% Intact)** |
| **Post-Experiment** | `2026-10-21 20:01:58` | `checkpoints/six_class_vit/best_vit_model.pth` | `6efa33c8a11389f64b8fa5040a7de6c4cf8042fb5359889ca4a23f5653fdaa5c` | **MATCH (100% Intact)** |

- **Weight Updates**: $0$ (Frozen checkpoint, eval mode).
- **Production Pipeline Modifications**: $0$ (Standard `backend.vit_inference` used unchanged).
- **Dataset Cleanliness**: $0$ images added or moved into training, validation, or production dataset folders. Archive images were loaded in-memory from `Tamil Brahmi Stone Inscription.zip`.

---

## 3. Side-by-Side Empirical Results Matrix (21 Conditions)

All 7 test specimens (5 failure cases + 2 controls) were evaluated across three framing conditions:
1. **Original Full**: Uncropped original image as ingested by the baseline frozen evaluation.
2. **Crop Tight**: Tight crop bounding box isolating visible carved stroke relief.
3. **Crop Context**: Intermediate crop preserving surrounding stone texture and bed ledge contours.

| Specimen ID | Framing Condition | Dimensions ($W \times H$) | AR | Predicted Class | Conf. | Cosine TB | Cosine Brahmi | Cosine Kharosthi | Outcome vs. Baseline |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`SET_A_01`** *(Failure)* | Original Full | $4608 \times 3456$ | 1.33 | `brahmi` | 25.6% | +0.2138 | **+0.2555** | +0.0722 | **BASELINE FAIL** |
| Sittanavasal Bed #1 | Crop Tight | $3850 \times 600$ | 6.42 | `brahmi` | 23.5% | +0.2181 | **+0.2349** | -0.0680 | UNCHANGED (FAIL) |
| Frontal Bed View | Crop Context | $4050 \times 1100$ | 3.68 | `brahmi` | 42.8% | -0.0624 | **+0.4275** | -0.1524 | UNCHANGED (FAIL) |
| **`SET_A_02`** *(Failure)* | Original Full | $4000 \times 3000$ | 1.33 | `kharosthi` | 19.7% | -0.1801 | +0.0840 | **+0.1971** | **BASELINE FAIL** |
| Sittanavasal Bed #1 | Crop Tight | $3600 \times 450$ | 8.00 | `kharosthi` | 19.4% | +0.1082 | +0.0594 | **+0.1937** | UNCHANGED (FAIL) |
| High-Angle Bed View | Crop Context | $3800 \times 900$ | 4.22 | `brahmi` | 28.7% | +0.1210 | **+0.2866** | +0.0606 | UNCHANGED (FAIL) |
| **`SET_A_03`** *(Failure)* | Original Full | $3648 \times 2736$ | 1.33 | `kharosthi` | 40.0% | -0.1313 | -0.0151 | **+0.4003** | **BASELINE FAIL** |
| Sittanavasal Bed #1 | Crop Tight | $2600 \times 320$ | 8.12 | **`tamil_brahmi`** | **41.6%** | **+0.4161** | +0.1902 | -0.0739 | **IMPROVED (FIXED)** |
| Cavern Wide View | Crop Context | $3000 \times 600$ | 5.00 | **`tamil_brahmi`** | **24.7%** | **+0.2468** | +0.2325 | +0.1458 | **IMPROVED (FIXED)** |
| **`SET_B_01`** *(Failure)* | Original Full | $2497 \times 1011$ | 2.47 | `brahmi` | 35.5% | -0.0630 | **+0.3554** | +0.0509 | **BASELINE FAIL** |
| Sittanavasal Bed #1 | Crop Tight | $2247 \times 500$ | 4.49 | **`tamil_brahmi`** | **27.4%** | **+0.2738** | +0.2137 | -0.0676 | **IMPROVED (FIXED)** |
| Archive Photo (`TBSI_075`) | Crop Context | $2497 \times 750$ | 3.33 | `brahmi` | 41.3% | -0.0143 | **+0.4130** | -0.0166 | UNCHANGED (FAIL) |
| **`SET_B_08`** *(Failure)* | Original Full | $2500 \times 911$ | 2.74 | `brahmi` | 33.9% | +0.3358 | **+0.3392** | +0.0210 | **BASELINE FAIL** |
| Varichiyur Drip Ledge #1 | Crop Tight | $2250 \times 500$ | 4.50 | **`tamil_brahmi`** | **43.3%** | **+0.4326** | +0.2993 | -0.0161 | **IMPROVED (FIXED)** |
| Archive Photo (`TBSI_085`) | Crop Context | $2500 \times 600$ | 4.17 | **`tamil_brahmi`** | **45.4%** | **+0.4538** | +0.3155 | +0.0131 | **IMPROVED (FIXED)** |
| **`SET_A_04`** *(Control)* | Original Full | $3648 \times 2736$ | 1.33 | **`tamil_brahmi`** | **19.6%** | **+0.1959** | -0.0122 | +0.1771 | **BASELINE PASS** |
| Sittanavasal Bed #1 | Crop Tight | $3200 \times 400$ | 8.00 | `brahmi` | 25.3% | +0.2341 | **+0.2528** | +0.0788 | **REGRESSED (BROKEN)** |
| Close-Up Stroke Relief | Crop Context | $3350 \times 900$ | 3.72 | `brahmi` | 26.7% | +0.0582 | **+0.2669** | +0.0924 | **REGRESSED (BROKEN)** |
| **`SET_B_02`** *(Control)* | Original Full | $2500 \times 828$ | 3.02 | **`tamil_brahmi`** | **58.6%** | **+0.5858** | +0.3117 | -0.0809 | **BASELINE PASS** |
| Vikkiramangalam Ledge #1 | Crop Tight | $2250 \times 500$ | 4.50 | **`tamil_brahmi`** | **57.5%** | **+0.5752** | +0.2169 | -0.0207 | MAINTAINED (PASS) |
| Archive Photo (`TBSI_088`) | Crop Context | $2500 \times 550$ | 4.55 | **`tamil_brahmi`** | **56.8%** | **+0.5681** | +0.2204 | -0.0061 | MAINTAINED (PASS) |

---

## 4. Detailed Analysis by Failure Mechanism

### Category A: Background Attributions Successfully Eliminated by Cropping
#### 1. `SET_A_03` (*Chithannavasal_cave_inscription_1.JPG*)
- **Baseline Failure**: In the $4:3$ uncropped image, the actual inscription occupies less than $5\%$ of the total pixels. Over $80\%$ of the image consists of flat uncarved granite cave floor and uncarved bed ledge. The ViT patch embeddings of the floor texture produced strong spurious correlation with `kharosthi` ($\cos = +0.4003$, $40.0\%$ confidence).
- **Crop Effect**: Isolating the headrest ridge ($2600 \times 320$, $AR = 8.12$) activates aspect-preserving sliding window tiling (5 tiles). The ViT tokens directly process the carved character sequence *Eruminatu kumul-ur...*, flipping the prediction to true positive **`tamil_brahmi`** ($41.6\%$ confidence, $\cos = +0.4161$, runner-up Brahmi at $+0.1902$).
- **Conclusion**: Clear win for spatial localization.

#### 2. `SET_B_01` (*TBSI_075_original.jpg*)
- **Baseline Failure**: Archival scan containing heavy upper dark bedrock and modern Tamil graffiti below the bed ridge. Full image was confused with Ashokan `brahmi` ($35.5\%$, $\cos = +0.3554$).
- **Crop Effect**: Cropping tightly ($2247 \times 500$, $AR = 4.49$) removes the modern graffiti and upper dark void, flipping the prediction to true positive **`tamil_brahmi`** ($27.4\%$, $\cos = +0.2738$ vs Brahmi $+0.2137$).

#### 3. `SET_B_08` (*TBSI_085_original.jpg*)
- **Baseline Failure**: In the baseline evaluation, this specimen narrowly missed true positive classification by a minuscule cosine margin of **$0.0034$** (Brahmi $+0.3392$ vs Tamil-Brahmi $+0.3358$).
- **Crop Effect**: Removing the shadowed bottom rock wall increases Tamil-Brahmi cosine to $+0.4326$ (tight) and $+0.4538$ (context), expanding the winning margin over Brahmi to $+0.1383$.

---

### Category B: Persistent Failures Resistant to Cropping
#### 1. `SET_A_01` (*Sittanavasal_Rock_Bed_Inscription.jpg*)
- **Behavior**: The tight crop isolates the entire inscribed ridge ($3850 \times 600$, $AR = 6.42$), yielding a competitive Tamil-Brahmi cosine of $+0.2181$. However, Brahmi remains slightly higher at $+0.2349$ ($23.5\%$ confidence).
- **Root Cause**: Unlike the Mendeley TBSI training images (which feature high-contrast chalk-highlighted strokes on dark rock), `SET_A_01` is an authentic, unchalked daylight photograph. The natural weathered granite chisel marks share identical geometric primitives with Ashokan Brahmi consonants. The model requires chalk-filled contrast to clearly separate the two closely related scripts on this specific capture.

#### 2. `SET_A_02` (*Stone_bed_with_inscription_in_sittanavasal.JPG*)
- **Behavior**: High-angle camera perspective taken from a distance. The tight crop ($3600 \times 450$) increases Tamil-Brahmi cosine from negative ($-0.1801$) to positive ($+0.1082$), but Kharosthi still edges it out ($+0.1937$).
- **Root Cause**: The native stroke height is only ~15–20 pixels, and motion/compression blur softens the chisel edges. Without sharp stroke relief, patch embeddings remain ambiguous.

---

### Category C: Regressions Induced on Previously Correct Captures
#### 1. `SET_A_04` (*Chithannavasal_cave_inscription_2.JPG*) — **CRITICAL REGRESSION**
- **Baseline Performance**: In the uncropped full image, the camera perspective was already an upward raking close-up. The model correctly classified it as **`tamil_brahmi`** ($19.6\%$, $\cos = +0.1959$, Brahmi negative at $-0.0122$).
- **Crop Effect**: Applying the tight crop ($3200 \times 400$, $AR = 8.00$) triggered horizontal sliding-window tiling. While Tamil-Brahmi cosine rose to $+0.2341$, **Brahmi cosine jumped even faster to $+0.2528$**, flipping the prediction to `brahmi` ($25.3\%$). In the context crop, Brahmi won by $+0.2669$ vs $+0.0582$.
- **Significance**: This empirical regression demonstrates that forced horizontal cropping and tiling can inadvertently boost Ashokan Brahmi activations on unchalked stone relief by truncating the surrounding vertical rock framing that previously anchored the prediction.

---

## 5. Contact Sheet Visualization

The visual contact sheet showing the original photographs (with cyan tight crop boxes and yellow context crop boxes overlaid) alongside the tested crops and prediction outcomes is preserved at:
[`crop_contact_sheet.png`](file:///c:/Users/hanvi/OneDrive/Desktop/Deepscript/Deepscript/experiments/dataset_audit/frozen_evaluation/crop_failure_analysis/crop_contact_sheet.png)

```
+----------------------------------------------------------------------------------------------------+
| Visual Contact Sheet Overview (crop_contact_sheet.png)                                             |
+----------------------------------------------------------------------------------------------------+
| Row 1 [SET_A_01]: Full: FAIL (brahmi)       | Tight: FAIL (brahmi)       | Context: FAIL (brahmi)   |
| Row 2 [SET_A_02]: Full: FAIL (kharosthi)    | Tight: FAIL (kharosthi)    | Context: FAIL (brahmi)   |
| Row 3 [SET_A_03]: Full: FAIL (kharosthi)    | Tight: PASS (tamil_brahmi) | Context: PASS (TB)       |
| Row 4 [SET_B_01]: Full: FAIL (brahmi)       | Tight: PASS (tamil_brahmi) | Context: FAIL (brahmi)   |
| Row 5 [SET_B_08]: Full: FAIL (brahmi)       | Tight: PASS (tamil_brahmi) | Context: PASS (TB)       |
| Row 6 [SET_A_04]: Full: PASS (tamil_brahmi) | Tight: FAIL (REGRESSION)   | Context: FAIL (REGRESS)  |
| Row 7 [SET_B_02]: Full: PASS (tamil_brahmi) | Tight: PASS (tamil_brahmi) | Context: PASS (TB)       |
+----------------------------------------------------------------------------------------------------+
```

---

## 6. Epigraphic & Technical Limitations

1. **No Generalization Claims from Small Sample Sets**:
   This experiment evaluated 5 failure cases and 2 controls (representing 3 distinct archaeological sites: Sittanavasal, Varichiyur, and Vikkiramangalam). The fact that cropping fixed 3 failures cannot be extrapolated as proof that cropping will improve whole-inscription classification across all scripts or sites.
2. **Double-Edged Sword of Tiling**:
   When aspect-preserving tiling is applied to tight crops ($AR > 1.8$), individual square patches zoom into small glyph sequences. For Early Tamil-Brahmi, individual characters (such as *ma*, *ta*, *ya*, *ra*, *ka*) are morphologically indistinguishable from Northern Brahmi characters. The distinguishing orthographic features of Tamil-Brahmi (such as the *pulli* virama dot and specialized Dravidian letters *zha*, *rra*, *lla*, *nna*) occur only sparsely. Tiled crops that isolate sub-segments lacking these diagnostic letters are highly vulnerable to Ashokan Brahmi misclassification.
3. **Automated Bounding Box Dependency**:
   All bounding boxes in this diagnostic were manually reviewed to isolate visible carved relief. In a fully automated production pipeline, relying on an imperfect automated text detector could produce noisy crops that cut characters in half, inducing severe misclassifications.

---

## 7. Conclusions & Recommendation on Follow-Up Experiments

### Summary of What the Evidence Shows
- **Did cropping help?** Yes, on wide-angle images where uncarved cave floor/void dominated the pixel budget (`SET_A_03`, `SET_B_01`, `SET_B_08`).
- **Did cropping hurt?** Yes, on close-up stone relief (`SET_A_04`), where it caused an outright regression from correct Tamil-Brahmi to Brahmi.
- **Did cropping fix unchalked stone confusion?** No (`SET_A_01`, `SET_A_02`).

### Recommendation
**Do NOT deploy an automatic cropping heuristic to the production pipeline.**  
Blanket cropping risks breaking currently correct whole-stone classifications (as proven by `SET_A_04`).

**Evidence Justifies a Separate, Controlled Follow-Up Experiment**:
The results provide strong empirical justification for a dedicated, modular follow-up study:
- **Title**: *Dual-Stream Inference Prototype: Whole-Stone Framing + Inscription Crop Attention*
- **Scope**: Rather than replacing whole-image inference with cropping, evaluate a dual-stream ensemble where the model receives both the uncropped context and the localized text ribbon, combining their cosine embeddings to preserve global stone texture while amplifying local stroke features.
- **Requirement**: Must evaluate on a balanced multi-script test set, not a single-class sample.

---

## 8. Artifact Inventory

All experimental artifacts and data records have been saved under `experiments/dataset_audit/frozen_evaluation/crop_failure_analysis/`:
- **Evaluation Script**: [`evaluate_crop_experiment.py`](file:///c:/Users/hanvi/OneDrive/Desktop/Deepscript/Deepscript/experiments/dataset_audit/frozen_evaluation/crop_failure_analysis/evaluate_crop_experiment.py)
- **Predictions CSV (21 rows)**: [`crop_evaluation_results.csv`](file:///c:/Users/hanvi/OneDrive/Desktop/Deepscript/Deepscript/experiments/dataset_audit/frozen_evaluation/crop_failure_analysis/crop_evaluation_results.csv)
- **Summary JSON**: [`crop_evaluation_summary.json`](file:///c:/Users/hanvi/OneDrive/Desktop/Deepscript/Deepscript/experiments/dataset_audit/frozen_evaluation/crop_failure_analysis/crop_evaluation_summary.json)
- **Visual Contact Sheet (6.95 MB PNG)**: [`crop_contact_sheet.png`](file:///c:/Users/hanvi/OneDrive/Desktop/Deepscript/Deepscript/experiments/dataset_audit/frozen_evaluation/crop_failure_analysis/crop_contact_sheet.png)
- **Extracted Crops & Annotations (35 files)**: [`crops/`](file:///c:/Users/hanvi/OneDrive/Desktop/Deepscript/Deepscript/experiments/dataset_audit/frozen_evaluation/crop_failure_analysis/crops/)

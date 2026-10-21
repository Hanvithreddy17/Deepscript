# DeepScript Automatic Inscription Localization Prototype: Audit & Evaluation Report

## 1. Executive Summary & Objective

This study investigates whether a **classical computer vision approach** (bilateral filtering, CLAHE, morphological blackhat/tophat transforms, and horizontal ribbon grouping) can automatically propose candidate inscription regions on whole stone monument photographs prior to neural feature extraction.

> [!CAUTION]
> **DEFINITIVE EXPERIMENTAL VERDICT: NOT FEASIBLE WITH CLASSICAL CV**
> - Classical computer vision **CANNOT reliably localize ancient unchalked stone inscriptions**.
> - On weathered granite monuments, high-contrast ambient distractors (deeply scratched modern graffiti, rock fracture edges, and cavern shadow transitions) **consistently produce higher gradient energy and ribbon continuity than 2,100-year-old shallow, weathered petroglyphic glyphs**.
> - In **4 out of 5 benchmark photographs**, the top-ranked algorithmic proposal completely missed the inscription ($\text{IoU} = 0.00$ to $0.15$), instead capturing cave floor fissures, ceiling shadows, or modern graffiti.
> - **Critical Safety Finding**: Downstream ViT-B/16 predicted `tamil_brahmi` with **$>99.8\%$ confidence on non-inscribed cave floor cracks and modern graffiti**, demonstrating that unvalidated automatic bounding boxes cause severe "garbage-in, confident-out" misattributions.
> - **Recommendation**: Classical CV does **NOT merit further investigation** for automatic inscription localization on natural rock monuments. It must NOT be deployed to production.

---

## 2. Safety & Preservation Verification

- **Active Checkpoint**: `checkpoints/six_class_vit/best_vit_model.pth`
  - **Pre-Test SHA-256**: `6EFA33C8A11389F64B8FA5040A7DE6C4CF8042FB5359889CA4A23F5653FDAA5C`
  - **Post-Test SHA-256**: `6EFA33C8A11389F64B8FA5040A7DE6C4CF8042FB5359889CA4A23F5653FDAA5C` (Verified identical bit-for-bit).
- **Production Files**: `backend/vit_inference.py`, `models/`, `training/` are **100% untouched**.
- **Dataset Safety**: Manifests, image files, labels, and splits remain unmodified.
- **Zero Retraining**: No model parameters or projection matrices were trained or fine-tuned.

---

## 3. Classical Computer Vision Localization Architecture

The prototype in `experiments/automatic_localization_prototype/auto_localize_and_infer.py` operates as follows without using ground-truth labels:

1. **Resolution Normalization**: Input image scaled to canonical width $W_{\text{canon}} = 1200$ to ensure scale-invariant morphological kernel behavior.
2. **Granite Grain Suppression**: Bilateral filter ($d=7, \sigma_c=50, \sigma_s=50$) smooths high-frequency quartz/feldspar crystal textures while preserving edge boundaries.
3. **Dual Stroke Extraction**:
   - **Blackhat Transform** ($15 \times 3$ rect kernel) isolates dark incised grooves on lighter weathered granite.
   - **Tophat Transform** ($15 \times 3$ rect kernel) isolates bright/chalked strokes on dark stone.
   - Combined stroke magnitude: $\mathbf{M}_{\text{stroke}} = \max(\mathbf{M}_{\text{blackhat}}, \mathbf{M}_{\text{tophat}})$.
4. **Edge Fusion**: Canny edge detection ($T_{\text{low}}=30, T_{\text{high}}=90$) combined with stroke response via bitwise OR.
5. **Horizontal Ribbon Closing**: Rectangular horizontal closing ($45 \times 7$) and bridging dilation ($35 \times 5$) connects adjacent character strokes into continuous horizontal text bands.
6. **Contour Filtering & Ranking**: Contours filtered by horizontal aspect ratio ($AR \ge 1.4$), width ($W \ge 0.15 \times W_{\text{image}}$), and reasonable area ($0.5\% \le \text{Area} \le 50\%$). Candidates scored by mean stroke density and horizontal length bonus.
7. **Non-Maximum Suppression (NMS)**: Overlapping proposals pruned at $\text{IoU} \ge 0.30$, producing Top-K ranked candidate bounding boxes.

---

## 4. Empirical Benchmark Evaluation: Spatial Localization Quality vs. Classification

The benchmark was executed on the four Sittanavasal photographs and the `TBSI_029_original.jpg` chalked control.

### Spatial Quality Definitions
- **IoU**: $\frac{\text{Area}(\text{Cand} \cap \text{Manual Ref})}{\text{Area}(\text{Cand} \cup \text{Manual Ref})}$
- **Recall (Coverage)**: $\frac{\text{Area}(\text{Cand} \cap \text{Manual Ref})}{\text{Area}(\text{Manual Ref})}$ (Fraction of ancient inscription captured)
- **Precision**: $\frac{\text{Area}(\text{Cand} \cap \text{Manual Ref})}{\text{Area}(\text{Cand})}$ (Fraction of candidate crop that is genuine inscription)

### Summary Comparison Table
| Image File | Ground Truth | Manual Reference Inscription Box | Top-1 Auto Proposed Box | Spatial IoU | Inscription Recall | Precision | Top-1 Physical Content Assessment | Full Image Pred | Top-1 Crop Pred | Manual Crop Pred |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :--- | :--- |
| **`Sittanavasal_Rock_Bed_Inscription.jpg`** | `tamil_brahmi` | `(700, 1250, 4550, 1850)` | `(0, 0, 4608, 1582)` | **`0.1536`** | 55.3% | 17.5% | **Cave ceiling + upper headrest edge** | `brahmi` (63.7%) | `grantha` (37.1%) | `brahmi` (77.0%) |
| **`Stone_bed_with_inscription_in_sittanavasal.JPG`** | `tamil_brahmi` | `(0, 700, 3600, 1150)` | `(0, 2746, 3526, 3000)` | **`0.0000`** | 0.0% | 0.0% | **Modern scratched graffiti on bed floor** | `kharosthi` (81.7%) | `tamil_brahmi` (99.9%) ⚠️ | `brahmi` (72.1%) |
| **`Chithannavasal_cave_inscription_1.JPG`** | `tamil_brahmi` | `(500, 360, 3100, 680)` | `(30, 2243, 3647, 2736)` | **`0.0000`** | 0.0% | 0.0% | **Cavern floor rock fissure / crack** | `kharosthi` (99.7%) | `tamil_brahmi` (99.8%) ⚠️ | `tamil_brahmi` (97.3%) |
| **`Chithannavasal_cave_inscription_2.JPG`** | `tamil_brahmi` | `(300, 1450, 3500, 1850)` | `(1298, 1325, 2474, 1413)` | **`0.0000`** | 0.0% | 0.0% | **Overhang rock shadow boundary** | `tamil_brahmi` (55.1%) | `brahmi` (87.9%) | `brahmi` (82.2%) |
| **`TBSI_029_original.jpg`** *(Control)* | `tamil_brahmi` | `(100, 180, 2400, 750)` | `(1786, 354, 2186, 625)` | **`0.0827`** | 8.3% | 100.0% | **Fragmentary 2-glyph snippet of chalked line** | `tamil_brahmi` (99.9%) | `tamil_brahmi` (86.3%) | `tamil_brahmi` (98.8%) |

---

## 5. Detailed Case-by-Case Analysis

### Case 1: `Sittanavasal_Rock_Bed_Inscription.jpg`
- **Manual Reference**: Headrest band `(700, 1250, 4550, 1850)`.
- **Top-1 Auto Proposal**: `(0, 0, 4608, 1582)` ($\text{IoU} = 0.1536$).
- **Failure Mode**: The massive contrast step between the dark cave ceiling and the daylit stone bed completely dominated the morphological closing filter. The proposed box encompasses the entire top third of the frame, capturing mostly dark blank rock and only shaving the top half of the headrest.
- **Classifier Impact**: ViT-B/16 received mostly dark ceiling rock, causing predictions to scatter across classes: `grantha` (37.1%), `tamil_brahmi` (36.5%), `kharosthi` (12.8%), `brahmi` (8.3%).

### Case 2: `Stone_bed_with_inscription_in_sittanavasal.JPG` (Alarming Hallucination)
- **Manual Reference**: Headrest band `(0, 700, 3600, 1150)`.
- **Top-1 Auto Proposal**: `(0, 2746, 3526, 3000)` ($\text{IoU} = 0.0000$, $\text{Recall} = 0.0\%$).
- **Failure Mode**: The algorithm completely bypassed the 2,100-year-old Tamil-Brahmi inscription at the top and instead locked onto the **modern scratched graffiti** ("N. Mariyappan") along the bottom of the bed. Modern scratches are sharp, white, and unweathered, generating higher gradient density than ancient incisions.
- **Classifier Impact**: Despite containing **zero ancient glyphs**, ViT-B/16 predicted **`tamil_brahmi` with $99.996\%$ confidence and $+0.7136$ cosine similarity!** This proves that high-frequency scratched lines on stone fool the model's projection head into hallucinating Tamil-Brahmi.

### Case 3: `Chithannavasal_cave_inscription_1.JPG` (Floor Crack Hallucination)
- **Manual Reference**: Upper headrest `(500, 360, 3100, 680)`.
- **Top-1 Auto Proposal**: `(30, 2243, 3647, 2736)` ($\text{IoU} = 0.0000$, $\text{Recall} = 0.0\%$).
- **Failure Mode**: The algorithm selected a horizontal geological fissure across the cave floor at the bottom of the frame.
- **Classifier Impact**: ViT-B/16 predicted **`tamil_brahmi` with $99.8\%$ confidence and $+0.6363$ cosine similarity** on natural rock fractures!

### Case 4: `Chithannavasal_cave_inscription_2.JPG`
- **Manual Reference**: Raised headrest `(300, 1450, 3500, 1850)`.
- **Top-1 Auto Proposal**: `(1298, 1325, 2474, 1413)` ($\text{IoU} = 0.0000$, $\text{Recall} = 0.0\%$).
- **Failure Mode**: The algorithm selected the dark shadow boundary cast by the cave ceiling overhang immediately above the headrest.
- **Classifier Impact**: ViT-B/16 predicted `brahmi` (87.9%, cosine $+0.4273$).

### Case 5: `TBSI_029_original.jpg` (Chalked Positive Control)
- **Manual Reference**: Full chalked ribbon `(100, 180, 2400, 750)`.
- **Top-1 Auto Proposal**: `(1786, 354, 2186, 625)` ($\text{IoU} = 0.0827$, $\text{Recall} = 8.3\%$, $\text{Precision} = 100.0\%$).
- **Behavior**: On high-contrast chalked script, the algorithm successfully isolated text, but **fragmented the single line into multiple disjoint boxes** (Candidates 1, 2, and 3 captured 2-character snippets along the line). While precision was $100\%$, coverage was only $8.3\%$.

---

## 6. Exact Reproduction Commands

The experiment is fully reproducible via CLI:

```bash
# 1. Run full benchmark suite on all 5 images
python experiments/automatic_localization_prototype/auto_localize_and_infer.py --run-benchmark-suite

# 2. Test interactive candidate generation on a single photograph
python experiments/automatic_localization_prototype/auto_localize_and_infer.py --image dataset/whole_inscriptions/tamil_brahmi/Sittanavasal_Rock_Bed_Inscription.jpg --select-candidate 1

# 3. Verify active checkpoint integrity
powershell -Command "Get-FileHash checkpoints/six_class_vit/best_vit_model.pth -Algorithm SHA256"
# Expected Hash: 6EFA33C8A11389F64B8FA5040A7DE6C4CF8042FB5359889CA4A23F5653FDAA5C
```

---

## 7. Recommendation & Conclusion

1. **Definitive Finding**: Classical CV (morphological filtering, edge clustering, adaptive thresholding) is **fundamentally unsuited** for locating ancient in-situ petroglyphs on natural rock monuments. High-contrast modern graffiti, ceiling shadows, and floor cracks consistently outscore weathered 2,100-year-old incisions.
2. **Safety Risk**: Allowing unverified automated proposals to pass into the ViT classifier creates extreme false positives, as modern scratches and natural cracks produce $>99\%$ confident predictions for `tamil_brahmi`.
3. **Recommended Direction**:
   - For production: Maintain manual region selection (or user-guided bounding box UI) where an epigraphist specifies the inscription extent.
   - For research: Automatic localization would require modern deep text-line detectors (e.g. fine-tuned DBNet, CRAFT, or YOLOv11-OBB) trained on an annotated corpus of petroglyphic and rock-cut inscriptions, combined with confidence-gated rejection.

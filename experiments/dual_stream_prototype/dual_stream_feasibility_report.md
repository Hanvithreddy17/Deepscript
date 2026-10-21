# DeepScript — Dual-Stream Prototype Feasibility Audit
**Isolated Research Assessment: Global Whole-Stone Framing + Local Inscription Crop Fusion**

- **Audit Date**: October 21, 2026
- **Target Repository**: `DeepScript (c:\Users\hanvi\OneDrive\Desktop\Deepscript\Deepscript)`
- **Target Checkpoint**: `checkpoints/six_class_vit/best_vit_model.pth`
- **Verified Checkpoint SHA-256**: `6efa33c8a11389f64b8fa5040a7de6c4cf8042fb5359889ca4a23f5653fdaa5c` (Integrity: 100% Intact)
- **Status**: Audit & Design Only — Strictly No Implementation, No Model Training, No Production Modification

---

## 1. Executive Summary & Feasibility Verdict

### 1.1 Context and Motivation
In the recent frozen-checkpoint zero-leakage evaluation (`experiments/dataset_audit/frozen_evaluation/`) and controlled failure analysis (`experiments/dataset_audit/frozen_evaluation/crop_failure_analysis/`), three critical empirical phenomena were discovered:
1. **Background Clutter Fix**: Cropping out non-inscribed cavern floor, cave ceilings, and modern graffiti resolved 3 out of 5 Tamil-Brahmi failure cases (`SET_A_03`, `SET_B_01`, `SET_B_08`), driving cosine similarity improvements up to $+0.5474$.
2. **Crop-Induced Regression Hazard**: On positive control `SET_A_04` (a clean close-up stone relief), applying a tight crop triggered horizontal sliding-window tiling that stripped surrounding stone context and inadvertently boosted Ashokan Brahmi activations, causing an outright **regression from correct Tamil-Brahmi to false Brahmi**.
3. **Unchalked Stone Ambiguity**: On unchalked daylight weathered granite (`SET_A_01`, `SET_A_02`), cropping alone failed to separate Tamil-Brahmi from Ashokan Brahmi because both scripts share identical geometric consonant primitives, and natural weathered relief lacks the high contrast of the chalked TBSI training set.

These findings establish that **automatic cropping cannot be deployed as a naive front-end heuristic in production**, as it introduces catastrophic regressions on clean captures. However, they provide strong empirical motivation for a **Dual-Stream Prototype** that processes both the uncropped macro-stone image (preserving rock morphology, global illumination, and spatial context) and an inscription-focused crop (amplifying chiseled stroke contours and fine diacritics).

### 1.2 Feasibility Verdict: PROCEED TO ISOLATED PROTOTYPE (Subject to Multi-Class Curation)
- **Architectural Feasibility**: **HIGH**. The existing ViT-B/16 backbone and MLP projector output $L_2$-normalized 256-dimensional embeddings on the unit hypersphere $S^{255}$. A training-free dual-stream spherical fusion can be implemented immediately in an isolated evaluation script without modifying production inference or retraining the checkpoint.
- **Learned Fusion Feasibility**: **MEDIUM**. A learned cross-attention or gating module is architecturally straightforward but requires training. Training such a module while keeping the ViT backbone frozen is technically feasible, but requires an authentic, multi-class training/validation split to avoid catastrophic overfitting.
- **Dataset Feasibility**: **CRITICAL BLOCKER (Multi-Class)**. An audit of `dataset/metadata.csv` reveals that the repository contains only **13 test images across all six script classes**, of which several are non-stone artifacts (a clay seal, a steatite reliquary, a paper rubbing facsimile) and four are duplicate photographs of the same Sittanavasal stone bed. **A valid, balanced, site-disjoint, multi-class whole-stone evaluation benchmark DOES NOT CURRENTLY EXIST in the repository.** Multi-class claims cannot be evaluated until authentic external stone inscriptions are curated.

---

## 2. Architectural & Interface Verification (Source Code Ground Truth)

Every component of the existing inference and model pipeline was verified directly against the active source code:

| Component | Source File | Parameter / Implementation Details | Verified Status |
| :--- | :--- | :--- | :--- |
| **Vision Backbone** | `models/backbone.py` (`ViTFeatureExtractor`) | `torchvision.models.vit_b_16(weights=None)`, top classification head replaced with `nn.Identity()`. Raw feature output dimension: $\mathbf{h} \in \mathbb{R}^{768}$. | Confirmed |
| **Metric Projector** | `models/backbone.py` (`EmbeddingProjector`) | Two-layer MLP: `LayerNorm(768) -> Linear(768, 768) -> GELU() -> Dropout(0.1) -> Linear(768, 256) -> F.normalize(p=2, dim=-1)`. Embeddings reside on unit hypersphere $S^{255} \subset \mathbb{R}^{256}$. | Confirmed |
| **Metric Head** | `models/classifier.py` (`CosineSimilarityHead`) | Normalized linear operator: $\mathbf{W} \in \mathbb{R}^{6 \times 256}$, $\mathbf{W}_{\text{norm}} = \text{L2Norm}(\mathbf{W})$. Cosine similarities: $\mathbf{c} = \mathbf{z} \cdot \mathbf{W}_{\text{norm}}^T \in [-1, 1]^6$. Scale parameter: $s \approx 15.9930$. Logits: $\mathbf{L} = s \cdot \mathbf{c}$. | Confirmed |
| **Few-Shot Head** | `models/classifier.py` (`PrototypicalHead`) | Centroid computation: $\mathbf{p}_c = \frac{1}{|S_c|}\sum_{i \in S_c} \mathbf{z}_i$, with optional $L_2$ normalization for cosine metric: $\mathbf{p}_c \leftarrow \text{L2Norm}(\mathbf{p}_c)$. | Confirmed |
| **Production Inference** | `backend/vit_inference.py` (`ViTInferenceEngine`) | Single-stream inference with aspect-ratio tiling ($AR > 1.8$). Logits averaged across tiles: $\bar{\mathbf{L}} = \frac{1}{M+1}\sum_{m=1}^{M+1} \mathbf{L}_m$. Rejection filters: blank image check, cosine similarity floor ($0.280$), classification margin floor ($0.040$). | Confirmed |
| **Active Checkpoint** | `checkpoints/six_class_vit/best_vit_model.pth` | SHA-256: `6efa33c8a11389f64b8fa5040a7de6c4cf8042fb5359889ca4a23f5653fdaa5c`. Classes: `["brahmi", "grantha", "gupta", "kadamba", "kharosthi", "tamil_brahmi"]`. Scale: $s = 15.9930$. | Confirmed (Intact) |

### 2.1 Critical Finding: The Unit Hypersphere Constraint
The projector enforces strict $L_2$ normalization:
$$\mathbf{z} = \frac{\mathbf{v}}{\|\mathbf{v}\|_2}, \quad \|\mathbf{z}\|_2 = 1.0$$
Consequently, the cosine similarity head computes pure angular dot products:
$$\cos(\theta_c) = \mathbf{z} \cdot \mathbf{w}_c, \quad \text{where } \|\mathbf{z}\|_2 = 1, \; \|\mathbf{w}_c\|_2 = 1$$
**Any fusion strategy that operates in the embedding space must guarantee that the fused vector $\mathbf{z}_{\text{fused}}$ has unit norm ($\|\mathbf{z}_{\text{fused}}\|_2 = 1$).** Feeding an unnormalized vector into downstream distance or similarity calculations distorts the scale factor $s$ and invalidates the calibrated confidence and margin thresholds ($0.280$ cosine floor, $0.040$ margin).

---

## 3. The Dual-Stream Formulation

```
                                  [ Input Image (Whole Stone) ]
                                                │
                       ┌────────────────────────┴────────────────────────┐
                       ▼                                                 ▼
             [ Stream 1: Global ]                              [ Stream 2: Local ]
          Original Uncropped Image                         Inscription Bounding Box
          (Macro Context & Surface)                        (Micro Stroke & Diacritics)
                       │                                                 │
            ViT-B/16 Backbone (Frozen)                        ViT-B/16 Backbone (Frozen)
          Feature Output: h_g ∈ R^768                       Feature Output: h_l ∈ R^768
                       │                                                 │
            MLP Projector (Frozen)                            MLP Projector (Frozen)
          Embedding: z_g ∈ S^255                            Embedding: z_l ∈ S^255
                       │                                                 │
                       └────────────────────────┬────────────────────────┘
                                                │
                                    [ Feature Fusion Module ]
                                    (Training-Free or Learned)
                                                │
                                                ▼
                                    Fused Embedding: z_fused ∈ S^255
                                                │
                                                ▼
                                    [ Cosine Similarity Head ]
                                    Logits = s · (z_fused · W^T)
                                                │
                                                ▼
                                     Script Class Prediction
```

### 3.1 Roles of the Two Streams
1. **Global Stream ($\mathbf{z}_g$)**:
   - Ingests the full field photograph resized to $224 \times 224$ (or tiled if wide).
   - Captures macro-archaeological features: rock face geometry, chisel groove depth, surface weathering patterns, patination, and whole-stone layout.
   - Prevents catastrophic context loss (the mechanism behind the `SET_A_04` regression).
2. **Local Stream ($\mathbf{z}_l$)**:
   - Ingests the cropped inscription ribbon isolating visible carved characters.
   - If aspect ratio $W/H > 1.8$, applies aspect-preserving sliding window tiling (1–5 square patches) and aggregates patch embeddings.
   - Eliminates background cavern floor, sky, surrounding modern vegetation, and graffiti clutter (the mechanism that fixed `SET_A_03`, `SET_B_01`, `SET_B_08`).

---

## 4. Proposed Feature Fusion Strategies

We propose three distinct fusion strategies ranging from a zero-training, mathematically rigorous baseline to a learned attention architecture.

### 4.1 Strategy 1: Training-Free Normalized Spherical Embedding Fusion (Recommended Baseline)
This strategy requires **zero retraining**, preserves the active checkpoint weights completely, and can be evaluated immediately in an isolated prototype script.

#### Mathematical Formulation
Let $\mathbf{z}_g \in S^{255}$ be the $L_2$-normalized embedding from the global image, and let $\mathbf{z}_l \in S^{255}$ be the $L_2$-normalized embedding from the local crop (aggregated across tiles if tiled).
We define the fused embedding as the normalized linear combination:
$$\mathbf{z}_{\text{fused}}(\alpha) = \frac{\alpha \mathbf{z}_g + (1 - \alpha) \mathbf{z}_l}{\|\alpha \mathbf{z}_g + (1 - \alpha) \mathbf{z}_l\|_2}, \quad \alpha \in [0, 1]$$

#### Geometric Property: Spherical Linear Interpolation (Slerp)
On the unit hypersphere $S^{255}$, the angle between global and local embeddings is $\Omega = \arccos(\mathbf{z}_g \cdot \mathbf{z}_l)$. The geodesic interpolation (Slerp) is:
$$\mathbf{z}_{\text{slerp}}(t) = \frac{\sin((1-t)\Omega)}{\sin \Omega}\mathbf{z}_g + \frac{\sin(t\Omega)}{\sin \Omega}\mathbf{z}_l, \quad t \in [0, 1]$$
- When $\Omega$ is small (i.e. global and local streams agree on the visual style), the chordal normalized average $\mathbf{z}_{\text{fused}}(\alpha)$ closely approximates $\mathbf{z}_{\text{slerp}}(t)$.
- **Default Hyperparameter Recommendation**: $\alpha = 0.40$ (giving $60\%$ weight to local stroke relief while maintaining $40\%$ global contextual regularization).

#### Score-Level Equivalent (Logit Blending)
Alternatively, fusion can be executed at the logit level:
$$\mathbf{L}_{\text{blend}} = \alpha \mathbf{L}_g + (1 - \alpha) \mathbf{L}_l$$
Since $\mathbf{L}_g = s (\mathbf{z}_g \mathbf{W}^T)$ and $\mathbf{L}_l = s (\mathbf{z}_l \mathbf{W}^T)$, by linearity:
$$\mathbf{L}_{\text{blend}} = s \left[ (\alpha \mathbf{z}_g + (1 - \alpha)\mathbf{z}_l) \mathbf{W}^T \right]$$
**Key Distinction**: In logit blending, the effective embedding is $\mathbf{v} = \alpha \mathbf{z}_g + (1-\alpha)\mathbf{z}_l$, whose norm $\|\mathbf{v}\|_2 \le 1$. If global and local streams disagree ($\mathbf{z}_g \cdot \mathbf{z}_l \ll 1$), $\|\mathbf{v}\|_2$ drops significantly, which acts as an **inherent uncertainty penalty** reducing the scale of the logits and softening softmax confidence. In contrast, embedding-level renormalization forces $\|\mathbf{z}_{\text{fused}}\|_2 = 1.0$, preserving the calibrated scale $s = 15.9930$. Both variants should be evaluated side-by-side in the prototype.

---

### 4.2 Strategy 2: Learned Gated Cross-Attention / Feature Modulation Fusion (Trainable)
This strategy introduces a trainable module to dynamically weight and modulate features based on visual content, while keeping the ViT-B/16 backbone frozen.

#### Architecture: Dual-Stream Gated Cross-Attention Unit (DS-GCA)
Rather than a fixed scalar $\alpha$, the network learns a content-dependent gating vector $\mathbf{g} \in [0, 1]^{256}$:

```
     z_g [256] ────────┐
                       ▼
                 [ Concat: 512 ] ──► [ Linear(512, 128) ] ──► [ GELU ] ──► [ Linear(128, 256) ] ──► [ Sigmoid ]
                       ▲                                                                                     │
     z_l [256] ────────┤                                                                            Gate g ∈ [0, 1]^256
                       │                                                                                     │
                       ├─────────────────────────────────────────────────────────────────────────────────────┘
                       ▼
     z_mod = g ⊙ z_l + (1 - g) ⊙ z_g
                       │
                       ▼
     z_fused = L2Norm( z_mod ) ∈ S^255
                       │
                       ▼
     Logits = s · (z_fused · W^T)
```

#### Why This Strategy Requires Training
1. **Content-Selective Gating**: The model must learn when local strokes contain diagnostic letterforms (e.g., Sittanavasal *Eruminatu kumul-ur...* where local stream should dominate, $g \to 1$) versus when local strokes are ambiguous or unchalked (e.g., `SET_A_01` where global context is needed, $g \to 0$).
2. **Parameters to Train**:
   - Gating network: $\sim 98\text{k}$ parameters.
   - Backbone ViT and Projector: **100% Frozen** to preserve pre-trained and fine-tuned feature representations.
   - Cosine head weights $\mathbf{W}$: Optionally fine-tuned with a tiny learning rate ($10^{-5}$) or kept frozen.
3. **Training Objective**:
   Normalized Cosine Cross-Entropy Loss:
   $$\mathcal{L} = -\log \frac{\exp(s \cdot \mathbf{z}_{\text{fused}} \cdot \mathbf{w}_y)}{\sum_{c=1}^C \exp(s \cdot \mathbf{z}_{\text{fused}} \cdot \mathbf{w}_c)}$$
   Regularization: $L_2$ weight decay on the gating MLP, plus an entropy penalty on the gate $g$ to prevent collapse to a single stream.

---

### 4.3 Strategy 3: Dynamic Margin-Gated Confidence Selection (Heuristic Hybrid)
A rule-based meta-arbitration scheme that uses the existing validation-calibrated cosine metrics without requiring any training:
1. Extract $\mathbf{z}_g$ and compute top cosine $c_g^{(1)}$ and margin $\Delta_g = c_g^{(1)} - c_g^{(2)}$.
2. Extract $\mathbf{z}_l$ and compute top cosine $c_l^{(1)}$ and margin $\Delta_l = c_l^{(1)} - c_l^{(2)}$.
3. **Arbitration Logic**:
   - If $\Delta_l \ge 0.080$ and $c_l^{(1)} \ge 0.350$ (local crop shows decisive, high-confidence script alignment), assign $\alpha = 0.15$ (local dominant).
   - If $\Delta_l < 0.040$ (local crop is ambiguous or tiled characters are indistinct), assign $\alpha = 0.70$ (global dominant).
   - Otherwise, set $\alpha = 0.40$ (balanced spherical blend).

---

## 5. Mathematical Guardrails: Preventing Feature Space Mismatches & Normalization Corruption

To ensure the prototype produces mathematically valid cosine similarities that match the calibration of `backend/vit_inference.py`, the following four guardrails must be strictly enforced:

### Guardrail 1: Never Concatenate Incompatible Feature Spaces
- **Prohibited**: Concatenating raw 768-d ViT features with 256-d projected embeddings ($[768 \,\|\, 256] \in \mathbb{R}^{1024}$).
- **Reason**: The 768-d space is an unnormalized, highly redundant Euclidean representation containing ImageNet semantics; the 256-d space is an $L_2$-normalized metric manifold shaped by cosine loss. Concatenating them corrupts the angular distance metric.
- **Rule**: All fusion operations must occur **either** entirely within the 768-d unprojected space (before the projector) **or** entirely within the 256-d metric space (after the projector).

### Guardrail 2: Strictly Re-Normalize Fused Embeddings to Unit Length
- **Prohibited**: Passing $\mathbf{v} = \alpha \mathbf{z}_g + (1-\alpha)\mathbf{z}_l$ directly into distance metrics without normalization.
- **Mathematical Proof of Distortion**:
  Let $\mathbf{z}_g, \mathbf{z}_l \in S^{255}$, with cosine separation $\cos \theta = \mathbf{z}_g \cdot \mathbf{z}_l < 1$.
  The squared norm of the unnormalized average is:
  $$\|\mathbf{v}\|_2^2 = \alpha^2 + (1-\alpha)^2 + 2\alpha(1-\alpha)\cos \theta < 1.0$$
  If $\cos \theta = 0.20$ and $\alpha = 0.50$, then $\|\mathbf{v}\|_2 = \sqrt{0.25 + 0.25 + 2(0.25)(0.20)} = \sqrt{0.60} \approx 0.7746$.
  If $\mathbf{v}$ is evaluated against class prototype $\mathbf{w}_c$, the raw dot product is:
  $$\mathbf{v} \cdot \mathbf{w}_c = 0.7746 \cdot \cos \phi_c$$
  This causes an artificial **$22.5\%$ attenuation** of all cosine similarities, causing perfectly valid inscriptions to fall below the calibrated cosine floor ($0.280$) and triggering false `unsupported_script` rejections.
- **Mandatory Correction**: Always apply `F.normalize(z_fused, p=2, dim=-1)` prior to metric evaluation.

### Guardrail 3: Proper Aspect-Ratio Tiling Aggregation in the Local Stream
- Inscription crops frequently have high aspect ratios ($AR > 3.0$ or even $AR > 8.0$ for long rock bed inscriptions like Sittanavasal).
- `vit_inference.py` tiles such images into $K$ overlapping square crops ($1 \le K \le 5$).
- **Rule for Local Stream Tiling**:
  Each tile $k$ must be passed through the backbone and projector to obtain $\mathbf{z}_{l,k} \in S^{255}$.
  The local stream embedding is then formed by centroid pooling and spherical projection:
  $$\bar{\mathbf{z}}_l = \frac{1}{K}\sum_{k=1}^K \mathbf{z}_{l,k}, \quad \mathbf{z}_l = \frac{\bar{\mathbf{z}}_l}{\|\bar{\mathbf{z}}_l\|_2}$$
  Only after $\mathbf{z}_l$ has been formed on $S^{255}$ is it fused with the global embedding $\mathbf{z}_g$.

### Guardrail 4: Scale Parameter Isolation
- The scale factor $s = 15.9930$ stored in `model.head.scale` was learned during training on normalized embeddings.
- In inference, logits are computed as $\mathbf{L} = s \cdot (\mathbf{z}_{\text{fused}} \cdot \mathbf{W}^T)$.
- Cosine similarities must be extracted by unscaling: $\mathbf{c} = \mathbf{L} / s$.
- **Rule**: Never multiply the fused embedding by $s$ before fusion; scaling must remain strictly the final operation producing logits.

---

## 6. Multi-Class Evaluation Protocol (6 Scripts, Site-Disjoint, Leakage-Resistant)

To scientifically validate the dual-stream prototype and ensure it does not merely overfit to Tamil-Brahmi failure cases, it must be evaluated across all six supported script families under strict zero-leakage conditions.

### 6.1 Supported Script Families
1. `brahmi` (Ashokan Brahmi)
2. `tamil_brahmi` (Early Southern Brahmi / Sangam Epigraphy)
3. `grantha` (Pallava & Chola Grantha)
4. `gupta` (Late Northern Brahmi / Siddhamatrika)
5. `kadamba` (Box-Headed Western Brahmi)
6. `kharosthi` (Gandharan Right-to-Left)

### 6.2 Strict Leakage Prevention Rules
To prevent optimistic bias, the evaluation set must strictly enforce two orthogonal isolation boundaries:
1. **Site-Level Disjointness**:
   - No image in the evaluation set may originate from an archaeological site that appears in the training or validation splits of `checkpoints/six_class_vit/best_vit_model.pth`.
   - Specifically prohibited for test inclusion due to training exposure:
     - *Brahmi*: Ramgarh / Jogimara Caves, Kangra Fort, Nashik / Ajanta Cave 17, Erragudi (Yerragudi), Vidisha (Heliodorus Pillar).
     - *Tamil-Brahmi*: Mamandur, Mangulam, Pugalur, Arittapatti, Mettuppatti, Alagarmalai, Karungalakkudi, Aiyarmalai, Thiruparankundram.
     - *Grantha*: Mandagapattu, Thalavanur, Velvikudi, Chera sites, Chola sites.
     - *Gupta*: Arang, Raipur, Allahabad Fort (Prayag Prashasti), Eran.
     - *Kadamba*: Banavasi, Malavalli, Talagunda.
     - *Kharosthi*: Shahbazgarhi, Niya.
2. **Physical-Monument / Inscription-Level Disjointness**:
   - Multiple photographs, angles, or lighting conditions of the **same physical stone bed, pillar, or rock boulder** must NEVER be split between train/val and test.
   - In evaluation, multiple captures of the same monument (e.g. Sittanavasal Bed #1) must be grouped as a single monument cluster, reporting both capture-level and monument-aggregated metrics.

### 6.3 Mandatory Inclusion of Regression Controls
The evaluation benchmark must include positive controls that already succeed on uncropped images, ensuring the dual-stream prototype improves difficult cases without degrading baseline accuracy:
1. **Tamil-Brahmi Positive Control 1 (`SET_A_04`)**:
   - Sittanavasal close-up relief.
   - *Baseline Full Image*: PASS (`tamil_brahmi`, $19.6\%$).
   - *Single-Stream Crop*: **REGRESSION FAIL** (`brahmi`, $25.3\%$).
   - *Prototype Acceptance Gate*: Must remain **PASS (`tamil_brahmi`)**.
2. **Tamil-Brahmi Positive Control 2 (`SET_B_02`)**:
   - TBSI_088 Vikkiramangalam drip ledge.
   - *Baseline Full Image*: PASS (`tamil_brahmi`, $58.6\%$).
   - *Single-Stream Crop*: PASS (`tamil_brahmi`, $57.5\%$).
   - *Prototype Acceptance Gate*: Must remain **PASS (`tamil_brahmi`)** with confidence $\ge 50\%$.
3. **Non-Tamil Script Positive Controls (5 Classes)**:
   - At least 2 verified, uncropped whole-stone captures per class for `brahmi`, `grantha`, `gupta`, `kadamba`, and `kharosthi`.
   - *Prototype Acceptance Gate*: Dual-stream inference must achieve $0\%$ regression rate on these established controls.

### 6.4 Metrics & Disaggregation Protocol
Results must be reported across four disaggregated levels:
1. **Capture-Level Sensitivity**: Percentage of individual test images correctly classified.
2. **Monument-Grouped Sensitivity**: Majority vote or mean-pooled embedding prediction per physical inscription monument (eliminates capture-count bias).
3. **Control Regression Rate**:
   $$\text{Regression Rate} = \frac{\sum \mathbb{I}(\text{Full}=\text{Correct} \land \text{DualStream}=\text{Incorrect})}{\sum \mathbb{I}(\text{Full}=\text{Correct})} \times 100\%$$
   Acceptance threshold: **$\text{Regression Rate} = 0.0\%$**.
4. **Cosine Margin Shift ($\Delta_{\text{margin}}$)**: Mean shift in winning class separation:
   $$\Delta_{\text{margin}} = \left[\cos(y_{\text{true}}) - \max_{c \ne y_{\text{true}}}\cos(c)\right]_{\text{dual}} - \left[\cos(y_{\text{true}}) - \max_{c \ne y_{\text{true}}}\cos(c)\right]_{\text{global}}$$
   A positive shift confirms that local stroke fusion amplifies true-class signal.

---

## 7. Critical Dataset Gap Analysis: Current Test Set Audit

To determine whether the repository currently contains a valid multi-class evaluation benchmark, an exhaustive audit of `dataset/metadata.csv` (74 total images across all splits) was conducted:

### 7.1 Script-by-Script Breakdown of Existing Test Split

| Script Class | Total in Repo | Train | Val | Test Split in `metadata.csv` | Description of Current Test Items | Archaeological & Epigraphic Validity for Whole-Stone Test |
| :--- | :---: | :---: | :---: | :---: | :--- | :--- |
| **`brahmi`** | 10 | 6 | 2 | **2** | 1. `Devanampiyena Piyadasi in the Lumbini Edict of Ashoka.jpg`<br>2. `Paderia Pillar.png` | **INVALID**: Both items represent the exact same monument (Lumbini / Rummindei Ashokan Pillar). Item #2 is a **2D black-and-white ink rubbing facsimile**, not a real stone photograph. Effective $N=1$. |
| **`grantha`** | 10 | 6 | 2 | **2** | 1. `Vijaynagar Venkathiraya Inscription, 1605 AD.jpg`<br>2. `Vijaynagar Virupaksha Inscription, 1481 AD.jpg` | **MARGINAL**: Authentic stone inscriptions from Vellore and Tiruvannamalai, but sample size is severely underpowered ($N=2$). |
| **`gupta`** | 10 | 6 | 2 | **2** | 1. `Nalanda clay seal of Vishnugupta.jpg`<br>2. `Sanchi, inscription of Chandragupta II.jpg` | **INVALID**: Item #1 is a small **terracotta clay seal**, not a stone inscription. Item #2 is a single stone railing inscription from Sanchi. Effective stone sample size $N=1$. |
| **`kadamba`** | 9 | 6 | 2 | **1** | 1. `Kadamba inscription from Kerala.jpg` | **INVALID**: Extreme sparsity ($N=1$). A single stone slab cannot constitute an evaluation benchmark. |
| **`kharosthi`** | 10 | 7 | 2 | **1** | 1. `Reliquary casket with Kharoshti characters, Bimran Stupa.jpg` | **INVALID**: Item is a small **steatite reliquary casket / bowl**, not a monumental rock or stone inscription. Effective monumental stone sample size $N=0$. |
| **`tamil_brahmi`** | 25 | 15 | 5 | **5** | 1. `TBSI_029_original.jpg` (Kongarpuliyankulam)<br>2. `Sittanavasal_Rock_Bed_Inscription.jpg`<br>3. `Stone_bed_with_inscription_in_sittanavasal.JPG`<br>4. `Chithannavasal_cave_inscription_1.JPG`<br>5. `Chithannavasal_cave_inscription_2.JPG` | **PSEUDO-REPLICATED**: 4 of the 5 test images are different photographic angles of the **exact same stone bed** at Sittanavasal! Effective independent monument sample size $N=2$. |
| **Total** | **74** | **46** | **15** | **13** | **13 images total across 6 script classes** | **FATALLY UNDERPOWERED & CONTAMINATED** |

### 7.2 Findings & Deficiencies
1. **Modality Contamination**:
   The current test set mixes radically different physical media:
   - 1 paper ink rubbing (`Paderia Pillar.png`).
   - 1 terracotta monastic sealing (`Nalanda clay seal of Vishnugupta.jpg`).
   - 1 steatite reliquary casket (`Reliquary casket with Kharoshti characters...`).
   These non-stone objects do not possess stone weathering, rock grain, or chisel relief, violating the stated objective of evaluating whole-stone inscriptions.
2. **Extreme Sparsity**:
   Non-Tamil scripts have only 1 or 2 test samples each (Kadamba: 1, Kharosthi: 1, Gupta: 1 stone, Brahmi: 1 stone). A single misclassification in Kadamba swings test accuracy between $0\%$ and $100\%$.
3. **Severe Pseudo-Replication**:
   In Tamil-Brahmi, 4 of the 5 test images (`SET_A_01`, `SET_A_02`, `SET_A_03`, `SET_A_04`) depict the identical physical monument (Sittanavasal Bed #1).
4. **Definitive Conclusion**:
   **A valid, balanced, site-disjoint, multi-class whole-stone evaluation set DOES NOT CURRENTLY EXIST in the repository.**
   Evaluating a dual-stream prototype exclusively on the existing 13 test images would yield statistically meaningless metrics and risk tuning fusion parameters to idiosyncratic quirks of three non-stone artifacts.

---

## 8. Research Prototype Implementation Roadmap & Recommended Next Steps

### 8.1 Proposed Research Prototype Directory Layout
To ensure complete isolation from production code, the prototype must be developed entirely under `experiments/dual_stream_prototype/`:

```
experiments/dual_stream_prototype/
├── dual_stream_feasibility_report.md       # This audit report
├── prototype_dataset/                      # Isolated evaluation manifests & crop annotations
│   ├── crop_bounding_boxes.json            # Verified manual bounding boxes for test specimens
│   └── test_manifest_multiclass.json       # Curated multi-class test metadata
├── evaluate_dual_stream_baseline.py        # Isolated script: Strategy 1 (Training-Free Spherical Fusion)
├── train_learned_fusion_gate.py            # Isolated script: Strategy 2 (DS-GCA Training, frozen backbone)
└── results/
    ├── dual_stream_evaluation_results.csv  # 6-class results matrix across α ∈ [0.0, 1.0]
    └── fusion_performance_summary.json     # Capture vs monument metrics, regression analysis
```

### 8.2 Recommended Step-by-Step Next Actions

#### Phase 1: Curate a Minimum Viable Multi-Class Stone Evaluation Set (Pre-Requisite)
Before implementing or benchmarking dual-stream inference, curate an authentic, site-disjoint evaluation set:
- Sourcing: 5 authentic whole-stone inscriptions per script (30 total images across all 6 classes).
- Criteria: Must be authentic stone (rock cut, cave drip ledge, boulder edict, or stone slab), 100% site-disjoint from training, with visible inscribed text.
- Metadata: Record site, monument, period, and manual bounding box coordinates for each image.

#### Phase 2: Implement Strategy 1 (Training-Free Spherical Fusion Prototype)
- Script: `evaluate_dual_stream_baseline.py` in `experiments/dual_stream_prototype/`.
- Load active checkpoint `checkpoints/six_class_vit/best_vit_model.pth` (verify SHA-256).
- Evaluate parameter sweep: $\alpha \in [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]$:
  - $\alpha = 0.0$: Pure local crop.
  - $\alpha = 1.0$: Pure global uncropped baseline.
  - $\alpha \in (0, 1)$: Spherical blend on $S^{255}$.
- Measure:
  1. Does $\alpha = 0.4$ maintain the fixes on `SET_A_03`, `SET_B_01`, and `SET_B_08`?
  2. Does $\alpha = 0.4$ prevent the regression on `SET_A_04`?
  3. Does $\alpha = 0.4$ cause regressions on any non-Tamil classes?

#### Phase 3: Evaluate Learned Gated Fusion (Strategy 2) Only if Strategy 1 Fails
- If Strategy 1 successfully resolves failure cases without control regressions, learned fusion may be unnecessary.
- If fixed $\alpha$ cannot simultaneously protect `SET_A_04` and fix `SET_A_03`, implement the lightweight gating MLP (`train_learned_fusion_gate.py`) trained on an isolated split with frozen ViT backbone.

---

## 9. Checkpoint Verification Sign-Off

In accordance with strict safety requirements, the active production checkpoint was cryptographically verified:
- **File**: `checkpoints/six_class_vit/best_vit_model.pth`
- **Expected SHA-256**: `6efa33c8a11389f64b8fa5040a7de6c4cf8042fb5359889ca4a23f5653fdaa5c`
- **Computed SHA-256**: `6efa33c8a11389f64b8fa5040a7de6c4cf8042fb5359889ca4a23f5653fdaa5c`
- **Verification Status**: **MATCH (100% Intact & Untouched)**
- **Production Files Modified**: **NONE (0 files changed)**
- **Model Training Performed**: **NONE (Audit Only)**

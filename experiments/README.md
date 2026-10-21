# DeepScript Research & Experiments Index

This directory documents the systematic research, dataset auditing, failure diagnostic, localization prototyping, and metric fine-tuning experiments conducted on the DeepScript 6-class whole-inscription classifier.

---

## Experiment Index

| # | Directory / Report | Objective & Key Finding | Active Checkpoint Integrity |
|---|---|---|:---:|
| **1** | [`experiments/faculty_failure/`](./faculty_failure/README.md) | **Faculty Inscription Failure Diagnostics**: Systematic ROI crop study on Medieval Tamil stone panel. Proved failure is due to out-of-distribution script (Medieval Tamil with pulli/grantha ligatures), not boundary clutter. | Preserved (`6efa33c8...`) |
| **2** | [`experiments/dataset_audit/`](./dataset_audit/external_dataset_suitability_report.md) | **External Dataset Suitability Audit**: Comprehensive audit of Mendeley TBSI, Kaggle Brahmi, and Hugging Face datasets. Developed leakage-resistant site-disjoint splits. | Preserved (`6efa33c8...`) |
| **3** | [`experiments/dataset_audit/frozen_evaluation/`](./dataset_audit/frozen_evaluation/leakage_audit_final_report.md) | **Frozen Checkpoint Site-Disjoint Benchmark**: Rigorous held-out evaluation across 4 unseen sites (Sittanavasal, Vikkiramangalam, Varichiyur, Kongarpuliyankulam). Established baseline sensitivity (70.6%). | Preserved (`6efa33c8...`) |
| **4** | [`experiments/dataset_audit/frozen_evaluation/crop_failure_analysis/`](./dataset_audit/frozen_evaluation/crop_failure_analysis/crop_failure_analysis_report.md) | **Controlled Crop Failure Analysis**: Evaluated manual and localized crops on the 5 misclassified captures, demonstrating that text-focused crops eliminate rock texture bias. | Preserved (`6efa33c8...`) |
| **5** | [`experiments/dual_stream_prototype/`](./dual_stream_prototype/dual_stream_feasibility_report.md) | **Dual-Stream Prototype Feasibility Audit**: Architectural blueprint combining global whole-stone embeddings with localized inscription crops using early/late fusion. | Preserved (`6efa33c8...`) |
| **6** | [`experiments/dual_stream_prototype/`](./dual_stream_prototype/dataset_collection_plan.md) | **Six-Class Whole-Inscription Dataset Collection Plan**: Detailed audit and provenance tracking of stone vs non-stone media, targeting epigraphic gaps across all 6 scripts. | Preserved (`6efa33c8...`) |
| **7** | [`experiments/automatic_localization_prototype/`](./automatic_localization_prototype/README.md) | **Autonomous Localization Prototypes**: Evaluated unsupervised edge/gradient bounding box proposal algorithms for hands-free crop extraction. | Preserved (`6efa33c8...`) |
| **8** | [`experiments/pretrained_detector_prototype/`](./pretrained_detector_prototype/README.md) | **CRAFT Text Detection Prototyping**: Explored pre-trained text character region detectors on ancient weathered stone surfaces. | Preserved (`6efa33c8...`) |
| **9** | [`experiments/tamil_brahmi_finetune/`](./tamil_brahmi_finetune/final_report.md) | **Fast-Track Tamil-Brahmi Fine-Tuning & Evaluation**: Integrated 61 verified unique TBSI captures across 17 sites into an isolated metric-learning checkpoint. Improved accuracy from **80.0% to 88.0%**, Tamil-Brahmi capture sensitivity from **70.6% to 88.2%**, and monument sensitivity to **100% (13/13)**. | Preserved (`6efa33c8...`) |

---

## Safety & Non-Destructive Principles

All experiments adhere strictly to DeepScript reproducible research guardrails:
1. **Production Checkpoint Invariance**: The production checkpoint (`checkpoints/six_class_vit/best_vit_model.pth`, SHA-256: `6efa33c8a11389f64b8fa5040a7de6c4cf8042fb5359889ca4a23f5653fdaa5c`) is never overwritten by experimental runs.
2. **Zero Site Leakage**: Archaeological sites in evaluation splits are completely disjoint from training and validation sets.
3. **Physical Inscription Grouping**: Multiple views, lighting angles, and crops of the same physical monument are grouped together before splitting.
4. **Honest Metric Reporting**: Capture-level accuracy and monument-level consensus sensitivity are reported separately with exact confidence and cosine margins.

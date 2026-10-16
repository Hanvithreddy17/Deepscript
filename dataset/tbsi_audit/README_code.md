# DMB / TBSI binarization benchmark

Set the dataset path in `config.py` (`DataConfig.dataset_root`, default `D:/phd papers/...`; several scripts also
have a `ROOT`/`CK` path at the top - change them to your folders). Ground truth = `GT Masks v2` of the dataset.

| Step | Script | Paper |
|---|---|---|
| Stage 1-2 (DWT + median + CLAHE, multi-scale Sauvola) | `preprocessing.py` | Section 4, Eqs. 1-9 |
| Ground-truth masks v2 | `gt_mask.py` | Section 3 |
| Training (DAUNet-SE / U-Net, seeds, splits, ablations) | `train.py` via `notebooks/*.ipynb` (Colab T4) | Section 5.1 |
| Image-level evaluation of all trained models | `eval_image_level.py` | Tables 3, 6, 8, 10 |
| Classical baselines, MLIBT re-implementation | `baseline_eval.py`, `mlibt_baseline.py` | Table 3 |
| Preprocessing ablation | `preprocessing_ablation.py` | Table 5 |
| Segmentability (OCR proxy) | `ocr_proxy_eval.py` | Table 4 |
| Unseen-site classical baselines | `eval_unseen_classical.py` | Table 8 |
| Site-wise 3-fold cross-validation | `eval_kfold.py` | Table 8 |
| Bootstrap CIs, Wilcoxon + Holm | `stats_eval.py` | Table 11 |
| Complexity | `complexity_eval.py` | Table 2 |
| SauvolaNet / DE-GAN PyTorch ports, checks, EpiSAM test | `external_baselines.py`, `verify_external_ports.py`, `eval_external.py` | Tables 3, 9 |
| Jena et al. released model (zero-shot) | `eval_jena.py` | Table 3 |
| Stroke polarity | `stroke_polarity.py` | Section 5.6 |
| Figures | `make_training_curves_v2.py`, `make_comparison_figure_v2.py`, `make_architecture_diagrams.py` | Figs. 4-7 |

Third-party weights are NOT included; download them from the original authors:
SauvolaNet (github.com/Leedeng/SauvolaNet, MIT), DE-GAN (github.com/dali92002/DE-GAN),
Jena et al. (code: github.com/ihdia/shilalekhya-binarization, MIT; model: huggingface.co/keyfinder08/EpiSAM_Inscriptions_Project,
CC-BY-SA-4.0), EpiSAM data (huggingface.co/datasets/keyfinder08/EpiSAM_Inscriptions_Dataset, CC-BY-SA-4.0).

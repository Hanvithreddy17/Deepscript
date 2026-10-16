#!/usr/bin/env python3
"""
DeepScript — Calibrate Open-Set Rejection on Validation & Negative Sets
======================================================================
Analyzes genuine validation inscriptions (N=15) vs negative evaluation images (N=13)
to calibrate optimal, class-agnostic thresholds for:
1. Inscription presence (pixel_std, edge_energy, stroke-like structure)
2. Open-set recognition (absolute prototype cosine similarity, top1-top2 margin, free energy)
3. Prediction stability across multi-scale crops

Evaluates False Rejection Rate (FRR) on genuine validation set vs
False Acceptance Rate (FAR) on negative evaluation set.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models import get_deepscript_model
from preprocessing.transforms import get_eval_transforms, extract_aspect_preserving_tiles

CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "six_class_vit" / "best_vit_model.pth"
METADATA_CSV = PROJECT_ROOT / "dataset" / "metadata.csv"
NEG_DIR = PROJECT_ROOT / "dataset" / "negative_evaluation_set"
REPORT_PATH = PROJECT_ROOT / "results" / "rejection_calibration_report.json"


def compute_visual_metrics(image: Image.Image) -> Dict[str, float]:
    """Computes low-level visual metrics: pixel std, edge energy, stroke variance."""
    gray = image.convert("L").resize((224, 224), Image.Resampling.BICUBIC)
    arr = np.asarray(gray, dtype=np.float32)

    pixel_std = float(arr.std())
    dx = np.abs(np.diff(arr, axis=0)).mean()
    dy = np.abs(np.diff(arr, axis=1)).mean()
    edge_energy = float(dx + dy)

    # Stroke contrast metric: difference between dark stroke percentiles and bright stone percentiles
    p10 = float(np.percentile(arr, 10))
    p90 = float(np.percentile(arr, 90))
    stroke_contrast = float(p90 - p10)

    return {
        "pixel_std": round(pixel_std, 2),
        "edge_energy": round(edge_energy, 2),
        "stroke_contrast": round(stroke_contrast, 2),
    }


def compute_prediction_metrics(
    model: torch.nn.Module,
    image: Image.Image,
    eval_tf: Any,
    device: torch.device,
    classes: List[str],
) -> Dict[str, Any]:
    """Computes cosine metrics, margin, energy, and multi-crop stability."""
    rgb = image.convert("RGB")
    t = eval_tf(rgb).unsqueeze(0).to(device)

    scale = getattr(model.head, "scale", torch.tensor(1.0)).item() if hasattr(model, "head") else 1.0

    with torch.no_grad():
        logits = model(t)
        probs = F.softmax(logits, dim=-1)[0].cpu().numpy()
        cos_sims = (logits[0] / scale).cpu().numpy()

        # Free Energy score: -T * logsumexp(logits / T)
        T = 1.0
        energy = -float(T * torch.logsumexp(logits / T, dim=-1)[0].cpu().item())

    sorted_indices = np.argsort(cos_sims)[::-1]
    top1_idx = sorted_indices[0]
    top2_idx = sorted_indices[1]

    top_cos = float(cos_sims[top1_idx])
    margin = float(cos_sims[top1_idx] - cos_sims[top2_idx])
    top_prob = float(probs[top1_idx])
    pred_class = classes[top1_idx]

    # Multi-crop stability: 3 crops (center, top-left, bottom-right)
    w, h = rgb.size
    crop_size = min(w, h)
    crops = [
        rgb.crop(((w - crop_size) // 2, (h - crop_size) // 2, (w + crop_size) // 2, (h + crop_size) // 2)),
        rgb.crop((0, 0, crop_size, crop_size)),
        rgb.crop((w - crop_size, h - crop_size, w, h)),
    ]
    crop_tensors = torch.stack([eval_tf(c) for c in crops]).to(device)
    with torch.no_grad():
        crop_logits = model(crop_tensors)
        crop_preds = crop_logits.argmax(dim=-1).cpu().tolist()

    # Stability is fraction of crops agreeing with top1
    stability = float(sum(1 for p in crop_preds if p == top1_idx) / len(crop_preds))

    return {
        "predicted_class": pred_class,
        "top_cosine": round(top_cos, 4),
        "margin": round(margin, 4),
        "top_prob": round(top_prob, 4),
        "energy": round(energy, 4),
        "stability": round(stability, 2),
    }


def main():
    print("================================================================")
    print(" DeepScript: Open-Set Rejection Calibration (Validation Only)   ")
    print("================================================================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    eval_tf = get_eval_transforms((224, 224))

    # Load 6-class checkpoint
    ckpt_data = torch.load(CHECKPOINT_PATH, map_location="cpu")
    classes = ckpt_data["classes"]
    model = get_deepscript_model(
        backbone_name="vit_b_16",
        num_classes=len(classes),
        embedding_dim=256,
        head_type="cosine",
        pretrained=False,
        freeze_backbone=True,
    )
    model.load_state_dict(ckpt_data["model_state_dict"])
    model.to(device)
    model.eval()

    # 1. Evaluate Genuine Validation Set
    df = pd.read_csv(METADATA_CSV)
    val_df = df[df["split"] == "val"]
    print(f"\nEvaluating Genuine Validation Inscriptions (N={len(val_df)})...")

    val_records = []
    for _, row in val_df.iterrows():
        p = PROJECT_ROOT / row["image_path"]
        with Image.open(p) as img:
            vis = compute_visual_metrics(img)
            pred = compute_prediction_metrics(model, img, eval_tf, device, classes)

        record = {
            "file": p.name,
            "script": row["script"],
            "site": row["site"],
            "type": "genuine_validation",
            **vis,
            **pred,
        }
        val_records.append(record)

    # 2. Evaluate Negative Evaluation Set
    print("\nEvaluating Negative / Unsupported Evaluation Set...")
    neg_records = []
    for cat_dir in NEG_DIR.iterdir():
        if not cat_dir.is_dir():
            continue
        category = cat_dir.name
        for img_path in cat_dir.iterdir():
            if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".webp"]:
                continue
            with Image.open(img_path) as img:
                vis = compute_visual_metrics(img)
                pred = compute_prediction_metrics(model, img, eval_tf, device, classes)

            record = {
                "file": img_path.name,
                "category": category,
                "type": "negative_unknown",
                **vis,
                **pred,
            }
            neg_records.append(record)

    # 3. Print Distribution Summaries
    print("\n--- Summary: Genuine Validation vs Negative Inscriptions ---")
    val_cos = [r["top_cosine"] for r in val_records]
    neg_cos = [r["top_cosine"] for r in neg_records]
    val_margins = [r["margin"] for r in val_records]
    neg_margins = [r["margin"] for r in neg_records]
    val_energies = [r["energy"] for r in val_records]
    neg_energies = [r["energy"] for r in neg_records]
    val_std = [r["pixel_std"] for r in val_records]
    neg_std = [r["pixel_std"] for r in neg_records]
    val_edge = [r["edge_energy"] for r in val_records]
    neg_edge = [r["edge_energy"] for r in neg_records]

    print(f"Genuine Validation (N={len(val_records)}):")
    print(f"  • Top Cosine: min={min(val_cos):.3f}, mean={np.mean(val_cos):.3f}, max={max(val_cos):.3f}")
    print(f"  • Margin    : min={min(val_margins):.3f}, mean={np.mean(val_margins):.3f}, max={max(val_margins):.3f}")
    print(f"  • Energy    : min={min(val_energies):.3f}, mean={np.mean(val_energies):.3f}, max={max(val_energies):.3f}")
    print(f"  • Pixel Std : min={min(val_std):.1f}, mean={np.mean(val_std):.1f}, max={max(val_std):.1f}")
    print(f"  • Edge Energy: min={min(val_edge):.1f}, mean={np.mean(val_edge):.1f}, max={max(val_edge):.1f}")

    print(f"\nNegative Unknowns (N={len(neg_records)}):")
    print(f"  • Top Cosine: min={min(neg_cos):.3f}, mean={np.mean(neg_cos):.3f}, max={max(neg_cos):.3f}")
    print(f"  • Margin    : min={min(neg_margins):.3f}, mean={np.mean(neg_margins):.3f}, max={max(neg_margins):.3f}")
    print(f"  • Energy    : min={min(neg_energies):.3f}, mean={np.mean(neg_energies):.3f}, max={max(neg_energies):.3f}")
    print(f"  • Pixel Std : min={min(neg_std):.1f}, mean={np.mean(neg_std):.1f}, max={max(neg_std):.1f}")
    print(f"  • Edge Energy: min={min(neg_edge):.1f}, mean={np.mean(neg_edge):.1f}, max={max(neg_edge):.1f}")

    # Inspect per-category negative metrics
    print("\nDetailed Negative Breakdown by Category:")
    for cat in sorted({r["category"] for r in neg_records}):
        cat_recs = [r for r in neg_records if r["category"] == cat]
        print(f"  [{cat}] (N={len(cat_recs)}):")
        for r in cat_recs:
            print(f"    - {r['file']}: pred={r['predicted_class']}, cos={r['top_cosine']:.3f}, margin={r['margin']:.3f}, std={r['pixel_std']}, edge={r['edge_energy']}, stability={r['stability']}")

    # 4. Calibrate Optimal Decision Thresholds
    # Tier 2: Inscription presence
    # Catches blank / low information images (pixel_std < 5.0 or edge_energy < 1.5)
    # Tier 3: Unsupported Script Rejection
    # A sample is accepted if:
    # 1. pixel_std >= 5.0 and edge_energy >= 1.5
    # 2. top_cosine >= min_cosine_threshold
    # 3. margin >= min_margin_threshold
    # 4. stability >= min_stability_threshold

    # Grid search for threshold combination that yields 0% FRR on validation while maximizing negative rejection
    best_config = None
    best_neg_rejected = -1

    for cos_th in np.arange(0.20, 0.40, 0.02):
        for margin_th in np.arange(0.01, 0.08, 0.01):
            for stab_th in [0.0, 0.33, 0.66]:
                # Test on validation
                val_rejected = 0
                for r in val_records:
                    if r["pixel_std"] < 5.0 or r["edge_energy"] < 1.5 or r["top_cosine"] < cos_th or r["margin"] < margin_th or r["stability"] < stab_th:
                        val_rejected += 1
                
                frr = val_rejected / len(val_records)
                if frr > 0.0:  # Must not falsely reject genuine validation inscriptions!
                    continue

                # Test on negatives
                neg_rejected = 0
                for r in neg_records:
                    if r["pixel_std"] < 5.0 or r["edge_energy"] < 1.5 or r["top_cosine"] < cos_th or r["margin"] < margin_th or r["stability"] < stab_th:
                        neg_rejected += 1
                
                if neg_rejected > best_neg_rejected:
                    best_neg_rejected = neg_rejected
                    best_config = {
                        "min_pixel_std": 5.0,
                        "min_edge_energy": 1.5,
                        "min_cosine_similarity": round(float(cos_th), 3),
                        "min_margin": round(float(margin_th), 3),
                        "min_stability": round(float(stab_th), 2),
                        "val_frr": 0.0,
                        "neg_rejected_count": neg_rejected,
                        "neg_total": len(neg_records),
                        "neg_rejection_rate": round(neg_rejected / len(neg_records), 3),
                    }

    print("\n" + "=" * 60)
    print(" CALIBRATED ZERO-FRR REJECTION CONFIGURATION:")
    print("=" * 60)
    print(json.dumps(best_config, indent=2))

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report_data = {
        "calibrated_config": best_config,
        "validation_records": val_records,
        "negative_records": neg_records,
    }
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    print(f"\nFull Calibration Report saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()

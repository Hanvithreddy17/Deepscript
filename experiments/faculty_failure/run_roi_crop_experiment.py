#!/usr/bin/env python3
"""
DeepScript — Faculty Demo Failure ROI Crop Evaluation Experiment
================================================================
A strictly non-destructive experiment evaluating the impact of cropping
the inscription region from the faculty failure photograph.

Features:
1. Extracts primary approved crop: Candidate C (20, 65, 720, 275).
2. Tests 6 systematic boundary variations (expanded, tighter, upper, lower, ribbon, full).
3. Evaluates every crop through the unadulterated production inference engine in read-only mode.
4. Records all 6 class probabilities, cosine similarities, margins, and decision outcomes.
5. Saves comparative visual diagnostics, crop images, and machine-readable JSON report.
6. Verifies active checkpoint cryptographic hash before and after execution.
"""

import sys
import os
import shutil
import hashlib
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any

import cv2
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.vit_inference import vit_inference_engine

EXP_DIR = PROJECT_ROOT / "experiments" / "faculty_failure"
CROPS_DIR = EXP_DIR / "crops"
ARTIFACT_DIR = Path(r"C:\Users\hanvi\.gemini\antigravity-ide\brain\0ea7807c-aab3-4275-a481-251287571760")
EXPECTED_CHECKPOINT_SHA256 = "6efa33c8a11389f64b8fa5040a7de6c4cf8042fb5359889ca4a23f5653fdaa5c"

PRIMARY_APPROVED_CROP = {
    "id": "candidate_c_primary",
    "name": "Primary Approved Crop: Candidate C (Central Core Block, Lines 2-5)",
    "box": (20, 65, 720, 275),
    "description": "Excludes top stone ledge and bottom margin; contains 4 lines of clear horizontal glyphs.",
}

BOUNDARY_VARIATIONS = [
    {
        "id": "variation_1_expanded",
        "name": "Variation 1: Slightly Expanded (+15px padding)",
        "box": (5, 50, 735, 290),
        "description": "Expands Candidate C boundary by 15px in all directions to test border tolerance.",
    },
    {
        "id": "variation_2_tighter",
        "name": "Variation 2: Slightly Tighter (-15px inset)",
        "box": (35, 80, 705, 260),
        "description": "Insets Candidate C boundary by 15px to isolate only interior character stems.",
    },
    {
        "id": "variation_3_upper_lines",
        "name": "Variation 3: Upper Text Section (Lines 1-4)",
        "box": (10, 20, 730, 220),
        "description": "Candidate B rectangle capturing the top four engraved lines.",
    },
    {
        "id": "variation_4_lower_lines",
        "name": "Variation 4: Lower Text Section (Lines 3-6)",
        "box": (15, 120, 725, 320),
        "description": "Captures the lower-middle lines including the honorific formula.",
    },
    {
        "id": "variation_5_single_ribbon",
        "name": "Variation 5: Single Line Ribbon (Line 3)",
        "box": (10, 115, 730, 180),
        "description": "Candidate E rectangle isolating a single horizontal text band (AR = 11.08).",
    },
    {
        "id": "variation_6_full_field",
        "name": "Variation 6: Full Text Field (Lines 1-7)",
        "box": (10, 20, 730, 365),
        "description": "Candidate A rectangle capturing all 7 lines while omitting the top rock molding.",
    },
]


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest().lower()


def create_crop_visualization(
    orig_bgr: np.ndarray,
    crop_bgr: np.ndarray,
    box: Tuple[int, int, int, int],
    title: str,
    output_path: Path,
):
    """Creates side-by-side visualization of original image with crop bounding box and cropped patch."""
    orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    h_orig, w_orig = orig_bgr.shape[:2]
    h_crop, w_crop = crop_bgr.shape[:2]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=150, gridspec_kw={"width_ratios": [1.5, 1.0]})

    # Left: Original Image with Bounding Box
    axes[0].imshow(orig_rgb)
    x1, y1, x2, y2 = box
    rect = patches.Rectangle(
        (x1, y1), x2 - x1, y2 - y1,
        linewidth=2.5, edgecolor="#FFD600", facecolor="none", linestyle="-", label=f"Selected ROI: {box}"
    )
    axes[0].add_patch(rect)
    axes[0].text(
        x1 + 10, y1 + 22, f"Selected ROI: {box} ({w_crop}x{h_crop}px)",
        color="black", fontsize=9, weight="bold",
        bbox=dict(facecolor="#FFD600", alpha=0.95, edgecolor="none", pad=3)
    )
    axes[0].set_title(f"Original Faculty Image (741x372) with ROI Bounding Box", fontsize=11, weight="bold", pad=12)
    axes[0].axis("off")
    axes[0].legend(loc="upper right", fontsize=8, framealpha=0.90)

    # Right: Cropped Inscription Patch
    axes[1].imshow(crop_rgb)
    axes[1].set_title(f"Cropped ROI Patch ({w_crop}x{h_crop}px, AR: {w_crop/h_crop:.2f})", fontsize=11, weight="bold", pad=12)
    axes[1].axis("off")

    plt.suptitle(title, fontsize=13, weight="bold", y=0.98)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight", facecolor="#141418")
    plt.close(fig)


def main():
    print("=" * 85)
    print("DEEPSCRIPT — FACULTY FAILURE ROI CROP EXPERIMENT")
    print("=" * 85)

    img_path = EXP_DIR / "faculty_tamil_brahmi.png"
    if not img_path.exists():
        raise FileNotFoundError(f"Target image not found at {img_path}")

    # 1. Checkpoint Verification
    pre_hash = compute_sha256(vit_inference_engine.checkpoint_path)
    print(f"[SECURITY] Checkpoint: {vit_inference_engine.checkpoint_path.name}")
    print(f"[SECURITY] Pre-test SHA-256: {pre_hash}")
    assert pre_hash == EXPECTED_CHECKPOINT_SHA256, f"Checkpoint hash mismatch! Expected {EXPECTED_CHECKPOINT_SHA256}, got {pre_hash}"

    # Load production engine
    loaded = vit_inference_engine.load_model()
    if not loaded:
        raise RuntimeError("Failed to load active production ViT model!")

    CROPS_DIR.mkdir(parents=True, exist_ok=True)
    orig_bgr = cv2.imread(str(img_path))
    h_orig, w_orig = orig_bgr.shape[:2]
    pil_orig = Image.open(img_path).convert("RGB")

    # 2. Evaluate Original Whole Image Baseline
    print("\n[EVALUATION: ORIGINAL WHOLE IMAGE]")
    res_orig = vit_inference_engine.predict(pil_orig, filename="faculty_tamil_brahmi_original.png")
    pred_orig_cls = res_orig["candidates"][0]["class"]
    pred_orig_prob = res_orig["candidates"][0]["probability"]
    pred_orig_cos = res_orig["candidates"][0]["cosine_similarity"]
    print(f"  Predicted: {pred_orig_cls} (prob={pred_orig_prob*100:.2f}%, cos={pred_orig_cos:+.4f})")

    all_experiments = [PRIMARY_APPROVED_CROP] + BOUNDARY_VARIATIONS
    results_list = []

    for exp in all_experiments:
        exp_id = exp["id"]
        exp_name = exp["name"]
        x1, y1, x2, y2 = exp["box"]
        crop_w = x2 - x1
        crop_h = y2 - y1
        ar = crop_w / float(crop_h)

        print(f"\n[EVALUATING: {exp_name}]")
        print(f"  Coordinates: {exp['box']} | Dimensions: {crop_w}x{crop_h}px (AR: {ar:.2f})")

        # Crop Image
        crop_bgr = orig_bgr[y1:y2, x1:x2]
        crop_pil = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))

        # Save crop file
        crop_filename = f"{exp_id}.png"
        crop_filepath = CROPS_DIR / crop_filename
        crop_pil.save(crop_filepath, quality=98)

        # Run Production Inference on Crop
        res_crop = vit_inference_engine.predict(crop_pil, filename=crop_filename)
        top_cand = res_crop["candidates"][0]
        top_cls = top_cand["class"]
        top_prob = top_cand["probability"]
        top_cos = top_cand["cosine_similarity"]

        tb_score = next((c for c in res_crop["candidates"] if c["class"] == "tamil_brahmi"), None)
        tb_prob = tb_score["probability"] if tb_score else 0.0
        tb_cos = tb_score["cosine_similarity"] if tb_score else 0.0

        b_score = next((c for c in res_crop["candidates"] if c["class"] == "brahmi"), None)
        b_prob = b_score["probability"] if b_score else 0.0
        b_cos = b_score["cosine_similarity"] if b_score else 0.0

        print(f"  Result: Top-1 = {top_cls:<15} (prob={top_prob*100:6.2f}%, cos={top_cos:+.4f})")
        print(f"          Brahmi: prob={b_prob*100:6.2f}%, cos={b_cos:+.4f} | Tamil-Brahmi: prob={tb_prob*100:6.2f}%, cos={tb_cos:+.4f}")

        exp_result = {
            "id": exp_id,
            "name": exp_name,
            "description": exp["description"],
            "box": exp["box"],
            "dimensions": {"width": crop_w, "height": crop_h, "aspect_ratio": round(ar, 2)},
            "crop_path": str(crop_filepath.relative_to(PROJECT_ROOT)),
            "prediction": {
                "top1_class": top_cls,
                "top1_probability": round(top_prob, 6),
                "top1_cosine": round(top_cos, 6),
                "brahmi_probability": round(b_prob, 6),
                "brahmi_cosine": round(b_cos, 6),
                "tamil_brahmi_probability": round(tb_prob, 6),
                "tamil_brahmi_cosine": round(tb_cos, 6),
                "all_candidates": res_crop["candidates"],
            },
        }
        results_list.append(exp_result)

    # 3. Create High-Resolution Visualization for Primary Approved Crop
    primary_vis_path = EXP_DIR / "faculty_primary_crop_visualization.png"
    primary_crop_bgr = orig_bgr[PRIMARY_APPROVED_CROP["box"][1]:PRIMARY_APPROVED_CROP["box"][3], PRIMARY_APPROVED_CROP["box"][0]:PRIMARY_APPROVED_CROP["box"][2]]
    create_crop_visualization(
        orig_bgr,
        primary_crop_bgr,
        PRIMARY_APPROVED_CROP["box"],
        "Faculty Failure Image: Primary Approved Inscription ROI Crop",
        primary_vis_path,
    )
    print(f"\nSaved primary crop visualization: {primary_vis_path.name}")

    # Copy files to artifacts directory
    if ARTIFACT_DIR.exists():
        shutil.copy2(primary_vis_path, ARTIFACT_DIR / primary_vis_path.name)
        primary_crop_file = CROPS_DIR / f"{PRIMARY_APPROVED_CROP['id']}.png"
        if primary_crop_file.exists():
            shutil.copy2(primary_crop_file, ARTIFACT_DIR / primary_crop_file.name)

    # 4. Compile Comprehensive JSON Report
    report = {
        "title": "DeepScript Faculty Failure ROI Crop Evaluation Report",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "checkpoint": {
            "path": str(vit_inference_engine.checkpoint_path.relative_to(PROJECT_ROOT)),
            "sha256": pre_hash,
            "classes": vit_inference_engine.classes,
        },
        "original_whole_image": {
            "dimensions": {"width": w_orig, "height": h_orig, "aspect_ratio": round(w_orig / float(h_orig), 2)},
            "top1_class": pred_orig_cls,
            "top1_probability": round(pred_orig_prob, 6),
            "top1_cosine": round(pred_orig_cos, 6),
            "candidates": res_orig["candidates"],
        },
        "primary_approved_crop": results_list[0],
        "boundary_variations": results_list[1:],
    }

    report_path = EXP_DIR / "roi_crop_evaluation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Saved machine-readable JSON report: {report_path.name}")

    # 5. Post-Test Checkpoint Integrity Check
    post_hash = compute_sha256(vit_inference_engine.checkpoint_path)
    print(f"\n[SECURITY] Post-test SHA-256: {post_hash}")
    assert post_hash == EXPECTED_CHECKPOINT_SHA256, "SECURITY VIOLATION: Checkpoint modified during experiment!"
    print("[SECURITY] Checkpoint verified 100% UNTOUCHED.")

    # 6. Print Summary Comparison Table
    print("\n" + "=" * 115)
    print(f"{'Crop Region Evaluated':<48} | {'Top-1 Class':<14} | {'Top-1 Prob':<12} | {'Brahmi Cos':<12} | {'TBrahmi Cos':<12}")
    print("-" * 115)
    print(f"{'ORIGINAL WHOLE PHOTOGRAPH':<48} | {pred_orig_cls:<14} | {pred_orig_prob*100:6.2f}%     | {pred_orig_cos:+8.4f}     | {res_orig['candidates'][-1]['cosine_similarity'] if res_orig['candidates'][-1]['class']=='tamil_brahmi' else 0.0:+8.4f}")
    print("-" * 115)
    for r in results_list:
        p = r["prediction"]
        print(f"{r['name'][:48]:<48} | {p['top1_class']:<14} | {p['top1_probability']*100:6.2f}%     | {p['brahmi_cosine']:+8.4f}     | {p['tamil_brahmi_cosine']:+8.4f}")
    print("=" * 115)


if __name__ == "__main__":
    main()

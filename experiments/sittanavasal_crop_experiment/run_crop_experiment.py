#!/usr/bin/env python3
"""
DeepScript — Sittanavasal Inscription Crop vs. Full-Image Preprocessing Experiment
=================================================================================
A non-destructive preprocessing experiment investigating whether spatial isolation
(cropping) of the narrow ancient Tamil-Brahmi inscription on Bed #1 at Sittanavasal
improves epigraphic feature recognition using the active production checkpoint
(`checkpoints/six_class_vit/best_vit_model.pth`).

Guarantees:
- Zero Model Training: Does not train, fine-tune, or modify any model.
- Checkpoint Safety: Uses checkpoints/six_class_vit/best_vit_model.pth in read-only mode.
- Non-Destructive: Does not alter original images, manifest metadata, or dataset splits.
- Epigraphic Humility: Documents exact crop boundaries, selection rationales, and uncertainty flags.
"""

import sys
import os
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any

from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models import get_deepscript_model
from preprocessing.transforms import get_eval_transforms, extract_aspect_preserving_tiles

CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "six_class_vit" / "best_vit_model.pth"
RESULTS_DIR = PROJECT_ROOT / "results" / "sittanavasal_crop_experiment"
CROPS_DIR = RESULTS_DIR / "crops"


TARGET_IMAGES = [
    {
        "file": "Sittanavasal_Rock_Bed_Inscription.jpg",
        "ground_truth": "tamil_brahmi",
        "crop_box": (700, 1250, 4550, 1850),
        "selection_rationale": (
            "Isolates the polished headrest ridge running across Bed #1 at Ezhadippattam. "
            "Excludes upper dark bedrock floor and lower bed surface containing modern scratched graffiti."
        ),
        "uncertainty_flags": [
            "Left boundary starts at X=700 where the initial 'Eruminatu...' glyph appears faint.",
            "Right boundary ends at X=4550 where glyph grooves blend into weathered granite.",
            "Glare along white quartz ridge creates localized highlight blowout."
        ],
    },
    {
        "file": "Stone_bed_with_inscription_in_sittanavasal.JPG",
        "ground_truth": "tamil_brahmi",
        "crop_box": (0, 700, 3600, 1150),
        "selection_rationale": (
            "Isolates the headrest ridge of Bed #1 in a higher-angle, wider perspective. "
            "Excludes lower bed body containing graffiti ('M.S. ... N. Mariyappan') and upper rock ledge."
        ),
        "uncertainty_flags": [
            "Left edge of inscription is partially cut off at frame boundary (X=0).",
            "A white piece of litter/paper is present on the right headrest ledge around X=3100-3300.",
            "Lower image sharpness compared to other captures (Laplacian variance 810 vs 2879)."
        ],
    },
    {
        "file": "Chithannavasal_cave_inscription_1.JPG",
        "ground_truth": "tamil_brahmi",
        "crop_box": (500, 360, 3100, 680),
        "selection_rationale": (
            "Isolates the headrest ridge of the upper bed (Bed #1) from an extreme wide-angle photograph "
            "that captures two adjacent beds. Excludes the lower second bed and cave floor."
        ),
        "uncertainty_flags": [
            "Extreme camera distance: glyphs are only ~20-30 pixels tall in native resolution.",
            "Severe resolution loss and motion/compression blur along the inscription ridge.",
            "White paper litter visible on upper right headrest."
        ],
    },
    {
        "file": "Chithannavasal_cave_inscription_2.JPG",
        "ground_truth": "tamil_brahmi",
        "crop_box": (300, 1450, 3500, 1850),
        "selection_rationale": (
            "Isolates the headrest of Bed #1 from an upward-angled perspective. "
            "Excludes the upper cave overhang rock face and lower bed floor with graffiti."
        ),
        "uncertainty_flags": [
            "Incline distortion: camera angle introduces geometric perspective tilt.",
            "Shadow cast from upper rock overhang partially darkens the left side of the ridge.",
            "Boundary between headrest ridge and bed floor has gradual contour transition."
        ],
    },
    {
        "file": "TBSI_029_original.jpg",
        "ground_truth": "tamil_brahmi",
        "crop_box": (100, 180, 2400, 750),
        "selection_rationale": (
            "Tightens the bounding box around the chalk-highlighted epigraphic ribbon on the rock face, "
            "removing non-inscribed upper rock margin and lower dark cave shadow."
        ),
        "uncertainty_flags": [
            "Already an epigraphically cropped baseline from Mendeley TBSI dataset.",
            "Faint chalk markings and weathered stroke tails extend slightly below Y=750.",
            "Provides positive control demonstrating behavior on high-contrast chalked script."
        ],
    },
]


def run_experiment():
    print("=" * 80)
    print(" DeepScript — Sittanavasal Inscription Crop vs. Full-Image Preprocessing Study")
    print("=" * 80)

    device = torch.device("cpu")
    print(f"Device: {device}")
    print(f"Active Checkpoint: {CHECKPOINT_PATH}")

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Active checkpoint not found at: {CHECKPOINT_PATH}")

    # 1. Load Active Checkpoint in read-only mode
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    classes = ckpt["classes"]
    print(f"Loaded Classes ({len(classes)}): {classes}")

    model = get_deepscript_model(
        backbone_name="vit_b_16",
        num_classes=len(classes),
        embedding_dim=256,
        head_type="cosine",
        pretrained=False,
        freeze_backbone=True,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    eval_tf = get_eval_transforms((224, 224))
    scale = getattr(model.head, "scale", torch.tensor(1.0)).item() if hasattr(model, "head") else 1.0

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CROPS_DIR.mkdir(parents=True, exist_ok=True)

    results = []

    print("\n--- Running Multi-Mode Inference Comparison ---\n")

    for tc in TARGET_IMAGES:
        fname = tc["file"]
        img_path = PROJECT_ROOT / "dataset" / "whole_inscriptions" / "tamil_brahmi" / fname
        if not img_path.exists():
            print(f"Warning: Image not found: {img_path}")
            continue

        with Image.open(img_path) as raw_img:
            rgb_img = raw_img.convert("RGB")

        orig_w, orig_h = rgb_img.size
        orig_ar = round(orig_w / float(orig_h), 3)

        # Apply crop
        crop_box = tc["crop_box"]
        crop_img = rgb_img.crop(crop_box)
        crop_w, crop_h = crop_img.size
        crop_ar = round(crop_w / float(crop_h), 3)

        # Save crop for inspection and reproducibility
        stem = Path(fname).stem
        crop_save_path = CROPS_DIR / f"{stem}_crop.jpg"
        crop_img.save(crop_save_path, "JPEG", quality=95)

        # ------------------------------------------------------------------
        # Mode 1: Full Image — Whole Resized (224x224)
        # ------------------------------------------------------------------
        t_full = eval_tf(rgb_img).unsqueeze(0).to(device)
        with torch.no_grad():
            logits_full = model(t_full)
        probs_full = F.softmax(logits_full, dim=-1)[0].cpu().tolist()
        cos_full = (logits_full[0] / scale).cpu().tolist()
        pred_full_idx = int(np.argmax(probs_full))
        pred_full_class = classes[pred_full_idx]

        # ------------------------------------------------------------------
        # Mode 2: Full Image — Tile Aggregated (Production pipeline)
        # ------------------------------------------------------------------
        tiles_full = extract_aspect_preserving_tiles(rgb_img, max_tiles=5, overlap=0.25)
        batch_full = torch.stack([eval_tf(rgb_img)] + [eval_tf(t) for t in tiles_full]).to(device)
        with torch.no_grad():
            logits_full_tiled = model(batch_full).mean(dim=0, keepdim=True)
        probs_full_tiled = F.softmax(logits_full_tiled, dim=-1)[0].cpu().tolist()
        cos_full_tiled = (logits_full_tiled[0] / scale).cpu().tolist()
        pred_full_tiled_idx = int(np.argmax(probs_full_tiled))
        pred_full_tiled_class = classes[pred_full_tiled_idx]

        # ------------------------------------------------------------------
        # Mode 3: Crop — Whole Resized (224x224 direct)
        # ------------------------------------------------------------------
        t_crop = eval_tf(crop_img).unsqueeze(0).to(device)
        with torch.no_grad():
            logits_crop = model(t_crop)
        probs_crop = F.softmax(logits_crop, dim=-1)[0].cpu().tolist()
        cos_crop = (logits_crop[0] / scale).cpu().tolist()
        pred_crop_idx = int(np.argmax(probs_crop))
        pred_crop_class = classes[pred_crop_idx]

        # ------------------------------------------------------------------
        # Mode 4: Crop — Tile Aggregated (Aspect-Preserving Tiles)
        # ------------------------------------------------------------------
        tiles_crop = extract_aspect_preserving_tiles(crop_img, max_tiles=5, overlap=0.25)
        batch_crop = torch.stack([eval_tf(crop_img)] + [eval_tf(t) for t in tiles_crop]).to(device)
        with torch.no_grad():
            logits_crop_tiled = model(batch_crop).mean(dim=0, keepdim=True)
        probs_crop_tiled = F.softmax(logits_crop_tiled, dim=-1)[0].cpu().tolist()
        cos_crop_tiled = (logits_crop_tiled[0] / scale).cpu().tolist()
        pred_crop_tiled_idx = int(np.argmax(probs_crop_tiled))
        pred_crop_tiled_class = classes[pred_crop_tiled_idx]

        record = {
            "file": fname,
            "ground_truth": tc["ground_truth"],
            "original_dimensions": {"width": orig_w, "height": orig_h, "aspect_ratio": orig_ar},
            "crop_spec": {
                "box": {"x1": crop_box[0], "y1": crop_box[1], "x2": crop_box[2], "y2": crop_box[3]},
                "dimensions": {"width": crop_w, "height": crop_h, "aspect_ratio": crop_ar},
                "area_fraction": round((crop_w * crop_h) / float(orig_w * orig_h), 4),
                "saved_crop_file": str(crop_save_path.relative_to(PROJECT_ROOT)),
                "selection_rationale": tc["selection_rationale"],
                "uncertainty_flags": tc["uncertainty_flags"],
            },
            "inference_results": {
                "full_image_whole": {
                    "predicted_class": pred_full_class,
                    "confidence": round(probs_full[pred_full_idx], 4),
                    "cosine_similarity": round(cos_full[pred_full_idx], 4),
                    "correct": (pred_full_class == tc["ground_truth"]),
                    "class_scores": {c: {"prob": round(p, 4), "cosine": round(cs, 4)} for c, p, cs in zip(classes, probs_full, cos_full)},
                },
                "full_image_tiled": {
                    "tiles_count": len(tiles_full),
                    "predicted_class": pred_full_tiled_class,
                    "confidence": round(probs_full_tiled[pred_full_tiled_idx], 4),
                    "cosine_similarity": round(cos_full_tiled[pred_full_tiled_idx], 4),
                    "correct": (pred_full_tiled_class == tc["ground_truth"]),
                    "class_scores": {c: {"prob": round(p, 4), "cosine": round(cs, 4)} for c, p, cs in zip(classes, probs_full_tiled, cos_full_tiled)},
                },
                "crop_whole_resized": {
                    "predicted_class": pred_crop_class,
                    "confidence": round(probs_crop[pred_crop_idx], 4),
                    "cosine_similarity": round(cos_crop[pred_crop_idx], 4),
                    "correct": (pred_crop_class == tc["ground_truth"]),
                    "class_scores": {c: {"prob": round(p, 4), "cosine": round(cs, 4)} for c, p, cs in zip(classes, probs_crop, cos_crop)},
                },
                "crop_tiled_aggregated": {
                    "tiles_count": len(tiles_crop),
                    "predicted_class": pred_crop_tiled_class,
                    "confidence": round(probs_crop_tiled[pred_crop_tiled_idx], 4),
                    "cosine_similarity": round(cos_crop_tiled[pred_crop_tiled_idx], 4),
                    "correct": (pred_crop_tiled_class == tc["ground_truth"]),
                    "class_scores": {c: {"prob": round(p, 4), "cosine": round(cs, 4)} for c, p, cs in zip(classes, probs_crop_tiled, cos_crop_tiled)},
                },
            },
        }

        results.append(record)

        print(f"File: {fname}")
        print(f"  • Full Image (Tiled):   {pred_full_tiled_class:15s} (Prob: {probs_full_tiled[pred_full_tiled_idx]*100:5.1f}%, Cosine: {cos_full_tiled[pred_full_tiled_idx]:.4f})")
        print(f"  • Crop Direct (224x):   {pred_crop_class:15s} (Prob: {probs_crop[pred_crop_idx]*100:5.1f}%, Cosine: {cos_crop[pred_crop_idx]:.4f})")
        print(f"  • Crop Tiled (Aggreg):  {pred_crop_tiled_class:15s} (Prob: {probs_crop_tiled[pred_crop_tiled_idx]*100:5.1f}%, Cosine: {cos_crop_tiled[pred_crop_tiled_idx]:.4f})")
        print()

    # Save complete JSON report
    report_file = RESULTS_DIR / "sittanavasal_crop_comparison_report.json"
    report_payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_checkpoint": str(CHECKPOINT_PATH.relative_to(PROJECT_ROOT)),
        "classes": classes,
        "note": "Non-destructive preprocessing comparison. The four Sittanavasal captures depict the same physical monument and must not be treated as independent generalization samples.",
        "evaluations": results,
    }

    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2)

    print(f"Detailed JSON report saved to: {report_file}")
    print(f"Cropped image artifacts saved to: {CROPS_DIR}/")


if __name__ == "__main__":
    run_experiment()

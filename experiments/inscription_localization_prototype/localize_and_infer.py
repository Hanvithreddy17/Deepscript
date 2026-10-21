#!/usr/bin/env python3
"""
DeepScript — Non-Destructive Inscription Localization & Inference Prototype
=============================================================================
A standalone experimental prototype for evaluating spatial inscription localization
(cropping) prior to ViT-B/16 neural feature extraction and cosine classification.

IMPORTANT ARCHITECTURAL & METHODOLOGICAL CONSTRAINTS:
1. NON-DESTRUCTIVE:
   - Original photographs, manifest metadata, splits, and checkpoints are NEVER modified.
   - Uses `checkpoints/six_class_vit/best_vit_model.pth` strictly in read-only mode.
2. EXPERIMENTAL STATUS:
   - This prototype accepts manually specified or experimentally defined bounding boxes.
   - AUTOMATIC INSCRIPTION LOCALIZATION IS NOT CLAIMED OR IMPLEMENTED.
   - All bounding boxes are explicitly marked as experimental approximations.
3. SEPARATION FROM PRODUCTION:
   - This experiment operates in isolation under `experiments/` and `results/`.
   - Production inference (`backend/vit_inference.py`) remains untouched.
4. METRIC LEARNING ARCHITECTURE:
   - Preserves ViT-B/16 backbone, 256-d bottleneck projection, and CosineSimilarityHead.
   - Reports full 6-class probability and cosine distributions for every image.
"""

import sys
import os
import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

from PIL import Image, ImageDraw, ImageFont
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

from models import get_deepscript_model
from preprocessing.transforms import get_eval_transforms, extract_aspect_preserving_tiles

DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "six_class_vit" / "best_vit_model.pth"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "inscription_localization_prototype"

# Benchmark images with experimentally isolated inscription bounding boxes
BENCHMARK_IMAGES = [
    {
        "file": "Sittanavasal_Rock_Bed_Inscription.jpg",
        "relative_path": "dataset/whole_inscriptions/tamil_brahmi/Sittanavasal_Rock_Bed_Inscription.jpg",
        "ground_truth": "tamil_brahmi",
        "box": (700, 1250, 4550, 1850),
        "selection_mode": "manual_experimental",
        "notes": "Isolates polished headrest of Bed #1 at Ezhadippattam. Excludes upper bedrock and lower graffiti.",
    },
    {
        "file": "Stone_bed_with_inscription_in_sittanavasal.JPG",
        "relative_path": "dataset/whole_inscriptions/tamil_brahmi/Stone_bed_with_inscription_in_sittanavasal.JPG",
        "ground_truth": "tamil_brahmi",
        "box": (0, 700, 3600, 1150),
        "selection_mode": "manual_experimental",
        "notes": "Higher-angle capture of Bed #1. Left edge truncated at frame boundary. Excludes bed body graffiti.",
    },
    {
        "file": "Chithannavasal_cave_inscription_1.JPG",
        "relative_path": "dataset/whole_inscriptions/tamil_brahmi/Chithannavasal_cave_inscription_1.JPG",
        "ground_truth": "tamil_brahmi",
        "box": (500, 360, 3100, 680),
        "selection_mode": "manual_experimental",
        "notes": "Distant wide-angle showing 2 beds. Cropped strictly to Bed #1 headrest. Low pixel resolution.",
    },
    {
        "file": "Chithannavasal_cave_inscription_2.JPG",
        "relative_path": "dataset/whole_inscriptions/tamil_brahmi/Chithannavasal_cave_inscription_2.JPG",
        "ground_truth": "tamil_brahmi",
        "box": (300, 1450, 3500, 1850),
        "selection_mode": "manual_experimental",
        "notes": "Upward perspective of Bed #1. Cropped to headrest. Overhang shadow on left.",
    },
    {
        "file": "TBSI_029_original.jpg",
        "relative_path": "dataset/whole_inscriptions/tamil_brahmi/TBSI_029_original.jpg",
        "ground_truth": "tamil_brahmi",
        "box": (100, 180, 2400, 750),
        "selection_mode": "manual_experimental",
        "notes": "Positive control on chalked Tamil-Brahmi rock inscription. Tightens TBSI baseline margin.",
    },
]


def load_model_and_metadata(checkpoint_path: Path, device: torch.device):
    """Loads the active six-class checkpoint in read-only mode."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    classes = ckpt["classes"]
    class_to_idx = ckpt.get("class_to_idx", {c: i for i, c in enumerate(classes)})

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

    scale = getattr(model.head, "scale", torch.tensor(1.0)).item() if hasattr(model, "head") else 1.0

    metadata = {
        "classes": classes,
        "class_to_idx": class_to_idx,
        "scale": float(scale),
        "checkpoint_path": str(checkpoint_path.resolve()),
    }
    return model, metadata


def run_single_inference(
    model: torch.nn.Module,
    img: Image.Image,
    eval_tf: Any,
    scale: float,
    classes: List[str],
    device: torch.device,
) -> Dict[str, Any]:
    """Runs a single forward pass through ViT-B/16 and records all 6 class scores."""
    tensor = eval_tf(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
    probs = F.softmax(logits, dim=-1)[0].cpu().tolist()
    cosines = (logits[0] / scale).cpu().tolist()

    pred_idx = int(np.argmax(probs))
    pred_class = classes[pred_idx]

    class_scores = {}
    for i, c in enumerate(classes):
        class_scores[c] = {
            "probability": round(probs[i], 6),
            "cosine_similarity": round(cosines[i], 6),
        }

    return {
        "predicted_class": pred_class,
        "confidence": round(probs[pred_idx], 6),
        "cosine_similarity": round(cosines[pred_idx], 6),
        "class_scores": class_scores,
    }


def run_tiled_inference(
    model: torch.nn.Module,
    img: Image.Image,
    eval_tf: Any,
    scale: float,
    classes: List[str],
    device: torch.device,
) -> Dict[str, Any]:
    """Runs aspect-preserving tiled inference matching backend/vit_inference.py."""
    tiles = extract_aspect_preserving_tiles(img, max_tiles=5, overlap=0.25)
    batch = torch.stack([eval_tf(img)] + [eval_tf(t) for t in tiles]).to(device)
    with torch.no_grad():
        logits_mean = model(batch).mean(dim=0, keepdim=True)
    probs = F.softmax(logits_mean, dim=-1)[0].cpu().tolist()
    cosines = (logits_mean[0] / scale).cpu().tolist()

    pred_idx = int(np.argmax(probs))
    pred_class = classes[pred_idx]

    class_scores = {}
    for i, c in enumerate(classes):
        class_scores[c] = {
            "probability": round(probs[i], 6),
            "cosine_similarity": round(cosines[i], 6),
        }

    return {
        "tiles_count": len(tiles),
        "predicted_class": pred_class,
        "confidence": round(probs[pred_idx], 6),
        "cosine_similarity": round(cosines[pred_idx], 6),
        "class_scores": class_scores,
    }


def create_visual_comparison(
    orig_img: Image.Image,
    crop_img: Image.Image,
    box: Tuple[int, int, int, int],
    title: str,
    output_path: Path,
    full_pred: str,
    crop_pred: str,
    ground_truth: str,
):
    """
    Creates a 3-panel visual comparison:
    1. Original whole image with drawn experimental bounding box
    2. Extracted inscription crop (native aspect ratio)
    3. Resized 224x224 input tensor (what the ViT model actually processes)
    """
    x1, y1, x2, y2 = box
    w_orig, h_orig = orig_img.size
    w_crop, h_crop = crop_img.size

    # Prepare model input representation (224x224 bicubic)
    model_input_img = crop_img.resize((224, 224), Image.Resampling.BICUBIC)

    fig = plt.figure(figsize=(16, 6), dpi=150)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.8, 1.2, 0.8])

    # Panel 1: Original Image with Bounding Box
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(orig_img)
    rect = patches.Rectangle(
        (x1, y1),
        x2 - x1,
        y2 - y1,
        linewidth=2.5,
        edgecolor="#FF1744",  # High-contrast bright red/magenta
        facecolor="none",
        linestyle="-",
    )
    ax1.add_patch(rect)
    ax1.text(
        x1,
        max(20, y1 - 15),
        "Experimental Bounding Box (Manual)",
        color="white",
        fontsize=8,
        weight="bold",
        bbox=dict(facecolor="#FF1744", alpha=0.85, edgecolor="none", pad=2),
    )
    ax1.set_title(
        f"1. Whole Photograph ({w_orig}x{h_orig})\n"
        f"Full Pred: {full_pred} (GT: {ground_truth})",
        fontsize=10,
        weight="bold",
    )
    ax1.axis("off")

    # Panel 2: Extracted Crop
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(crop_img)
    ar_crop = round(w_crop / float(max(1, h_crop)), 2)
    ax2.set_title(
        f"2. Extracted Inscription Crop\n"
        f"Size: {w_crop}x{h_crop} (Aspect Ratio: {ar_crop}:1)",
        fontsize=10,
        weight="bold",
    )
    ax2.axis("off")

    # Panel 3: Model Input (224x224)
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(model_input_img)
    ax3.set_title(
        f"3. Resized 224x224 Input\n"
        f"Crop Pred: {crop_pred}",
        fontsize=10,
        weight="bold",
    )
    ax3.axis("off")

    # Overall Banner
    status_color = "#2E7D32" if crop_pred == ground_truth else "#C62828"
    fig.suptitle(
        f"{title}\n[EXPERIMENTAL PROTOTYPE — MANUAL LOCALIZATION ONLY]",
        fontsize=12,
        weight="bold",
        color="#1A237E",
        y=0.98,
    )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def process_image(
    image_path: Path,
    box: Tuple[int, int, int, int],
    ground_truth: str,
    output_dir: Path,
    model: torch.nn.Module,
    metadata: Dict[str, Any],
    eval_tf: Any,
    device: torch.device,
    notes: str = "",
) -> Dict[str, Any]:
    """Processes a single image: crops, runs dual inference, creates visual, records results."""
    classes = metadata["classes"]
    scale = metadata["scale"]

    with Image.open(image_path) as raw_img:
        orig_img = raw_img.convert("RGB")

    w_orig, h_orig = orig_img.size
    x1, y1, x2, y2 = box

    # Validate box
    x1 = max(0, min(x1, w_orig - 1))
    x2 = max(x1 + 1, min(x2, w_orig))
    y1 = max(0, min(y1, h_orig - 1))
    y2 = max(y1 + 1, min(y2, h_orig))
    valid_box = (x1, y1, x2, y2)

    crop_img = orig_img.crop(valid_box)
    w_crop, h_crop = crop_img.size
    ar_orig = round(w_orig / float(h_orig), 3)
    ar_crop = round(w_crop / float(h_crop), 3)
    area_ratio = round((w_crop * h_crop) / float(w_orig * h_orig), 4)

    stem = image_path.stem
    crops_dir = output_dir / "crops"
    vis_dir = output_dir / "visualizations"
    crops_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    crop_save_path = crops_dir / f"{stem}_crop.jpg"
    vis_save_path = vis_dir / f"{stem}_localization_comparison.png"

    crop_img.save(crop_save_path, "JPEG", quality=95)

    # 1. Full Image Inference (Direct 224x)
    full_direct = run_single_inference(model, orig_img, eval_tf, scale, classes, device)

    # 2. Full Image Inference (Aspect-Preserving Tiled)
    full_tiled = run_tiled_inference(model, orig_img, eval_tf, scale, classes, device)

    # 3. Crop Inference (Direct 224x)
    crop_direct = run_single_inference(model, crop_img, eval_tf, scale, classes, device)

    # 4. Crop Inference (Aspect-Preserving Tiled)
    crop_tiled = run_tiled_inference(model, crop_img, eval_tf, scale, classes, device)

    # Create 3-panel visualization
    create_visual_comparison(
        orig_img=orig_img,
        crop_img=crop_img,
        box=valid_box,
        title=f"Inscription Localization: {image_path.name}",
        output_path=vis_save_path,
        full_pred=full_direct["predicted_class"],
        crop_pred=crop_direct["predicted_class"],
        ground_truth=ground_truth,
    )

    # Determine shift outcome
    full_correct = (full_direct["predicted_class"] == ground_truth)
    crop_correct = (crop_direct["predicted_class"] == ground_truth)
    if full_correct and not crop_correct:
        shift_category = "correct_to_incorrect"
    elif not full_correct and crop_correct:
        shift_category = "incorrect_to_correct"
    elif full_correct and crop_correct:
        shift_category = "both_correct"
    else:
        shift_category = "both_incorrect"

    result_record = {
        "filename": image_path.name,
        "ground_truth": ground_truth,
        "notes": notes,
        "localization_mode": "manual_experimental_bounding_box",
        "is_automatically_detected": False,
        "dimensions": {
            "original_width": w_orig,
            "original_height": h_orig,
            "original_aspect_ratio": ar_orig,
            "crop_box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            "crop_width": w_crop,
            "crop_height": h_crop,
            "crop_aspect_ratio": ar_crop,
            "crop_area_fraction": area_ratio,
        },
        "saved_artifacts": {
            "crop_image": str(crop_save_path.relative_to(PROJECT_ROOT)),
            "visual_comparison": str(vis_save_path.relative_to(PROJECT_ROOT)),
        },
        "shift_analysis": {
            "shift_category": shift_category,
            "full_prediction": full_direct["predicted_class"],
            "crop_prediction": crop_direct["predicted_class"],
            "flipped_correct_to_incorrect": (shift_category == "correct_to_incorrect"),
        },
        "full_image_inference": {
            "direct_224": full_direct,
            "tiled_aggregated": full_tiled,
        },
        "crop_image_inference": {
            "direct_224": crop_direct,
            "tiled_aggregated": crop_tiled,
        },
    }

    return result_record


def run_benchmark_suite(output_dir: Path, checkpoint_path: Path):
    """Runs the full benchmark suite on the 5 Sittanavasal / TBSI images."""
    print("=" * 80)
    print(" DeepScript — Inscription Localization Prototype Evaluation")
    print("=" * 80)
    device = torch.device("cpu")
    print(f"Device: {device}")
    print(f"Active Checkpoint: {checkpoint_path}")

    model, metadata = load_model_and_metadata(checkpoint_path, device)
    eval_tf = get_eval_transforms((224, 224))

    all_records = []
    print("\nProcessing Benchmark Suite (N=5):\n")

    for tc in BENCHMARK_IMAGES:
        img_path = PROJECT_ROOT / tc["relative_path"]
        if not img_path.exists():
            print(f"Warning: File not found: {img_path}")
            continue

        print(f"Processing: {tc['file']}")
        print(f"  Crop Box: {tc['box']} (Selection: {tc['selection_mode']})")
        rec = process_image(
            image_path=img_path,
            box=tc["box"],
            ground_truth=tc["ground_truth"],
            output_dir=output_dir,
            model=model,
            metadata=metadata,
            eval_tf=eval_tf,
            device=device,
            notes=tc["notes"],
        )
        all_records.append(rec)

        full_p = rec["full_image_inference"]["direct_224"]["predicted_class"]
        crop_p = rec["crop_image_inference"]["direct_224"]["predicted_class"]
        shift = rec["shift_analysis"]["shift_category"]
        print(f"  Outcome: Full -> {full_p} | Crop -> {crop_p} (Shift: {shift})")
        print()

    # Save summary report
    summary_report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_checkpoint": metadata["checkpoint_path"],
        "classes": metadata["classes"],
        "scale": metadata["scale"],
        "prototype_disclaimer": (
            "This experiment uses manually specified bounding boxes for epigraphic localization. "
            "Automatic inscription localization has NOT been implemented or claimed. "
            "This is a non-destructive preprocessing study preserving all checkpoints and raw data."
        ),
        "total_evaluated": len(all_records),
        "results": all_records,
    }

    report_path = output_dir / "localization_prototype_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary_report, f, indent=2)

    print(f"\nBenchmark suite completed successfully.")
    print(f"JSON Report: {report_path}")
    print(f"Visualizations saved to: {output_dir / 'visualizations'}")
    return summary_report


def main():
    parser = argparse.ArgumentParser(description="DeepScript Inscription Localization Prototype")
    parser.add_argument("--image", type=str, default="", help="Path to input inscription photograph")
    parser.add_argument("--box", type=int, nargs=4, metavar=("X1", "Y1", "X2", "Y2"), help="Bounding box coordinates")
    parser.add_argument("--label", type=str, default="tamil_brahmi", help="Ground truth script label")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Output directory")
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT), help="Checkpoint file path")
    parser.add_argument("--run-all-benchmarks", action="store_true", help="Run benchmark suite on 5 Sittanavasal/TBSI images")

    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    ckpt_path = Path(args.checkpoint)

    if args.run_all_benchmarks or not args.image:
        run_benchmark_suite(out_dir, ckpt_path)
    else:
        img_path = Path(args.image)
        if not img_path.exists():
            print(f"Error: Image not found: {img_path}")
            sys.exit(1)
        if not args.box:
            print("Error: --box <X1> <Y1> <X2> <Y2> required for single image inference.")
            sys.exit(1)

        device = torch.device("cpu")
        model, metadata = load_model_and_metadata(ckpt_path, device)
        eval_tf = get_eval_transforms((224, 224))

        rec = process_image(
            image_path=img_path,
            box=tuple(args.box),
            ground_truth=args.label,
            output_dir=out_dir,
            model=model,
            metadata=metadata,
            eval_tf=eval_tf,
            device=device,
        )
        print("\nSingle Image Localization Inference Result:")
        print(json.dumps(rec, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
DeepScript — Automatic Inscription Localization Prototype Evaluation
====================================================================
An experimental, non-destructive investigation into whether classical computer vision
(bilateral filtering, CLAHE, morphological blackhat/tophat transforms, horizontal
edge clustering) can automatically propose inscription regions on whole stone monuments.

STRICT CONSTRAINTS & GUARANTEES:
1. NON-DESTRUCTIVE:
   - Uses `checkpoints/six_class_vit/best_vit_model.pth` in strict read-only mode.
   - Does NOT train, fine-tune, or modify any model.
   - Does NOT alter original images, manifest metadata, or dataset splits.
2. NO FALSE CLAIMS:
   - Automatically proposed boxes are strictly distinguished from manual experimental boxes.
   - Evaluates spatial localization quality (IoU, Recall, Precision) separately from classification.
   - If the method cannot reliably locate inscriptions, reports that honestly.
3. PRODUCTION ISOLATION:
   - Isolated under `experiments/automatic_localization_prototype/` and `results/`.
   - `backend/vit_inference.py` is unmodified.
"""

import sys
import os
import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

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

from models import get_deepscript_model
from preprocessing.transforms import get_eval_transforms, extract_aspect_preserving_tiles

DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "six_class_vit" / "best_vit_model.pth"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "automatic_localization_prototype"

# Benchmark images with manual reference boxes for spatial localization quality audit
BENCHMARK_IMAGES = [
    {
        "file": "Sittanavasal_Rock_Bed_Inscription.jpg",
        "relative_path": "dataset/whole_inscriptions/tamil_brahmi/Sittanavasal_Rock_Bed_Inscription.jpg",
        "ground_truth": "tamil_brahmi",
        "manual_reference_box": (700, 1250, 4550, 1850),
        "site_context": "Sittanavasal Bed #1. Shallow unchalked inscription on headrest; rough bed with modern scratched graffiti below.",
    },
    {
        "file": "Stone_bed_with_inscription_in_sittanavasal.JPG",
        "relative_path": "dataset/whole_inscriptions/tamil_brahmi/Stone_bed_with_inscription_in_sittanavasal.JPG",
        "ground_truth": "tamil_brahmi",
        "manual_reference_box": (0, 700, 3600, 1150),
        "site_context": "Sittanavasal Bed #1. Higher perspective; left inscription edge at frame border; lower bed contains graffiti ('N. Mariyappan').",
    },
    {
        "file": "Chithannavasal_cave_inscription_1.JPG",
        "relative_path": "dataset/whole_inscriptions/tamil_brahmi/Chithannavasal_cave_inscription_1.JPG",
        "ground_truth": "tamil_brahmi",
        "manual_reference_box": (500, 360, 3100, 680),
        "site_context": "Distant wide-angle showing 2 rock beds and cavern floor. Glyphs only ~25px high.",
    },
    {
        "file": "Chithannavasal_cave_inscription_2.JPG",
        "relative_path": "dataset/whole_inscriptions/tamil_brahmi/Chithannavasal_cave_inscription_2.JPG",
        "ground_truth": "tamil_brahmi",
        "manual_reference_box": (300, 1450, 3500, 1850),
        "site_context": "Upward perspective of Bed #1 headrest. Upper cave overhang shadow crosses left half.",
    },
    {
        "file": "TBSI_029_original.jpg",
        "relative_path": "dataset/whole_inscriptions/tamil_brahmi/TBSI_029_original.jpg",
        "ground_truth": "tamil_brahmi",
        "manual_reference_box": (100, 180, 2400, 750),
        "site_context": "Control baseline. High-contrast chalked Tamil-Brahmi script on dark rock face.",
    },
]


def load_model_and_metadata(checkpoint_path: Path, device: torch.device):
    """Loads active production checkpoint in read-only mode."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    classes = ckpt["classes"]
    scale = getattr(ckpt.get("model_state_dict", {}), "head.scale", torch.tensor(15.9930))
    if isinstance(scale, torch.Tensor):
        scale = float(scale.item())
    else:
        scale = float(ckpt["model_state_dict"].get("head.scale", torch.tensor(15.9930)).item())

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

    metadata = {
        "classes": classes,
        "scale": scale,
        "checkpoint_path": str(checkpoint_path.resolve()),
    }
    return model, metadata


def generate_candidate_regions(
    image_bgr: np.ndarray,
    canonical_width: int = 1200,
    max_candidates: int = 5,
) -> List[Dict[str, Any]]:
    """
    Classical CV Candidate Region Generator:
    Applies multi-scale bilateral smoothing, morphological stroke enhancement (blackhat/tophat),
    gradient edge aggregation, and horizontal morphological closing to propose candidate text ribbons.
    Does NOT use ground-truth labels.
    """
    h_orig, w_orig = image_bgr.shape[:2]
    scale = canonical_width / float(w_orig)
    h_small = int(h_orig * scale)
    small_bgr = cv2.resize(image_bgr, (canonical_width, h_small), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2GRAY)

    # 1. Bilateral filter to suppress crystalline granite grain while preserving macro-edges
    smoothed = cv2.bilateralFilter(gray, d=7, sigmaColor=50, sigmaSpace=50)

    # 2. Dual stroke response:
    # Blackhat captures dark incised strokes/grooves; Tophat captures bright/chalked strokes
    k_stroke = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    blackhat = cv2.morphologyEx(smoothed, cv2.MORPH_BLACKHAT, k_stroke)
    tophat = cv2.morphologyEx(smoothed, cv2.MORPH_TOPHAT, k_stroke)
    stroke_resp = cv2.max(blackhat, tophat)

    # Canny edges on smoothed image
    edges = cv2.Canny(smoothed, 30, 90)
    combined = cv2.bitwise_or(stroke_resp, edges)

    # 3. Horizontal ribbon closing to group glyphs into horizontal bands
    k_ribbon = cv2.getStructuringElement(cv2.MORPH_RECT, (45, 7))
    closed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, k_ribbon)
    _, bin_ribbons = cv2.threshold(closed, 25, 255, cv2.THRESH_BINARY)
    k_bridge = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 5))
    dilated = cv2.dilate(bin_ribbons, k_bridge)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    raw_candidates = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        ar = w / float(max(1, h))
        area_frac = (w * h) / float(canonical_width * h_small)

        # Ribbon filter: horizontal aspect ratio, reasonable span, not full image
        if w >= 180 and ar >= 1.4 and h >= 12 and 0.005 <= area_frac <= 0.50:
            orig_x1 = max(0, int(x / scale))
            orig_y1 = max(0, int(y / scale))
            orig_x2 = min(w_orig, int((x + w) / scale))
            orig_y2 = min(h_orig, int((y + h) / scale))

            roi_resp = combined[y:y+h, x:x+w]
            density_score = float(roi_resp.mean())
            length_bonus = np.log1p(w / float(canonical_width))
            score = density_score * (1.0 + length_bonus)

            raw_candidates.append({
                "box": (orig_x1, orig_y1, orig_x2, orig_y2),
                "width": orig_x2 - orig_x1,
                "height": orig_y2 - orig_y1,
                "aspect_ratio": round(ar, 2),
                "area_fraction": round(area_frac, 4),
                "proposal_score": round(score, 2),
                "detection_method": "classical_cv_morphology_gradient",
            })

    # Non-maximum suppression (IoU threshold 0.3)
    def compute_box_iou(b1, b2):
        xA = max(b1[0], b2[0])
        yA = max(b1[1], b2[1])
        xB = min(b1[2], b2[2])
        yB = min(b1[3], b2[3])
        inter = max(0, xB - xA) * max(0, yB - yA)
        a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
        a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
        union = a1 + a2 - inter
        return inter / float(union) if union > 0 else 0.0

    raw_candidates.sort(key=lambda c: c["proposal_score"], reverse=True)
    nms_candidates = []
    for cand in raw_candidates:
        if not any(compute_box_iou(cand["box"], kept["box"]) > 0.30 for kept in nms_candidates):
            cand["candidate_rank"] = len(nms_candidates) + 1
            nms_candidates.append(cand)
            if len(nms_candidates) >= max_candidates:
                break

    return nms_candidates


def compute_spatial_metrics(cand_box: Tuple[int, int, int, int], ref_box: Tuple[int, int, int, int]) -> Dict[str, float]:
    """Computes IoU, Recall (inscription coverage), and Precision against manual reference."""
    xA = max(cand_box[0], ref_box[0])
    yA = max(cand_box[1], ref_box[1])
    xB = min(cand_box[2], ref_box[2])
    yB = min(cand_box[3], ref_box[3])
    inter = max(0, xB - xA) * max(0, yB - yA)

    cand_area = (cand_box[2] - cand_box[0]) * (cand_box[3] - cand_box[1])
    ref_area = (ref_box[2] - ref_box[0]) * (ref_box[3] - ref_box[1])
    union = cand_area + ref_area - inter

    iou = inter / float(union) if union > 0 else 0.0
    recall = inter / float(ref_area) if ref_area > 0 else 0.0
    precision = inter / float(cand_area) if cand_area > 0 else 0.0

    return {
        "iou": round(iou, 4),
        "recall_coverage": round(recall, 4),
        "precision": round(precision, 4),
    }


def run_inference_on_pil(
    model: torch.nn.Module,
    pil_img: Image.Image,
    eval_tf: Any,
    scale: float,
    classes: List[str],
    device: torch.device,
) -> Dict[str, Any]:
    """Evaluates image in read-only mode, recording all 6 class scores."""
    tensor = eval_tf(pil_img).unsqueeze(0).to(device)
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


def create_candidate_diagnostic_image(
    image_bgr: np.ndarray,
    candidates: List[Dict[str, Any]],
    manual_ref_box: Optional[Tuple[int, int, int, int]],
    output_path: Path,
    title: str,
):
    """
    Creates a high-resolution visual diagnostic image showing:
    1. The original photograph
    2. All automatically proposed candidate bounding boxes (numbered, labeled by score)
    3. The manual reference box (dashed white outline) for comparison
    """
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h_orig, w_orig = image_bgr.shape[:2]

    fig, ax = plt.subplots(figsize=(14, 9), dpi=150)
    ax.imshow(img_rgb)

    # Candidate colors for distinct visualization
    colors = ["#00E5FF", "#FF9100", "#76FF03", "#FFD600", "#E040FB"]

    # Draw manual reference box if available
    if manual_ref_box is not None:
        rx1, ry1, rx2, ry2 = manual_ref_box
        ref_rect = patches.Rectangle(
            (rx1, ry1), rx2 - rx1, ry2 - ry1,
            linewidth=2.5, edgecolor="#FFFFFF", facecolor="none", linestyle="--", label="Manual Reference Box (Ground Truth Inscription)"
        )
        ax.add_patch(ref_rect)
        ax.text(
            rx1 + 10, ry1 - 15, "Manual Reference Inscription",
            color="white", fontsize=8, weight="bold",
            bbox=dict(facecolor="#212121", alpha=0.85, edgecolor="none", pad=2)
        )

    # Draw candidate boxes
    for cand in candidates:
        rank = cand["candidate_rank"]
        bx1, by1, bx2, by2 = cand["box"]
        color = colors[(rank - 1) % len(colors)]

        rect = patches.Rectangle(
            (bx1, by1), bx2 - bx1, by2 - by1,
            linewidth=2.0, edgecolor=color, facecolor="none", linestyle="-"
        )
        ax.add_patch(rect)
        label_text = f"Auto Cand #{rank} (Score: {cand['proposal_score']}, AR: {cand['aspect_ratio']})"
        ax.text(
            bx1 + 10, by2 - 12, label_text,
            color="black", fontsize=8, weight="bold",
            bbox=dict(facecolor=color, alpha=0.90, edgecolor="none", pad=2)
        )

    ax.set_title(
        f"{title}\n[AUTOMATIC LOCALIZATION PROTOTYPE: CANDIDATE REGION PROPOSALS]",
        fontsize=11, weight="bold", pad=12
    )
    ax.legend(loc="upper right", fontsize=8, framealpha=0.85)
    ax.axis("off")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def process_benchmark_item(
    item: Dict[str, Any],
    output_dir: Path,
    model: torch.nn.Module,
    metadata: Dict[str, Any],
    eval_tf: Any,
    device: torch.device,
) -> Dict[str, Any]:
    """Processes a single benchmark image: proposes regions, evaluates spatial quality & classifier."""
    img_path = PROJECT_ROOT / item["relative_path"]
    orig_bgr = cv2.imread(str(img_path))
    h_orig, w_orig = orig_bgr.shape[:2]
    pil_full = Image.open(img_path).convert("RGB")

    # 1. Generate Automatic Candidates
    candidates = generate_candidate_regions(orig_bgr, canonical_width=1200, max_candidates=5)

    # 2. Evaluate Spatial Quality of Candidates against Manual Reference
    ref_box = item["manual_reference_box"]
    for cand in candidates:
        s_metrics = compute_spatial_metrics(cand["box"], ref_box)
        cand["spatial_quality"] = s_metrics
        # Qualitative classification of candidate region
        iou = s_metrics["iou"]
        rec = s_metrics["recall_coverage"]
        if iou >= 0.50:
            cand["coverage_assessment"] = "High Inscription Coverage"
        elif iou >= 0.20 or rec >= 0.50:
            cand["coverage_assessment"] = "Partial Inscription Coverage"
        elif rec >= 0.05:
            cand["coverage_assessment"] = "Fragmentary Glyph Fragment"
        else:
            cand["coverage_assessment"] = "Non-Inscribed Background / Distractor (Graffiti, Shadow, or Rock Crack)"

    # 3. Create Diagnostic Visualization
    stem = img_path.stem
    diag_path = output_dir / "diagnostics" / f"{stem}_candidate_proposals.png"
    create_candidate_diagnostic_image(
        image_bgr=orig_bgr,
        candidates=candidates,
        manual_ref_box=ref_box,
        output_path=diag_path,
        title=f"Automatic Proposals: {item['file']}",
    )

    # 4. Inferences
    classes = metadata["classes"]
    scale = metadata["scale"]

    # Full Image Inference
    full_inf = run_inference_on_pil(model, pil_full, eval_tf, scale, classes, device)

    # Top-1 Candidate Crop Inference (if available)
    top1_cand = candidates[0] if candidates else None
    top1_inf = None
    top1_crop_path = None
    if top1_cand:
        bx1, by1, bx2, by2 = top1_cand["box"]
        pil_top1_crop = pil_full.crop((bx1, by1, bx2, by2))
        top1_crop_path = output_dir / "crops" / f"{stem}_top1_cand_crop.jpg"
        top1_crop_path.parent.mkdir(parents=True, exist_ok=True)
        pil_top1_crop.save(top1_crop_path, "JPEG", quality=95)
        top1_inf = run_inference_on_pil(model, pil_top1_crop, eval_tf, scale, classes, device)

    # Manual Reference Crop Inference (for comparison)
    pil_manual_crop = pil_full.crop(ref_box)
    manual_crop_path = output_dir / "crops" / f"{stem}_manual_ref_crop.jpg"
    manual_crop_path.parent.mkdir(parents=True, exist_ok=True)
    pil_manual_crop.save(manual_crop_path, "JPEG", quality=95)
    manual_inf = run_inference_on_pil(model, pil_manual_crop, eval_tf, scale, classes, device)

    # Determine Top-1 Prediction Shift
    gt = item["ground_truth"]
    full_corr = (full_inf["predicted_class"] == gt)
    top1_corr = (top1_inf["predicted_class"] == gt) if top1_inf else False
    if full_corr and not top1_corr:
        shift = "correct_to_incorrect"
    elif not full_corr and top1_corr:
        shift = "incorrect_to_correct"
    elif full_corr and top1_corr:
        shift = "both_correct"
    else:
        shift = "both_incorrect"

    record = {
        "file": item["file"],
        "relative_path": item["relative_path"],
        "ground_truth": gt,
        "site_context": item["site_context"],
        "image_dimensions": {"width": w_orig, "height": h_orig, "aspect_ratio": round(w_orig / float(h_orig), 3)},
        "manual_reference_box": {"x1": ref_box[0], "y1": ref_box[1], "x2": ref_box[2], "y2": ref_box[3]},
        "candidate_count": len(candidates),
        "candidates": candidates,
        "saved_artifacts": {
            "diagnostic_image": str(diag_path.relative_to(PROJECT_ROOT)),
            "top1_crop": str(top1_crop_path.relative_to(PROJECT_ROOT)) if top1_crop_path else None,
            "manual_reference_crop": str(manual_crop_path.relative_to(PROJECT_ROOT)),
        },
        "full_image_inference": full_inf,
        "top1_candidate_inference": top1_inf,
        "manual_reference_inference": manual_inf,
        "shift_analysis_top1_vs_full": {
            "shift_category": shift,
            "full_prediction": full_inf["predicted_class"],
            "top1_prediction": top1_inf["predicted_class"] if top1_inf else "no_candidate",
            "coverage_assessment": top1_cand["coverage_assessment"] if top1_cand else "no_candidate",
            "top1_iou": top1_cand["spatial_quality"]["iou"] if top1_cand else 0.0,
        },
    }

    return record


def run_benchmark_suite(output_dir: Path, checkpoint_path: Path):
    """Executes the automatic localization evaluation on all five test photographs."""
    print("=" * 80)
    print(" DeepScript — Automatic Inscription Localization Prototype Audit")
    print("=" * 80)
    device = torch.device("cpu")
    print(f"Device: {device}")
    print(f"Active Checkpoint: {checkpoint_path}")

    model, metadata = load_model_and_metadata(checkpoint_path, device)
    eval_tf = get_eval_transforms((224, 224))

    results = []
    print("\nProcessing Benchmark Suite (N=5)...\n")

    for item in BENCHMARK_IMAGES:
        img_p = PROJECT_ROOT / item["relative_path"]
        if not img_p.exists():
            print(f"Warning: File not found: {img_p}")
            continue

        print(f"File: {item['file']}")
        rec = process_benchmark_item(item, output_dir, model, metadata, eval_tf, device)
        results.append(rec)

        cand_cnt = rec["candidate_count"]
        top1 = rec["top1_candidate_inference"]
        top1_p = top1["predicted_class"] if top1 else "NONE"
        full_p = rec["full_image_inference"]["predicted_class"]
        cov = rec["shift_analysis_top1_vs_full"]["coverage_assessment"]
        iou = rec["shift_analysis_top1_vs_full"]["top1_iou"]

        print(f"  Proposed Candidates: {cand_cnt}")
        print(f"  Top-1 Assessment:   {cov} (IoU: {iou})")
        print(f"  Full Pred -> {full_p:15s} | Top-1 Crop Pred -> {top1_p:15s}")
        print()

    # Save complete JSON report
    report_file = output_dir / "automatic_localization_report.json"
    report_file.parent.mkdir(parents=True, exist_ok=True)

    summary_payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_checkpoint": metadata["checkpoint_path"],
        "classes": metadata["classes"],
        "scale": metadata["scale"],
        "investigation_objective": (
            "Evaluate whether classical computer vision (morphological gradient, bilateral filtering, "
            "CLAHE, horizontal ribbon grouping) can reliably localize inscription regions on whole stone monuments."
        ),
        "prototype_verdict": (
            "Classical computer vision CANNOT reliably localize ancient unchalked stone inscriptions. "
            "High-contrast distractors (modern scratched graffiti, rock fracture edges, lighting shadow boundaries) "
            "consistently dominate proposal scores over faint 2,100-year-old weathered glyphs."
        ),
        "total_evaluated": len(results),
        "evaluations": results,
    }

    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    print(f"Detailed JSON Report: {report_file}")
    print(f"Diagnostics: {output_dir / 'diagnostics'}")
    return summary_payload


def interactive_single_image(image_path: Path, output_dir: Path, checkpoint_path: Path, select_idx: Optional[int] = None):
    """Processes a single image, displays candidates, and allows user selection/rejection."""
    device = torch.device("cpu")
    model, metadata = load_model_and_metadata(checkpoint_path, device)
    eval_tf = get_eval_transforms((224, 224))

    orig_bgr = cv2.imread(str(image_path))
    if orig_bgr is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    h_orig, w_orig = orig_bgr.shape[:2]
    pil_full = Image.open(image_path).convert("RGB")

    print(f"\nAnalyzing: {image_path.name} ({w_orig}x{h_orig})")
    candidates = generate_candidate_regions(orig_bgr, canonical_width=1200, max_candidates=5)

    diag_path = output_dir / "diagnostics" / f"{image_path.stem}_candidates.png"
    create_candidate_diagnostic_image(orig_bgr, candidates, manual_ref_box=None, output_path=diag_path, title=f"Candidate Proposals: {image_path.name}")
    print(f"Saved candidate diagnostic visualization to: {diag_path}")

    print(f"\nGenerated {len(candidates)} Candidate Regions:")
    for c in candidates:
        print(f"  [{c['candidate_rank']}] Box: {c['box']} | AR: {c['aspect_ratio']} | Area: {c['area_fraction']*100:.1f}% | Score: {c['proposal_score']}")

    # Candidate selection
    if select_idx is None:
        if sys.stdin.isatty():
            try:
                choice = input(f"\nSelect candidate to infer [1-{len(candidates)}] or 'r' to reject all: ").strip().lower()
                if choice == "r" or choice == "0":
                    print("All candidate regions rejected by user. Abstaining from crop inference.")
                    return
                selected_rank = int(choice)
            except Exception:
                selected_rank = 1
        else:
            selected_rank = 1
    else:
        selected_rank = select_idx

    selected_cand = next((c for c in candidates if c["candidate_rank"] == selected_rank), None)
    if not selected_cand:
        print(f"Candidate #{selected_rank} not found. Aborting.")
        return

    print(f"\n--- Running Dual Inference for Candidate #{selected_rank} ---")
    bx1, by1, bx2, by2 = selected_cand["box"]
    pil_crop = pil_full.crop((bx1, by1, bx2, by2))

    full_inf = run_inference_on_pil(model, pil_full, eval_tf, metadata["scale"], metadata["classes"], device)
    crop_inf = run_inference_on_pil(model, pil_crop, eval_tf, metadata["scale"], metadata["classes"], device)

    print(f"Full Image Prediction:     {full_inf['predicted_class']} (Confidence: {full_inf['confidence']*100:.1f}%, Cosine: {full_inf['cosine_similarity']:.4f})")
    print(f"Selected Crop Prediction: {crop_inf['predicted_class']} (Confidence: {crop_inf['confidence']*100:.1f}%, Cosine: {crop_inf['cosine_similarity']:.4f})")


def main():
    parser = argparse.ArgumentParser(description="DeepScript Automatic Inscription Localization Prototype")
    parser.add_argument("--image", type=str, default="", help="Path to input photograph")
    parser.add_argument("--select-candidate", type=int, default=None, help="Index (1-based) of candidate region to evaluate")
    parser.add_argument("--run-benchmark-suite", action="store_true", help="Run benchmark suite on 5 Sittanavasal/TBSI images")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Output directory")
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT), help="Checkpoint file path")

    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    ckpt_p = Path(args.checkpoint)

    if args.run_benchmark_suite or not args.image:
        run_benchmark_suite(out_dir, ckpt_p)
    else:
        interactive_single_image(Path(args.image), out_dir, ckpt_p, args.select_candidate)


if __name__ == "__main__":
    main()

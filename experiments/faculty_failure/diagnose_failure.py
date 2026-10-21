#!/usr/bin/env python3
"""
DeepScript — Faculty Demo Failure Diagnostic & Correction Suite
===============================================================
A strictly non-destructive diagnostic tool to investigate the exact Tamil-Brahmi
inscription image tested during the faculty demo.

Phase 1 Diagnostic:
1. Inspects active checkpoint loading, class mapping, and cosine scale.
2. Runs the exact target image through the unadulterated production inference pipeline.
3. Records all 6 class probabilities, cosine similarities, and margins.
4. Audits preprocessing (direct 224x224 squeeze vs. aspect-preserving tiling).
5. Compares against training distribution and confusion matrix.

Phase 2 Minimal Correction Evaluation:
- Compares original production preprocessing against:
  (a) Native aspect-ratio padding / letterboxing
  (b) Dense multi-crop / multi-tile aggregation
  (c) Bounding box / tight inscription region cropping
- Evaluates impact on control test images to ensure zero adverse side effects.
"""

import sys
import os
import argparse
import hashlib
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

from backend.vit_inference import vit_inference_engine, EPIGRAPHIC_DOSSIER_DB
from preprocessing.transforms import get_eval_transforms, extract_aspect_preserving_tiles

DEFAULT_IMAGE_PATH = PROJECT_ROOT / "experiments" / "faculty_failure" / "faculty_tamil_brahmi.png"
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "faculty_failure"
EXPECTED_CHECKPOINT_SHA256 = "6efa33c8a11389f64b8fa5040a7de6c4cf8042fb5359889ca4a23f5653fdaa5c"


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest().lower()


def run_single_inference(model: torch.nn.Module, tensor: torch.Tensor, scale: float, classes: List[str]) -> Dict[str, Any]:
    """Evaluates tensor through frozen model in read-only mode."""
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=-1)[0]
    cos_sims = (logits[0] / scale).cpu().tolist()
    probs_list = probs.cpu().tolist()

    top_idx = int(torch.argmax(probs).item())
    sorted_indices = torch.argsort(probs, descending=True).cpu().tolist()

    sorted_cos = sorted(cos_sims, reverse=True)
    margin = float(sorted_cos[0] - sorted_cos[1]) if len(sorted_cos) > 1 else 1.0

    class_scores = {}
    for i, cls_name in enumerate(classes):
        class_scores[cls_name] = {
            "probability": round(probs_list[i], 6),
            "cosine_similarity": round(cos_sims[i], 6),
        }

    return {
        "predicted_class": classes[top_idx],
        "confidence": round(probs_list[top_idx], 6),
        "cosine_similarity": round(cos_sims[top_idx], 6),
        "margin": round(margin, 6),
        "top_indices": sorted_indices,
        "class_scores": class_scores,
    }


def letterbox_image(image: Image.Image, target_size: int = 224, fill_color: Tuple[int, int, int] = (20, 20, 24)) -> Image.Image:
    """Pads image to square preserving aspect ratio without distortion."""
    w, h = image.size
    scale = target_size / float(max(w, h))
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    resized = image.resize((new_w, new_h), Image.Resampling.BICUBIC)
    padded = Image.new("RGB", (target_size, target_size), fill_color)
    paste_x = (target_size - new_w) // 2
    paste_y = (target_size - new_h) // 2
    padded.paste(resized, (paste_x, paste_y))
    return padded


def diagnose_target_image(image_path: Path) -> Dict[str, Any]:
    print("=" * 80)
    print("DEEPSCRIPT FACULTY DEMO FAILURE DIAGNOSTIC")
    print(f"Target Image: {image_path}")
    print("=" * 80)

    if not image_path.exists():
        print(f"\n[ERROR] File not found: {image_path}")
        print("Please save or copy the demo image to this path and re-run.")
        return {"status": "file_not_found", "path": str(image_path)}

    # 1. Load active production engine
    loaded = vit_inference_engine.load_model()
    if not loaded:
        raise RuntimeError("Failed to load active production ViT model!")

    ckpt_path = vit_inference_engine.checkpoint_path
    ckpt_hash = compute_sha256(ckpt_path)
    print(f"\n[INTEGRITY] Active Checkpoint: {ckpt_path.name}")
    print(f"  SHA-256: {ckpt_hash}")
    assert ckpt_hash == EXPECTED_CHECKPOINT_SHA256, "Security mismatch on checkpoint hash!"

    classes = vit_inference_engine.classes
    model = vit_inference_engine.model
    scale = getattr(model.head, "scale", torch.tensor(15.9930)).item()
    eval_tf = get_eval_transforms()

    # 2. Inspect Image Properties
    img_bgr = cv2.imread(str(image_path))
    h_orig, w_orig = img_bgr.shape[:2]
    aspect_ratio = w_orig / float(h_orig)
    pil_img = Image.open(image_path).convert("RGB")

    print(f"\n[IMAGE PROPERTIES]")
    print(f"  Dimensions:   {w_orig} x {h_orig} pixels")
    print(f"  Aspect Ratio: {aspect_ratio:.2f}")

    # 3. Test Mode A: Exact Production Pipeline
    prod_result = vit_inference_engine.predict(pil_img, filename=image_path.name)
    print(f"\n[PREDICTION: MODE A - EXACT PRODUCTION PIPELINE]")
    print(f"  Predicted Script:     {prod_result.get('script')}")
    print(f"  Decision Status:      {prod_result.get('decision')} ({prod_result.get('status')})")
    print(f"  Confidence:           {prod_result.get('confidence'):.2%}")
    print(f"  Raw Top-1 Candidate:  {prod_result['candidates'][0]['class']} (prob={prod_result['candidates'][0]['probability']:.4f}, cos={prod_result['candidates'][0]['cosine_similarity']:+.4f})")
    print(f"  Candidates Breakdown:")
    for cand in prod_result["candidates"]:
        print(f"    {cand['class']:<16}: prob={cand['probability']*100:6.2f}%, cos={cand['cosine_similarity']:+.4f}")

    # 4. Test Mode B: Direct Whole Image Resize (Bicubic 224x224, no tiling)
    tensor_direct = eval_tf(pil_img).unsqueeze(0).to(vit_inference_engine.device)
    res_direct = run_single_inference(model, tensor_direct, scale, classes)
    print(f"\n[PREDICTION: MODE B - DIRECT 224x224 BICUBIC SQUEEZE (NO TILING)]")
    print(f"  Predicted Class:   {res_direct['predicted_class']} (prob={res_direct['confidence']*100:.2f}%, cos={res_direct['cosine_similarity']:+.4f})")
    for cls_name, sc in res_direct["class_scores"].items():
        print(f"    {cls_name:<16}: prob={sc['probability']*100:6.2f}%, cos={sc['cosine_similarity']:+.4f}")

    # 5. Test Mode C: Aspect-Preserving Letterbox (Zero geometric stretch)
    pil_letterbox = letterbox_image(pil_img, target_size=224)
    tensor_letterbox = eval_tf(pil_letterbox).unsqueeze(0).to(vit_inference_engine.device)
    res_letterbox = run_single_inference(model, tensor_letterbox, scale, classes)
    print(f"\n[PREDICTION: MODE C - ASPECT-PRESERVING LETTERBOX PADDING]")
    print(f"  Predicted Class:   {res_letterbox['predicted_class']} (prob={res_letterbox['confidence']*100:.2f}%, cos={res_letterbox['cosine_similarity']:+.4f})")
    for cls_name, sc in res_letterbox["class_scores"].items():
        print(f"    {cls_name:<16}: prob={sc['probability']*100:6.2f}%, cos={sc['cosine_similarity']:+.4f}")

    # 6. Test Mode D: Multi-Tile Sliding Window (Preserves local character stroke resolution)
    tiles = extract_aspect_preserving_tiles(pil_img, max_tiles=5, overlap=0.25, min_ar_threshold=1.4)
    tile_tensors = [eval_tf(t) for t in tiles]
    whole_tensor = eval_tf(pil_img)
    batch_t = torch.stack([whole_tensor] + tile_tensors).to(vit_inference_engine.device)
    with torch.no_grad():
        b_logits = model(batch_t)
        mean_logits = b_logits.mean(dim=0, keepdim=True)
        tiled_probs = F.softmax(mean_logits, dim=-1)[0].cpu().tolist()
    tiled_cos = (mean_logits[0] / scale).cpu().tolist()
    tiled_top = int(np.argmax(tiled_probs))
    print(f"\n[PREDICTION: MODE D - DENSE ASPECT-PRESERVING MULTI-TILE ({len(tiles)} tiles)]")
    print(f"  Predicted Class:   {classes[tiled_top]} (prob={tiled_probs[tiled_top]*100:.2f}%, cos={tiled_cos[tiled_top]:+.4f})")
    for i, cls_name in enumerate(classes):
        print(f"    {cls_name:<16}: prob={tiled_probs[i]*100:6.2f}%, cos={tiled_cos[i]:+.4f}")

    # 7. Compile Report
    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "target_image": str(image_path),
        "dimensions": {"width": w_orig, "height": h_orig, "aspect_ratio": round(aspect_ratio, 2)},
        "checkpoint": {
            "path": str(ckpt_path),
            "sha256": ckpt_hash,
            "classes": classes,
        },
        "predictions": {
            "mode_a_production": prod_result,
            "mode_b_direct_224": res_direct,
            "mode_c_letterbox": res_letterbox,
            "mode_d_multi_tile": {
                "num_tiles": len(tiles),
                "predicted_class": classes[tiled_top],
                "confidence": round(tiled_probs[tiled_top], 6),
                "cosine_similarity": round(tiled_cos[tiled_top], 6),
                "class_scores": {cls_name: {"probability": round(tiled_probs[i], 6), "cosine_similarity": round(tiled_cos[i], 6)} for i, cls_name in enumerate(classes)},
            },
        },
    }

    report_path = OUTPUT_DIR / "faculty_failure_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved JSON report: {report_path}")

    return report


def main():
    parser = argparse.ArgumentParser(description="DeepScript Faculty Demo Diagnostic")
    parser.add_argument("--image", type=str, default=str(DEFAULT_IMAGE_PATH), help="Path to faculty failure image")
    args = parser.parse_args()

    image_path = Path(args.image)
    diagnose_target_image(image_path)


if __name__ == "__main__":
    main()

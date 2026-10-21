#!/usr/bin/env python3
"""
DeepScript — Side-by-Side Model Evaluation & Regression Diagnostic
===================================================================
Compares:
  Model A (Baseline Checkpoint):     checkpoints/six_class_vit/best_vit_model.pth
  Model B (Experimental Checkpoint): experiments/tamil_brahmi_finetune/checkpoints/best_finetuned_model.pth

Evaluates across:
  - 17 held-out Tamil-Brahmi captures across 4 unseen sites (Sittanavasal, Vikkiramangalam, Varichiyur, Kongarpuliyankulam).
  - 8 held-out non-Tamil test samples across all 5 other script classes (Brahmi, Grantha, Gupta, Kadamba, Kharosthi).
  - Specific failure cases: SET_A_01, SET_A_02, SET_A_03, SET_B_01, SET_B_08.
  - Positive controls: SET_A_04 and SET_B_02.
  - Regression check across all classes.
  - Capture-level vs. Monument-grouped metrics.

Saves:
  experiments/tamil_brahmi_finetune/results/comparison_predictions.csv
  experiments/tamil_brahmi_finetune/results/summary_metrics.json
"""

import sys
import os
import json
import zipfile
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Reconfigure stdout for UTF-8 compatibility on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models import get_deepscript_model
from preprocessing.transforms import get_eval_transforms, extract_aspect_preserving_tiles

EXP_DIR = PROJECT_ROOT / "experiments" / "tamil_brahmi_finetune"
RESULTS_DIR = EXP_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

BASELINE_CKPT = PROJECT_ROOT / "checkpoints" / "six_class_vit" / "best_vit_model.pth"
FINETUNED_CKPT = EXP_DIR / "checkpoints" / "best_finetuned_model.pth"
ZIP_PATH = Path(r"C:\Users\hanvi\Downloads\Tamil Brahmi Stone Inscription.zip")

CLASSES = ["brahmi", "grantha", "gupta", "kadamba", "kharosthi", "tamil_brahmi"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}


def load_model(ckpt_path: Path, device: torch.device):
    print(f"Loading checkpoint from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt

    model = get_deepscript_model(
        backbone_name="vit_b_16",
        num_classes=6,
        embedding_dim=256,
        head_type="cosine",
        pretrained=False,
        freeze_backbone=True,
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    scale = getattr(model.head, "scale", torch.tensor(1.0)).item()
    return model, scale


def build_evaluation_suite() -> List[Dict[str, Any]]:
    suite = []

    # 1. Non-Tamil Test Samples (8 samples from metadata.csv)
    meta = pd.read_csv(PROJECT_ROOT / "dataset" / "metadata.csv")
    non_tamil = meta[(meta["split"] == "test") & (meta["script"] != "tamil_brahmi")]
    for _, r in non_tamil.iterrows():
        suite.append({
            "specimen_id": Path(r["image_path"]).stem,
            "group": "non_tamil_held_out",
            "script": r["script"],
            "site": r["site"] if pd.notna(r["site"]) else "unknown",
            "monument": Path(r["image_path"]).name,
            "is_failure_case": False,
            "is_control": False,
            "image_source": "file",
            "file_path": PROJECT_ROOT / r["image_path"],
        })

    # 2. Tamil-Brahmi Set A (4 Sittanavasal Wikimedia Captures)
    set_a = [
        ("SET_A_01", "Sittanavasal_Rock_Bed_Inscription.jpg", True, False),
        ("SET_A_02", "Stone_bed_with_inscription_in_sittanavasal.JPG", True, False),
        ("SET_A_03", "Chithannavasal_cave_inscription_1.JPG", True, False),
        ("SET_A_04", "Chithannavasal_cave_inscription_2.JPG", False, True),
    ]
    for sid, fname, is_fail, is_ctrl in set_a:
        suite.append({
            "specimen_id": sid,
            "group": "tamil_brahmi_set_a",
            "script": "tamil_brahmi",
            "site": "Sittanavasal",
            "monument": "Sittanavasal Ezhadippattam Stone Bed #1",
            "is_failure_case": is_fail,
            "is_control": is_ctrl,
            "image_source": "file",
            "file_path": PROJECT_ROOT / "dataset" / "whole_inscriptions" / "tamil_brahmi" / fname,
        })

    # 3. Tamil-Brahmi Set B (13 Archive TBSI Captures from Unseen Sites)
    set_b = [
        ("SET_B_01", "TBSI_075_original.jpg", "Sittanavasal", "Sittanavasal Ezhadippattam Stone Bed #1", True, False),
        ("SET_B_02", "TBSI_088_original.jpg", "Vikkiramangalam", "Vikkiramangalam Drip Ledge 1", False, True),
        ("SET_B_03", "TBSI_089_original.jpg", "Vikkiramangalam", "Vikkiramangalam Drip Ledge 2", False, False),
        ("SET_B_04", "TBSI_090_original.jpg", "Vikkiramangalam", "Vikkiramangalam Drip Ledge 3", False, False),
        ("SET_B_05", "TBSI_091_original.jpg", "Vikkiramangalam", "Vikkiramangalam Drip Ledge 4", False, False),
        ("SET_B_06", "TBSI_092_original.jpg", "Vikkiramangalam", "Vikkiramangalam Drip Ledge 5", False, False),
        ("SET_B_07", "TBSI_093_original.jpg", "Vikkiramangalam", "Vikkiramangalam Drip Ledge 6", False, False),
        ("SET_B_08", "TBSI_085_original.jpg", "Varichiyur", "Varichiyur Drip Ledge 1", True, False),
        ("SET_B_09", "TBSI_086_original.jpg", "Varichiyur", "Varichiyur Drip Ledge 2", False, False),
        ("SET_B_10", "TBSI_087_original.jpg", "Varichiyur", "Varichiyur Stone Bed 1", False, False),
        ("SET_B_11", "TBSI_029_original.jpg", "Kongarpuliyankulam", "Kongarpuliyankulam Drip Ledge 1", False, False),
        ("SET_B_12", "TBSI_030_original.jpg", "Kongarpuliyankulam", "Kongarpuliyankulam Drip Ledge 2", False, False),
        ("SET_B_13", "TBSI_031_original.jpg", "Kongarpuliyankulam", "Kongarpuliyankulam Drip Ledge 3", False, False),
    ]
    for sid, fname, site, mon, is_fail, is_ctrl in set_b:
        suite.append({
            "specimen_id": sid,
            "group": "tamil_brahmi_set_b",
            "script": "tamil_brahmi",
            "site": site,
            "monument": mon,
            "is_failure_case": is_fail,
            "is_control": is_ctrl,
            "image_source": "zip",
            "zip_inner_path": f"Tamil Brahmi Stone Inscription/Original Images/{fname}",
        })

    return suite


def load_image(item: Dict[str, Any], zip_ref=None) -> Image.Image:
    if item["image_source"] == "file":
        with Image.open(item["file_path"]) as img:
            return img.convert("RGB")
    elif item["image_source"] == "zip":
        with zip_ref.open(item["zip_inner_path"]) as zf:
            with Image.open(zf) as img:
                return img.convert("RGB")
    raise ValueError(f"Unknown image source: {item['image_source']}")


def evaluate_specimen(model: torch.nn.Module, scale: float, img: Image.Image, transform, device: torch.device):
    # Multi-tile aggregation matching production vit_inference.py
    tiles = extract_aspect_preserving_tiles(img, max_tiles=5, overlap=0.25)
    with torch.no_grad():
        if len(tiles) > 1:
            tile_tensors = [transform(t) for t in tiles]
            whole_tensor = transform(img)
            batch = torch.stack([whole_tensor] + tile_tensors).to(device)
            batch_logits = model(batch)
            logits = batch_logits.mean(dim=0, keepdim=True)
        else:
            tensor = transform(img).unsqueeze(0).to(device)
            logits = model(tensor)

        probs = F.softmax(logits, dim=-1)[0]
        cos_sims = (logits[0] / scale).cpu().tolist()

    top_idx = torch.argmax(probs).item()
    top_class = CLASSES[top_idx]
    top_prob = float(probs[top_idx].item())
    top_cos = float(cos_sims[top_idx])

    # Sorted cosine similarities for margin computation
    sorted_cos = sorted(cos_sims, reverse=True)
    margin = float(sorted_cos[0] - sorted_cos[1]) if len(sorted_cos) > 1 else 1.0

    return {
        "pred_class": top_class,
        "confidence": top_prob,
        "cosine_similarity": top_cos,
        "margin": margin,
        "all_cosines": {c: round(cos_sims[i], 4) for i, c in enumerate(CLASSES)},
        "all_probs": {c: round(float(probs[i].item()), 4) for i, c in enumerate(CLASSES)},
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n=======================================================")
    print("DeepScript — Comparative Evaluation: Baseline vs. Finetuned")
    print(f"Device: {device}")
    print("=======================================================\n")

    baseline_model, baseline_scale = load_model(BASELINE_CKPT, device)
    finetuned_model, finetuned_scale = load_model(FINETUNED_CKPT, device)

    eval_tf = get_eval_transforms(image_size=(224, 224))
    suite = build_evaluation_suite()
    print(f"Evaluation Suite: {len(suite)} total held-out specimens.")

    results_rows = []
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        for idx, item in enumerate(suite):
            img = load_image(item, zip_ref=z)
            w, h = img.size

            b_res = evaluate_specimen(baseline_model, baseline_scale, img, eval_tf, device)
            f_res = evaluate_specimen(finetuned_model, finetuned_scale, img, eval_tf, device)

            gt = item["script"]
            b_correct = (b_res["pred_class"] == gt)
            f_correct = (f_res["pred_class"] == gt)

            gt_idx = CLASS_TO_IDX[gt]
            b_gt_cos = b_res["all_cosines"][gt]
            f_gt_cos = f_res["all_cosines"][gt]
            delta_cos = round(f_gt_cos - b_gt_cos, 4)

            # Determine transition outcome
            if not b_correct and f_correct:
                outcome = "IMPROVED (FIXED)"
            elif b_correct and not f_correct:
                outcome = "REGRESSED (BROKEN)"
            elif b_correct and f_correct:
                outcome = "MAINTAINED (PASS)"
            else:
                outcome = "UNCHANGED (FAIL)"

            row = {
                "specimen_id": item["specimen_id"],
                "group": item["group"],
                "ground_truth": gt,
                "site": item["site"],
                "monument": item["monument"],
                "is_failure_case": item["is_failure_case"],
                "is_control": item["is_control"],
                "dimensions": f"{w}x{h}",
                "baseline_pred": b_res["pred_class"],
                "baseline_conf": round(b_res["confidence"], 4),
                "baseline_gt_cos": b_gt_cos,
                "baseline_top_cos": round(b_res["cosine_similarity"], 4),
                "baseline_margin": round(b_res["margin"], 4),
                "finetuned_pred": f_res["pred_class"],
                "finetuned_conf": round(f_res["confidence"], 4),
                "finetuned_gt_cos": f_gt_cos,
                "finetuned_top_cos": round(f_res["cosine_similarity"], 4),
                "finetuned_margin": round(f_res["margin"], 4),
                "delta_gt_cos": delta_cos,
                "outcome": outcome,
                "baseline_correct": b_correct,
                "finetuned_correct": f_correct,
            }
            results_rows.append(row)

            flag = "FIXED" if outcome == "IMPROVED (FIXED)" else ("REGRESSION" if outcome == "REGRESSED (BROKEN)" else ("PASS" if f_correct else "FAIL"))
            print(
                f"[{idx+1:02d}/25] {item['specimen_id']:12s} ({gt:12s}) | "
                f"Base: {b_res['pred_class']:12s} ({b_res['confidence']*100:4.1f}%) -> "
                f"Fine: {f_res['pred_class']:12s} ({f_res['confidence']*100:4.1f}%) | "
                f"d_cos: {delta_cos:+0.4f} | {flag}"
            )

    df_res = pd.DataFrame(results_rows)
    csv_path = RESULTS_DIR / "comparison_predictions.csv"
    df_res.to_csv(csv_path, index=False)
    print(f"\nSaved detailed comparison predictions to: {csv_path}")

    # Summary Metrics Computation
    # 1. Overall Capture Accuracy
    base_acc = df_res["baseline_correct"].mean()
    fine_acc = df_res["finetuned_correct"].mean()

    # 2. Tamil-Brahmi Metrics (N=17 captures)
    tb_df = df_res[df_res["ground_truth"] == "tamil_brahmi"]
    tb_base_acc = tb_df["baseline_correct"].mean()
    tb_fine_acc = tb_df["finetuned_correct"].mean()

    # Monument-grouped Tamil-Brahmi metrics (N=13 monuments)
    tb_base_mon = tb_df.groupby("monument")["baseline_correct"].mean() >= 0.5
    tb_fine_mon = tb_df.groupby("monument")["finetuned_correct"].mean() >= 0.5
    tb_base_mon_acc = tb_base_mon.mean()
    tb_fine_mon_acc = tb_fine_mon.mean()

    # 3. Non-Tamil Script Regressions (N=8 samples across 5 scripts)
    nt_df = df_res[df_res["ground_truth"] != "tamil_brahmi"]
    nt_base_acc = nt_df["baseline_correct"].mean()
    nt_fine_acc = nt_df["finetuned_correct"].mean()
    regressions = df_res[df_res["outcome"] == "REGRESSED (BROKEN)"]

    # 4. Known Failure Cases (N=5: SET_A_01, SET_A_02, SET_A_03, SET_B_01, SET_B_08)
    fc_df = df_res[df_res["is_failure_case"] == True]
    fc_fixed = (fc_df["outcome"] == "IMPROVED (FIXED)").sum()

    # 5. Positive Controls (N=2: SET_A_04, SET_B_02)
    ctrl_df = df_res[df_res["is_control"] == True]
    ctrl_pass = ctrl_df["finetuned_correct"].sum()

    summary = {
        "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_specimens": len(df_res),
        "overall_accuracy": {
            "baseline": round(base_acc * 100, 2),
            "finetuned": round(fine_acc * 100, 2),
            "delta_percent": round((fine_acc - base_acc) * 100, 2),
        },
        "tamil_brahmi_capture_sensitivity": {
            "sample_size": len(tb_df),
            "baseline": round(tb_base_acc * 100, 2),
            "finetuned": round(tb_fine_acc * 100, 2),
            "delta_percent": round((tb_fine_acc - tb_base_acc) * 100, 2),
        },
        "tamil_brahmi_monument_sensitivity": {
            "monument_count": len(tb_base_mon),
            "baseline": round(tb_base_mon_acc * 100, 2),
            "finetuned": round(tb_fine_mon_acc * 100, 2),
            "delta_percent": round((tb_fine_mon_acc - tb_base_mon_acc) * 100, 2),
        },
        "non_tamil_script_metrics": {
            "sample_size": len(nt_df),
            "baseline": round(nt_base_acc * 100, 2),
            "finetuned": round(nt_fine_acc * 100, 2),
            "regressions_count": len(regressions),
            "regression_details": regressions["specimen_id"].tolist(),
        },
        "known_failure_cases_analysis": {
            "total_failure_cases": len(fc_df),
            "fixed_count": int(fc_fixed),
            "details": {
                r["specimen_id"]: {
                    "baseline_pred": r["baseline_pred"],
                    "finetuned_pred": r["finetuned_pred"],
                    "delta_cos": r["delta_gt_cos"],
                    "outcome": r["outcome"],
                }
                for _, r in fc_df.iterrows()
            },
        },
        "positive_controls_analysis": {
            "total_controls": len(ctrl_df),
            "passed_count": int(ctrl_pass),
            "details": {
                r["specimen_id"]: {
                    "baseline_pred": r["baseline_pred"],
                    "finetuned_pred": r["finetuned_pred"],
                    "outcome": r["outcome"],
                }
                for _, r in ctrl_df.iterrows()
            },
        },
    }

    json_path = RESULTS_DIR / "summary_metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary metrics to: {json_path}")

    print("\n=======================================================")
    print("EVALUATION EXECUTIVE SUMMARY")
    print("=======================================================")
    print(f"Overall Accuracy:        {base_acc*100:5.1f}% -> {fine_acc*100:5.1f}% (Delta: {(fine_acc-base_acc)*100:+5.1f}%)")
    print(f"TB Capture Sensitivity:  {tb_base_acc*100:5.1f}% -> {tb_fine_acc*100:5.1f}% (Delta: {(tb_fine_acc-tb_base_acc)*100:+5.1f}%)")
    print(f"TB Monument Sensitivity: {tb_base_mon_acc*100:5.1f}% -> {tb_fine_mon_acc*100:5.1f}% (Delta: {(tb_fine_mon_acc-tb_base_mon_acc)*100:+5.1f}%)")
    print(f"Failure Cases Fixed:     {fc_fixed}/{len(fc_df)}")
    print(f"Positive Controls Kept:  {ctrl_pass}/{len(ctrl_df)}")
    print(f"Non-Tamil Regressions:   {len(regressions)}")
    print("=======================================================\n")


if __name__ == "__main__":
    main()

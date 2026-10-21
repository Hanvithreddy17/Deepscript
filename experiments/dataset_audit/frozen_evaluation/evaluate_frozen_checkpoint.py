#!/usr/bin/env python3
"""
DeepScript — Frozen Checkpoint Site-Disjoint Evaluation Script
==============================================================
Evaluates the production 6-class Vision Transformer (ViT-B/16) checkpoint
without fine-tuning, training, or code modification.

Active Checkpoint: checkpoints/six_class_vit/best_vit_model.pth
Expected SHA-256: 6efa33c8a11389f64b8fa5040a7de6c4cf8042fb5359889ca4a23f5653fdaa5c

Strict Guarantees:
1. Re-verifies checkpoint SHA-256 before and after evaluation.
2. Uses unchanged production preprocessing and inference pipeline (vit_inference_engine).
3. Reads only approved zero-leakage specimens from corrected_evaluation_manifest.json.
4. Groups multiple views of the same physical monument/inscription together.
5. Evaluates as a Tamil-Brahmi-only sensitivity/recall benchmark on unseen stones.
"""

import sys
import os
import json
import csv
import hashlib
import zipfile
import io
from pathlib import Path
from typing import Dict, Any, List
from collections import defaultdict
from PIL import Image

# Reconfigure stdout for UTF-8 compatibility
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.vit_inference import vit_inference_engine, SIX_CLASS_CHECKPOINT_PATH

EXPECTED_SHA256 = "6efa33c8a11389f64b8fa5040a7de6c4cf8042fb5359889ca4a23f5653fdaa5c"
MANIFEST_PATH = PROJECT_ROOT / "experiments" / "dataset_audit" / "frozen_evaluation" / "corrected_evaluation_manifest.json"
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "dataset_audit" / "frozen_evaluation"
ZIP_PATH = Path(r"C:\Users\hanvi\Downloads\Tamil Brahmi Stone Inscription.zip")


def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def run_evaluation(mode: str = "set_a"):
    """
    mode: 'set_a' (in-repo Sittanavasal captures only)
          'set_b' (archive benchmark captures only)
          'combined' (both Set A and Set B, with Sittanavasal overlapping monument flagged)
    """
    print("=" * 90)
    print("DEEPSCRIPT FROZEN CHECKPOINT SITE-DISJOINT EVALUATION")
    print("=" * 90)

    # 1. Pre-evaluation integrity check
    pre_hash = compute_sha256(SIX_CLASS_CHECKPOINT_PATH)
    print(f"Target Checkpoint: {SIX_CLASS_CHECKPOINT_PATH}")
    print(f"Pre-Eval SHA-256:  {pre_hash}")
    if pre_hash != EXPECTED_SHA256:
        raise ValueError(f"CRITICAL: Checkpoint hash mismatch! Expected {EXPECTED_SHA256}, got {pre_hash}")
    print("[SECURITY] Checkpoint verified 100% UNTOUCHED before evaluation.")

    # 2. Load model into production inference engine
    print("\nLoading model into production ViT inference engine...")
    if not vit_inference_engine.load_model():
        raise RuntimeError("Failed to load production ViT inference engine!")
    print(f"Supported Classes in Engine: {vit_inference_engine.classes}")

    # 3. Load manifest
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    specimens = []

    canonical_sittanavasal_monument = "Sittanavasal Ezhadippattam Cavern, Bed #1 (Pudukkottai)"
    canonical_sittanavasal_site = "Sittanavasal"

    if mode in ["set_a", "combined"]:
        set_a = manifest_data["test_subsets"]["set_a_in_repository_stone_captures"]
        for spec in set_a["specimens"]:
            specimens.append({
                "specimen_id": spec["specimen_id"],
                "subset": "Set A (In-Repo)",
                "source_type": "in_repo",
                "image_path": str(PROJECT_ROOT / spec["file_path"]),
                "display_path": spec["file_path"],
                "monument_group": canonical_sittanavasal_monument,
                "site": canonical_sittanavasal_site,
                "district": "Pudukkottai",
                "monument_type": "Cavern Jain stone bed",
                "period": "ca. 1st Century BCE",
                "dimensions": spec["resolution"],
                "ground_truth": spec["ground_truth"]
            })

    if mode in ["set_b", "combined"] and ZIP_PATH.exists():
        set_b = manifest_data["test_subsets"]["set_b_archive_unseen_stone_inscriptions"]
        for group in set_b["inscriptions_by_monument_group"]:
            m_group_name = group["monument_group"]
            site = group["site"]
            dist = group["district"]
            m_type = group["monument_type"]
            period = group["period"]

            # Unify Sittanavasal across sets
            if "Sittan" in site or "Sittan" in m_group_name:
                m_group_name = canonical_sittanavasal_monument
                site = canonical_sittanavasal_site

            for spec in group["specimens"]:
                specimens.append({
                    "specimen_id": spec["specimen_id"],
                    "subset": "Set B (Archive)",
                    "source_type": "zip_archive",
                    "archive_member": spec["archive_member"],
                    "display_path": f"archive://{spec['archive_member']}",
                    "monument_group": m_group_name,
                    "site": site,
                    "district": dist,
                    "monument_type": m_type,
                    "period": period,
                    "dimensions": spec["resolution"],
                    "ground_truth": spec["ground_truth"]
                })

    print(f"\nEvaluating mode: '{mode}' -> {len(specimens)} total captures...")

    records = []
    zip_handle = zipfile.ZipFile(ZIP_PATH, "r") if mode in ["set_b", "combined"] and ZIP_PATH.exists() else None

    for i, spec in enumerate(specimens, 1):
        print(f"\n[{i:2d}/{len(specimens):2d}] Specimen: {spec['specimen_id']} ({spec['display_path']})")
        print(f"       Site: {spec['site']} ({spec['district']}) | Monument: {spec['monument_group']}")

        if spec["source_type"] == "in_repo":
            img = Image.open(spec["image_path"])
        else:
            img_bytes = zip_handle.read(spec["archive_member"])
            img = Image.open(io.BytesIO(img_bytes))

        # Run production inference with top_k=6
        result = vit_inference_engine.predict(img, top_k=6, filename=spec["display_path"])

        pred_class = result.get("raw_class")
        conf = result.get("confidence", 0.0)
        gt = spec["ground_truth"]
        is_correct = (pred_class == gt)

        # Extract cosine similarities and probabilities for all 6 classes
        cos_map = {}
        prob_map = {}
        for c in result.get("candidates", []):
            cos_map[c["class"]] = c.get("cosine_similarity", 0.0)
            prob_map[c["class"]] = c.get("probability", 0.0)

        record = {
            "specimen_id": spec["specimen_id"],
            "subset": spec["subset"],
            "image_path": spec["display_path"],
            "monument_group": spec["monument_group"],
            "site": spec["site"],
            "district": spec["district"],
            "monument_type": spec["monument_type"],
            "period": spec["period"],
            "ground_truth": gt,
            "predicted_class": pred_class,
            "is_correct": is_correct,
            "calibrated_confidence": conf,
            "decision": result.get("decision"),
            "execution_time_ms": result.get("execution_time_ms"),
            "cosine_brahmi": cos_map.get("brahmi", 0.0),
            "cosine_tamil_brahmi": cos_map.get("tamil_brahmi", 0.0),
            "cosine_grantha": cos_map.get("grantha", 0.0),
            "cosine_gupta": cos_map.get("gupta", 0.0),
            "cosine_kadamba": cos_map.get("kadamba", 0.0),
            "cosine_kharosthi": cos_map.get("kharosthi", 0.0),
            "prob_brahmi": prob_map.get("brahmi", 0.0),
            "prob_tamil_brahmi": prob_map.get("tamil_brahmi", 0.0),
            "prob_grantha": prob_map.get("grantha", 0.0),
            "prob_gupta": prob_map.get("gupta", 0.0),
            "prob_kadamba": prob_map.get("kadamba", 0.0),
            "prob_kharosthi": prob_map.get("kharosthi", 0.0),
        }
        records.append(record)

        mark = "✓ PASS (Tamil-Brahmi)" if is_correct else f"✗ MISCLASSIFIED -> {pred_class}"
        print(f"       Outcome: {mark} (Confidence: {conf*100:.2f}%)")
        print(f"       Cosines: TB={cos_map.get('tamil_brahmi'):+.4f} | Brahmi={cos_map.get('brahmi'):+.4f} | Grantha={cos_map.get('grantha'):+.4f}")

    if zip_handle:
        zip_handle.close()

    # 4. Post-evaluation integrity check
    post_hash = compute_sha256(SIX_CLASS_CHECKPOINT_PATH)
    print("\n" + "=" * 90)
    print(f"Post-Eval SHA-256: {post_hash}")
    if post_hash != EXPECTED_SHA256:
        raise ValueError("CRITICAL SECURITY ALERT: Checkpoint hash altered during evaluation!")
    print("[SECURITY] Checkpoint verified 100% UNTOUCHED after evaluation.")

    # 5. Monument-Grouped & Site-Grouped Metrics
    total_captures = len(records)
    correct_captures = sum(1 for r in records if r["is_correct"])
    capture_sensitivity = correct_captures / total_captures if total_captures > 0 else 0.0

    monuments = defaultdict(list)
    site_results = defaultdict(list)
    confusion_counts = defaultdict(int)

    for r in records:
        monuments[r["monument_group"]].append(r)
        site_results[r["site"]].append(r["is_correct"])
        confusion_counts[r["predicted_class"]] += 1

    all_classes = vit_inference_engine.classes

    # Monument-level aggregations
    monument_summaries = []
    monument_correct_vote = 0
    monument_correct_cosine = 0

    for m_name, m_records in monuments.items():
        m_gt = m_records[0]["ground_truth"]
        m_site = m_records[0]["site"]
        m_dist = m_records[0]["district"]
        k = len(m_records)

        # Method 1: Majority Vote across captures
        vote_counts = defaultdict(int)
        for r in m_records:
            vote_counts[r["predicted_class"]] += 1
        majority_class = max(vote_counts.items(), key=lambda x: x[1])[0]
        is_correct_vote = (majority_class == m_gt)
        if is_correct_vote:
            monument_correct_vote += 1

        # Method 2: Mean Cosine Similarity Ensemble across captures
        mean_cosines = {}
        for cls in all_classes:
            mean_cosines[cls] = sum(r[f"cosine_{cls}"] for r in m_records) / k
        mean_cosine_class = max(mean_cosines.items(), key=lambda x: x[1])[0]
        is_correct_cosine = (mean_cosine_class == m_gt)
        if is_correct_cosine:
            monument_correct_cosine += 1

        monument_summaries.append({
            "monument_group": m_name,
            "site": m_site,
            "district": m_dist,
            "captures_count": k,
            "ground_truth": m_gt,
            "capture_predictions": [(r["specimen_id"], r["predicted_class"], r["is_correct"]) for r in m_records],
            "majority_vote_class": majority_class,
            "majority_vote_correct": is_correct_vote,
            "mean_cosine_class": mean_cosine_class,
            "mean_cosine_correct": is_correct_cosine,
            "mean_cosines": mean_cosines
        })

    total_monuments = len(monuments)
    monument_sensitivity_vote = (monument_correct_vote / total_monuments) if total_monuments > 0 else 0.0
    monument_sensitivity_cosine = (monument_correct_cosine / total_monuments) if total_monuments > 0 else 0.0

    print("\n" + "=" * 90)
    print("BENCHMARK METRICS SUMMARY (TAMIL-BRAHMI UNSEEN STONE EVALUATION)")
    print("=" * 90)
    print(f"Evaluation Mode:                      {mode}")
    print(f"Total Photographic Captures:          {total_captures}")
    print(f"Unique Independent Sites:             {len(site_results)} ({', '.join(sorted(site_results.keys()))})")
    print(f"Distinct Physical Monuments:          {total_monuments}")
    print(f"Capture-Level Sensitivity:            {correct_captures} / {total_captures} ({capture_sensitivity*100:.2f}%)")
    print(f"Monument Sensitivity (Majority Vote): {monument_correct_vote} / {total_monuments} ({monument_sensitivity_vote*100:.2f}%)")
    print(f"Monument Sensitivity (Mean Cosine):   {monument_correct_cosine} / {total_monuments} ({monument_sensitivity_cosine*100:.2f}%)")

    print("\n--- Confusion Breakdown (Predicted Class Distribution) ---")
    for cls in all_classes:
        count = confusion_counts.get(cls, 0)
        pct = (count / total_captures * 100) if total_captures > 0 else 0
        tag = "(True Positive)" if cls == "tamil_brahmi" else "(Misclassification)"
        print(f"  {cls:<15}: {count:2d} / {total_captures:2d} ({pct:5.1f}%) {tag}")

    print("\n--- Per-Site Capture Accuracy Breakdown ---")
    for site, bools in sorted(site_results.items()):
        s_corr = sum(bools)
        s_tot = len(bools)
        print(f"  {site:<22}: {s_corr}/{s_tot} ({s_corr/s_tot*100:5.1f}%) across {s_tot} capture(s)")

    print("\n--- Per-Monument Aggregated Breakdown ---")
    for ms in sorted(monument_summaries, key=lambda x: x["monument_group"]):
        vote_tag = "PASS" if ms["majority_vote_correct"] else f"FAIL -> {ms['majority_vote_class']}"
        cos_tag = "PASS" if ms["mean_cosine_correct"] else f"FAIL -> {ms['mean_cosine_class']}"
        print(f"  {ms['monument_group']:<55}: k={ms['captures_count']} | Vote: {vote_tag} | Cosine: {cos_tag}")

    # 6. Save results
    csv_out = OUTPUT_DIR / f"evaluation_predictions_{mode}.csv"
    json_out = OUTPUT_DIR / f"evaluation_results_{mode}.json"

    fieldnames = list(records[0].keys())
    with open(csv_out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(records)

    summary_out = {
        "benchmark_type": "Tamil-Brahmi Unseen Stone Evaluation",
        "checkpoint": str(SIX_CLASS_CHECKPOINT_PATH),
        "checkpoint_sha256_pre": pre_hash,
        "checkpoint_sha256_post": post_hash,
        "hash_verification_passed": (pre_hash == post_hash == EXPECTED_SHA256),
        "evaluation_mode": mode,
        "total_captures": total_captures,
        "correct_captures_count": correct_captures,
        "capture_sensitivity": capture_sensitivity,
        "distinct_monuments_count": total_monuments,
        "monument_sensitivity_majority_vote": monument_sensitivity_vote,
        "monument_sensitivity_mean_cosine": monument_sensitivity_cosine,
        "unique_sites_count": len(site_results),
        "unique_sites": sorted(list(site_results.keys())),
        "confusion_breakdown": dict(confusion_counts),
        "per_site_results": {s: f"{sum(b)}/{len(b)}" for s, b in site_results.items()},
        "monument_summaries": monument_summaries,
        "records": records
    }

    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(summary_out, f, indent=2)

    print(f"\nSaved predictions CSV to: {csv_out}")
    print(f"Saved results JSON to:     {json_out}")
    print("Execution completed.")


if __name__ == "__main__":
    mode_arg = "combined" if "--combined" in sys.argv else ("set_b" if "--archive-only" in sys.argv else "set_a")
    run_evaluation(mode=mode_arg)

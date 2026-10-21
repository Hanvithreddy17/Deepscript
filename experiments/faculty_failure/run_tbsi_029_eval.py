"""
DeepScript — Evaluation of TBSI_029_original.jpg using Production ViT Pipeline
=============================================================================
Runs the exact production ViT-B/16 inference pipeline on the canonical Tamil-Brahmi
test specimen TBSI_029_original.jpg without modifying any code or checkpoints.
"""

import sys
import hashlib
from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.vit_inference import vit_inference_engine, SIX_CLASS_CHECKPOINT_PATH

def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def main():
    print("=" * 80)
    print("DEEPSCRIPT PRODUCTION INFERENCE EVALUATION: TBSI_029_original.jpg")
    print("=" * 80)

    # 1. Checkpoint integrity check
    ckpt_hash = compute_sha256(SIX_CLASS_CHECKPOINT_PATH)
    print(f"Checkpoint: {SIX_CLASS_CHECKPOINT_PATH}")
    print(f"SHA-256:    {ckpt_hash}")
    expected_hash = "6efa33c8a11389f64b8fa5040a7de6c4cf8042fb5359889ca4a23f5653fdaa5c"
    assert ckpt_hash == expected_hash, f"Hash mismatch! Expected {expected_hash}, got {ckpt_hash}"
    print("[SECURITY] Checkpoint verified 100% UNTOUCHED.")

    # 2. Image loading
    img_path = PROJECT_ROOT / "dataset" / "whole_inscriptions" / "tamil_brahmi" / "TBSI_029_original.jpg"
    print(f"\nImage Path: {img_path}")
    if not img_path.exists():
        print(f"ERROR: Image not found at {img_path}")
        return

    img = Image.open(img_path)
    w, h = img.size
    ar = w / h
    print(f"Image Dimensions: {w}x{h} px (Aspect Ratio: {ar:.2f})")

    # 3. Load model into production inference engine
    loaded = vit_inference_engine.load_model()
    if not loaded:
        print("ERROR: Failed to load production ViT model!")
        return

    print(f"Supported Classes in Engine: {vit_inference_engine.classes}")

    # 4. Run predict (top_k=6 so all classes are returned)
    result = vit_inference_engine.predict(img, top_k=6, filename="TBSI_029_original.jpg")

    print("\n" + "=" * 80)
    print("INFERENCE RESULT SUMMARY")
    print("=" * 80)
    print(f"Status:               {result.get('status')}")
    print(f"Decision:             {result.get('decision')}")
    print(f"Predicted Raw Class:  {result.get('raw_class')}")
    print(f"Script Family:        {result.get('script_family')}")
    print(f"Calibrated Confidence:{result.get('confidence'):.4f} ({result.get('confidence')*100:.2f}%)")
    print(f"Execution Time:       {result.get('execution_time_ms')} ms")
    
    morph = result.get("details", {}).get("morphologyMetrics", {})
    if morph:
        print(f"Top-1 Softmax Prob:   {morph.get('top1_probability')}")
        print(f"Max Cosine Similarity:{morph.get('max_cosine_similarity')}")
        print(f"Classification Margin:{morph.get('classification_margin')}")

    print("\n" + "-" * 80)
    print(f"{'Class':<15} | {'Probability':<12} | {'Cosine Similarity':<18} | {'Confidence':<12}")
    print("-" * 80)
    for c in result.get("candidates", []):
        print(f"{c.get('class'):<15} | {c.get('probability')*100:>10.2f}% | {c.get('cosine_similarity'):>18.4f} | {c.get('confidence')*100:>10.2f}%")
    print("-" * 80)

    # Re-verify checkpoint integrity
    post_hash = compute_sha256(SIX_CLASS_CHECKPOINT_PATH)
    assert post_hash == expected_hash, "Post-run hash changed!"
    print(f"\n[SECURITY] Post-run SHA-256 confirmed untouched: {post_hash}")

if __name__ == "__main__":
    main()

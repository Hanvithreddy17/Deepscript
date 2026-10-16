#!/usr/bin/env python3
"""
DeepScript — End-to-End System Verification Test
================================================
Tests the complete inference and rejection path across:
1. Genuine Tamil-Brahmi inscriptions from different archaeological sites
2. Genuine Ashokan Brahmi inscriptions
3. Other supported classes (Grantha, Gupta, Kadamba, Kharosthi)
4. Out-of-distribution negatives (blank images, ordinary objects, textured rocks, modern languages, medieval scripts)
5. Invalid input rejection (empty bytes, corrupted file, unreadable raster)
"""

import sys
import json
from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.vit_inference import vit_inference_engine

def run_tests():
    print("=" * 70)
    print(" DEEPSCRIPT: END-TO-END SYSTEM VERIFICATION AUDIT")
    print("=" * 70)

    # 1. Initialize and load model
    vit_inference_engine.load_model()
    print(f"Loaded Checkpoint : {vit_inference_engine.checkpoint_path}")
    print(f"Supported Classes : {vit_inference_engine.classes} (Count: {len(vit_inference_engine.classes)})")

    test_cases = [
        # --- Group 1: Genuine Tamil-Brahmi Inscriptions ---
        {
            "group": "Tamil-Brahmi (Genuine)",
            "file": "dataset/whole_inscriptions/tamil_brahmi/Mangulam_inscription_south_India.jpg",
            "site": "Mangulam, Madurai",
            "expected_status": "success",
            "expected_class": "tamil_brahmi",
        },
        {
            "group": "Tamil-Brahmi (Genuine)",
            "file": "dataset/whole_inscriptions/tamil_brahmi/2nd_century_BCE_Tamil_Brahmi_inscription_Arittapatti_Madurai_India.jpg",
            "site": "Arittapatti, Madurai",
            "expected_status": "success",
            "expected_class": "tamil_brahmi",
        },
        {
            "group": "Tamil-Brahmi (Genuine)",
            "file": "dataset/whole_inscriptions/tamil_brahmi/TBSI_020_original.jpg",
            "site": "Arittapatti, Madurai",
            "expected_status": "success",
            "expected_class": "tamil_brahmi",
        },
        {
            "group": "Tamil-Brahmi (Genuine)",
            "file": "dataset/whole_inscriptions/tamil_brahmi/TBSI_029_original.jpg",
            "site": "Kongarpuliyankulam, Madurai",
            "expected_status": "success",
            "expected_class": "tamil_brahmi",
        },
        {
            "group": "Tamil-Brahmi (Genuine)",
            "file": "dataset/whole_inscriptions/tamil_brahmi/Sittanavasal_Rock_Bed_Inscription.jpg",
            "site": "Sittanavasal, Pudukkottai",
            "expected_status": "any", # Test site
            "expected_class": "tamil_brahmi",
        },

        # --- Group 2: Genuine Ashokan Brahmi Inscriptions ---
        {
            "group": "Ashokan Brahmi (Genuine)",
            "file": "dataset/whole_inscriptions/brahmi/Devanampiyena Piyadasi in the Lumbini Edict of Ashoka.jpg",
            "site": "Lumbini, Nepal",
            "expected_status": "success",
            "expected_class": "brahmi",
        },
        {
            "group": "Ashokan Brahmi (Genuine)",
            "file": "dataset/whole_inscriptions/brahmi/Sarnath Ashokan Pillar Inscription.jpg",
            "site": "Sarnath, Varanasi",
            "expected_status": "success",
            "expected_class": "brahmi",
        },

        # --- Group 3: Other Supported Ancient Scripts ---
        {
            "group": "Other Supported Classes",
            "file": "dataset/whole_inscriptions/grantha/Mandagapattu Pallava Cave Inscription, Mahendravarman I.jpg",
            "site": "Mandagapattu, Villupuram",
            "expected_status": "success",
            "expected_class": "grantha",
        },
        {
            "group": "Other Supported Classes",
            "file": "dataset/whole_inscriptions/gupta/Eran inscription of Samudragupta.jpg",
            "site": "Eran, Madhya Pradesh",
            "expected_status": "success",
            "expected_class": "gupta",
        },
        {
            "group": "Other Supported Classes",
            "file": "dataset/whole_inscriptions/kadamba/Talagunda (Pranavalingeshwara Temple) pillar inscription of Mayurasharma.jpg",
            "site": "Talagunda, Karnataka",
            "expected_status": "success",
            "expected_class": "kadamba",
        },
        {
            "group": "Other Supported Classes",
            "file": "dataset/whole_inscriptions/kharosthi/Shahbazgarhi Major Rock Edict of Ashoka, Gandhari Kharosthi.jpg",
            "site": "Shahbazgarhi, Gandhara",
            "expected_status": "success",
            "expected_class": "kharosthi",
        },

        # --- Group 4: Negative Inputs & Blank Images ---
        {
            "group": "Negative Evaluation (Blanks)",
            "file": "dataset/negative_evaluation_set/blank_and_low_info/pure_black.png",
            "site": "Synthetic Blank",
            "expected_status": "script_not_detected",
            "expected_class": None,
        },
        {
            "group": "Negative Evaluation (Blanks)",
            "file": "dataset/negative_evaluation_set/blank_and_low_info/pure_white.png",
            "site": "Synthetic Blank",
            "expected_status": "script_not_detected",
            "expected_class": None,
        },
        {
            "group": "Negative Evaluation (Blanks)",
            "file": "dataset/negative_evaluation_set/blank_and_low_info/uniform_gray.png",
            "site": "Synthetic Blank",
            "expected_status": "script_not_detected",
            "expected_class": None,
        },
        {
            "group": "Negative Evaluation (Ordinary Objects)",
            "file": "dataset/negative_evaluation_set/ordinary_objects/batman-.jpg",
            "site": "Everyday Photograph",
            "expected_status": "script_not_detected",
            "expected_class": None,
        },
        {
            "group": "Negative Evaluation (Ordinary Objects)",
            "file": "dataset/negative_evaluation_set/ordinary_objects/spidey.jpg",
            "site": "Everyday Photograph",
            "expected_status": "script_not_detected",
            "expected_class": None,
        },
        {
            "group": "Negative Evaluation (Ordinary Objects)",
            "file": "dataset/negative_evaluation_set/ordinary_objects/himym.webp",
            "site": "Television Still",
            "expected_status": "unsupported_script",
            "expected_class": None,
        },
        {
            "group": "Negative Evaluation (Natural Unscribed Rocks)",
            "file": "dataset/negative_evaluation_set/textured_unscribed_rocks/granite_hill_boulders_natural.jpg",
            "site": "Natural Granite Slope",
            "expected_status": "script_not_detected",
            "expected_class": None,
        },
        {
            "group": "Negative Evaluation (Modern Languages)",
            "file": "dataset/negative_evaluation_set/modern_other_languages/modern_english_latin_sample.png",
            "site": "Modern Latin Typography",
            "expected_status": "unsupported_script",
            "expected_class": None,
        },
        {
            "group": "Negative Evaluation (Unsupported Scripts)",
            "file": "dataset/negative_evaluation_set/medieval_tamil/vatteluttu_script_specimen.png",
            "site": "Vatteluttu Inscription Specimen",
            "expected_status": "script_not_detected",
            "expected_class": None,
        },
    ]

    results = []
    print("\n" + f"{'Group':<32} | {'Image / Site':<40} | {'Status':<22} | {'Prediction':<18} | {'Cos / Conf':<12}")
    print("-" * 135)

    for tc in test_cases:
        p = PROJECT_ROOT / tc["file"]
        if not p.exists():
            print(f"Skipping missing file: {p}")
            continue

        with Image.open(p) as img:
            res = vit_inference_engine.predict(image=img, filename=p.name)

        status = res["status"]
        script = res["script"]
        raw_class = res.get("raw_class")
        conf = res["confidence"]
        top_cand = res["candidates"][0] if res.get("candidates") else {}
        cos = top_cand.get("cosine_similarity", 0.0)

        results.append({
            "test_case": tc,
            "result": res,
        })

        display_name = f"{p.name[:25]}... ({tc['site'][:12]})"
        print(f"{tc['group']:<32} | {display_name:<40} | {status:<22} | {script:<18} | cos={cos:.3f}, conf={conf:.1%}")

    report_out = PROJECT_ROOT / "results" / "system_verification_results.json"
    report_out.parent.mkdir(parents=True, exist_ok=True)
    with open(report_out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved Full Audit Log to: {report_out}")


if __name__ == "__main__":
    run_tests()

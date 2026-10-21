#!/usr/bin/env python3
"""
DeepScript — Isolated Tamil-Brahmi Dataset Preparation
=====================================================
Extracts verified unique Tamil-Brahmi stone inscriptions from the local TBSI archive:
  C:\\Users\\hanvi\\Downloads\\Tamil Brahmi Stone Inscription.zip
into:
  experiments/tamil_brahmi_finetune/extracted_images/

Generates:
  experiments/tamil_brahmi_finetune/experimental_manifest.csv
  experiments/tamil_brahmi_finetune/experimental_manifest.json

Ensures:
- 100% isolation: does NOT touch dataset/ or dataset/metadata.csv.
- Strict site-disjointness: only images from designated TRAIN sites are added to training.
- Zero leakage: held-out test sites (Sittanavasal, Vikkiramangalam, Varichiyur, Kongarpuliyankulam) remain 100% unseen.
- No redundant views: multiple views of the same inscription (e.g. TBSI_005, TBSI_006) are excluded.
"""

import sys
import os
import zipfile
import json
import pandas as pd
from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
EXP_DIR = PROJECT_ROOT / "experiments" / "tamil_brahmi_finetune"
EXTRACTED_DIR = EXP_DIR / "extracted_images"
ZIP_PATH = Path(r"C:\Users\hanvi\Downloads\Tamil Brahmi Stone Inscription.zip")

EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)

# 1. Load metadata
arch_meta = pd.read_csv(PROJECT_ROOT / "dataset" / "tbsi_audit" / "archaeological_metadata.csv")
site_splits = pd.read_csv(PROJECT_ROOT / "experiments" / "dataset_audit" / "leakage_resistant_site_splits.csv")
train_sites = set(site_splits[site_splits["proposed_split"] == "train"]["site_name"])

# 2. Existing TBSI images already in dataset/whole_inscriptions/tamil_brahmi/
ALREADY_IN_DATASET = {
    "TBSI_001", "TBSI_002", "TBSI_020", "TBSI_026",
    "TBSI_035", "TBSI_036", "TBSI_051", "TBSI_068"
}

# Redundant / overlapping segment exclusions
REDUNDANT_EXCLUSIONS = {
    "TBSI_005", "TBSI_006"  # Overlapping segments of Inscription 4 at Alagarmalai (TBSI_004 kept)
}

# 3. Filter candidates
candidates = arch_meta[
    arch_meta["site_name"].isin(train_sites) &
    (~arch_meta["image_id"].isin(ALREADY_IN_DATASET)) &
    (~arch_meta["image_id"].isin(REDUNDANT_EXCLUSIONS))
].copy()

print(f"Total verified candidate TBSI images from training sites: {len(candidates)}")
print(f"Sites represented ({candidates['site_name'].nunique()} sites):")
print(candidates["site_name"].value_counts())

# 4. Extract from ZIP
extracted_records = []
with zipfile.ZipFile(ZIP_PATH, "r") as z:
    for _, row in candidates.iterrows():
        img_id = row["image_id"]
        zip_inner_name = f"Tamil Brahmi Stone Inscription/Original Images/{img_id}_original.jpg"
        out_filename = f"{img_id}_original.jpg"
        out_path = EXTRACTED_DIR / out_filename

        if not out_path.exists():
            with z.open(zip_inner_name) as zf, open(out_path, "wb") as of:
                of.write(zf.read())

        # Verify PIL can open and validate dimensions
        with Image.open(out_path) as img:
            w, h = img.size
            fmt = img.format

        extracted_records.append({
            "image_path": str(out_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "script": "tamil_brahmi",
            "source": "Mendeley Data TBSI v2 / TNDA",
            "source_url": "https://data.mendeley.com/datasets/2spz2js9jp/1",
            "site": row["site_name"],
            "district": row["district"],
            "monument_type": row["monument_type"],
            "inscription_type": "stone_inscription",
            "image_type": "whole_inscription",
            "license": "CC BY 4.0",
            "split": "train",
            "verified": True,
            "notes": f"TBSI {img_id} from {row['site_name']}. Period: {row['Period']}. Ref: {row['archaeological_reference']}",
            "width": w,
            "height": h,
            "is_new_tbsi": True,
        })

print(f"Successfully extracted and validated {len(extracted_records)} unique images.")

# 5. Build full experimental manifest combining original baseline + new TBSI
orig_meta = pd.read_csv(PROJECT_ROOT / "dataset" / "metadata.csv")
manifest_rows = []

# Add original metadata rows
for _, row in orig_meta.iterrows():
    r = dict(row)
    r["is_new_tbsi"] = False
    manifest_rows.append(r)

# Add newly extracted TBSI training images
manifest_rows.extend(extracted_records)

manifest_df = pd.DataFrame(manifest_rows)
out_csv = EXP_DIR / "experimental_manifest.csv"
out_json = EXP_DIR / "experimental_manifest.json"

manifest_df.to_csv(out_csv, index=False)
with open(out_json, "w", encoding="utf-8") as f:
    json.dump(manifest_rows, f, indent=2)

print(f"\nSaved experimental manifest to:")
print(f"  • {out_csv}")
print(f"  • {out_json}")
print(f"\nManifest Summary:")
print(f"  • Total samples: {len(manifest_df)}")
print(f"  • Split breakdown:\n{manifest_df['split'].value_counts()}")
print(f"  • Train samples by class:\n{manifest_df[manifest_df['split'] == 'train']['script'].value_counts()}")

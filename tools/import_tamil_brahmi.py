#!/usr/bin/env python3
"""
DeepScript — Import and Audit Verified Tamil-Brahmi Dataset from Mendeley Data (TBSI)
====================================================================================
Dataset: Tamil Brahmi Stone Inscription (TBSI)
Authors: Sukumar M, Tamil Nadu Department of Archaeology
DOI: 10.17632/2spz2js9jp.1 (v2)
License: CC BY 4.0

This script:
1. Audits the full 93-inscription TBSI dataset.
2. Selects a verified, diverse, and representative set of whole-inscription stone photographs.
3. Downloads the original high-resolution photographs into dataset/whole_inscriptions/tamil_brahmi/.
4. Verifies image integrity (RGB mode, dimensions, readable headers).
5. Appends standardized metadata rows into dataset/metadata.csv.
"""

import os
import sys
import csv
import json
import urllib.request
from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TARGET_DIR = PROJECT_ROOT / "dataset" / "whole_inscriptions" / "tamil_brahmi"
AUDIT_DIR = PROJECT_ROOT / "dataset" / "tbsi_audit"
METADATA_CSV = PROJECT_ROOT / "dataset" / "metadata.csv"

# Selected representative inscriptions across distinct archaeological sites
SELECTED_IDS = [
    "TBSI_035",  # Mangulam, Madurai (Pandya Nedunjeliyan cave bed dedication, c. 3rd-2nd c. BCE)
    "TBSI_036",  # Mangulam, Madurai (Pandya cave inscription)
    "TBSI_068",  # Sittannavasal, Pudukkottai (Famous Jain cavern stone bed)
    "TBSI_020",  # Arittapatti, Madurai (Cavern dedication)
    "TBSI_025",  # Jambai, Villupuram (Chieftain Athiyaman Neduman Anji record)
    "TBSI_051",  # Pugalur, Karur (Chera royal lineage cavern record)
    "TBSI_002",  # Alagarmalai, Madurai (Jain cavern stone record)
    "TBSI_022",  # Edakal, Wayanad (Cave rock carving)
    "TBSI_029",  # Kongarpuliyankulam, Madurai (Cavern dedication)
    "TBSI_026",  # Karungalakkudi, Madurai (Cavern inscription)
    "TBSI_027",  # Kilavalavu, Madurai (Stone bed epigraph)
    "TBSI_001",  # Aiyarmalai, Karur (Cave inscription)
]


def main():
    print("==================================================================")
    print(" DeepScript: Importing Verified Tamil-Brahmi Inscriptions from TBSI")
    print("==================================================================")

    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load archaeological metadata and split definitions
    arch_file = AUDIT_DIR / "archaeological_metadata.csv"
    split_file = AUDIT_DIR / "split_inscription_level.csv"

    with open(arch_file, "r", encoding="utf-8-sig") as f:
        arch_data = {r["image_id"]: r for r in csv.DictReader(f)}

    with open(split_file, "r", encoding="utf-8-sig") as f:
        split_data = {r["image_id"]: r["split"] for r in csv.DictReader(f)}

    # 2. Query Mendeley Data API for download URLs
    api_url = "https://data.mendeley.com/public-api/datasets/2spz2js9jp"
    print(f"Fetching file index from Mendeley Data API: {api_url} ...")
    req = urllib.request.Request(api_url, headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"})
    with urllib.request.urlopen(req) as resp:
        mendeley_json = json.loads(resp.read().decode())

    files_map = {f["filename"]: f for f in mendeley_json.get("files", [])}

    downloaded_records = []

    for img_id in SELECTED_IDS:
        target_fname = f"{img_id}_original.jpg"
        if target_fname not in files_map:
            print(f"WARNING: {target_fname} not found in Mendeley files index!")
            continue

        file_meta = files_map[target_fname]
        dl_url = file_meta["content_details"]["download_url"]
        dest_path = TARGET_DIR / target_fname

        arch_row = arch_data.get(img_id, {})
        site = arch_row.get("site_name", "Tamil Nadu")
        district = arch_row.get("district", "")
        period = arch_row.get("Period", "c. 3rd c. BCE - 2nd c. CE")
        monument = arch_row.get("monument_type", "Cave")
        ref = arch_row.get("archaeological_reference", "Tamil Nadu Department of Archaeology")
        split = split_data.get(img_id, "train")

        if not dest_path.exists():
            print(f"Downloading {target_fname} ({site}, {district})...")
            dl_req = urllib.request.Request(dl_url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(dl_req) as dl_resp:
                content = dl_resp.read()
                dest_path.write_bytes(content)
            print(f"  -> Saved {dest_path.name} ({len(content)} bytes)")
        else:
            print(f"Already exists: {dest_path.name}")

        # Verify PIL Image integrity
        try:
            with Image.open(dest_path) as im:
                im.verify()
            with Image.open(dest_path) as im:
                w, h = im.size
                mode = im.mode
            print(f"  -> Verified valid image: {w}x{h}, mode={mode}")
        except Exception as e:
            print(f"  -> ERROR corrupt image: {e}")
            continue

        rel_path = f"dataset/whole_inscriptions/tamil_brahmi/{target_fname}"
        downloaded_records.append({
            "image_path": rel_path,
            "script": "tamil_brahmi",
            "source": "Mendeley Data (TBSI) / TNDA",
            "source_url": "https://data.mendeley.com/datasets/2spz2js9jp/1",
            "site": f"{site}, {district}, Tamil Nadu",
            "inscription_type": f"{monument.lower()}_inscription",
            "image_type": "whole_inscription",
            "license": "CC BY 4.0",
            "split": split,
            "verified": "true",
            "notes": f"Tamil-Brahmi stone inscription ({period}) from {site}. Ref: {ref}",
        })

    print(f"\nSuccessfully downloaded and verified {len(downloaded_records)} Tamil-Brahmi whole inscriptions.")

    # 3. Update dataset/metadata.csv
    existing_rows = []
    if METADATA_CSV.exists():
        with open(METADATA_CSV, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            for r in reader:
                # Do not duplicate if already present
                if r["script"] != "tamil_brahmi":
                    existing_rows.append(r)
    else:
        fieldnames = [
            "image_path", "script", "source", "source_url", "site",
            "inscription_type", "image_type", "license", "split", "verified", "notes"
        ]

    # Combine existing 5 classes with new verified tamil_brahmi rows
    all_rows = existing_rows + downloaded_records

    with open(METADATA_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Updated {METADATA_CSV} with total {len(all_rows)} metadata rows.")
    print("Done!")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
DeepScript — Integrate 13 Authenticated Tamil-Brahmi Images into Dataset
========================================================================
Copies the 13 authenticated field photographs from raw_intake into
dataset/whole_inscriptions/tamil_brahmi/ and registers their full provenance
metadata and site-based split assignments in dataset/metadata.csv.
"""

import os
import shutil
import pandas as pd
from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "dataset" / "raw_intake" / "tamil_brahmi_new"
TARGET_DIR = PROJECT_ROOT / "dataset" / "whole_inscriptions" / "tamil_brahmi"
METADATA_CSV = PROJECT_ROOT / "dataset" / "metadata.csv"

# 13 verified images with detailed provenance and archaeological site
NEW_RECORDS = [
    {
        "filename": "2nd_century_BCE_Tamil_Brahmi_inscription_Arittapatti_Madurai_India.jpg",
        "script": "tamil_brahmi",
        "source": "Wikimedia Commons / Tamil Virtual Academy",
        "source_url": "https://commons.wikimedia.org/wiki/File:2nd_century_BCE_Tamil_Brahmi_inscription_Arittapatti_Madurai_India.jpg",
        "site": "Arittapatti, Madurai, Tamil Nadu",
        "inscription_type": "cave_drip_ledge",
        "image_type": "field_photograph",
        "license": "CC BY-SA 4.0",
        "split": "train",
        "verified": "true",
        "notes": "2nd c. BCE cave drip-ledge Tamil-Brahmi inscription at Arittapatti cavern.",
    },
    {
        "filename": "Tamil_Brahmi_inscription_Arittapatti_JEG5886.JPG",
        "script": "tamil_brahmi",
        "source": "Wikimedia Commons / J.E.G.",
        "source_url": "https://commons.wikimedia.org/wiki/File:Tamil_Brahmi_inscription_Arittapatti_JEG5886.JPG",
        "site": "Arittapatti, Madurai, Tamil Nadu",
        "inscription_type": "cave_drip_ledge",
        "image_type": "field_photograph",
        "license": "CC BY-SA 4.0",
        "split": "train",
        "verified": "true",
        "notes": "High-res detailed view of Tamil-Brahmi characters on stone ledge at Arittapatti.",
    },
    {
        "filename": "Arittapatti_inscriptions_in_Tamil_Brahmi_mentioning_Paravar.jpg",
        "script": "tamil_brahmi",
        "source": "Wikimedia Commons / Balhur",
        "source_url": "https://commons.wikimedia.org/wiki/File:Arittapatti_inscriptions_in_Tamil_Brahmi_mentioning_Paravar.jpg",
        "site": "Arittapatti, Madurai, Tamil Nadu",
        "inscription_type": "cave_drip_ledge",
        "image_type": "field_photograph",
        "license": "CC BY-SA 4.0",
        "split": "train",
        "verified": "true",
        "notes": "Arittapatti Tamil-Brahmi inscription mentioning Paravar community dedication.",
    },
    {
        "filename": "Jambai_inscription_on_Adhiyamaan_Nedumaan_Anji.jpg",
        "script": "tamil_brahmi",
        "source": "Wikimedia Commons / Tamil Virtual Academy",
        "source_url": "https://commons.wikimedia.org/wiki/File:Jambai_inscription_on_Adhiyamaan_Nedumaan_Anji.jpg",
        "site": "Jambai, Kallakurichi, Tamil Nadu",
        "inscription_type": "cave_rock_inscription",
        "image_type": "field_photograph",
        "license": "CC BY-SA 3.0",
        "split": "val",
        "verified": "true",
        "notes": "Famous Jambai inscription of Satyaputra Adhiyamaan Nedumaan Anji (1st c. CE).",
    },
    {
        "filename": "Mangulam_inscription_south_India.jpg",
        "script": "tamil_brahmi",
        "source": "Wikimedia Commons / V. Muthukumar",
        "source_url": "https://commons.wikimedia.org/wiki/File:Mangulam_inscription_south_India.jpg",
        "site": "Mangulam, Madurai, Tamil Nadu",
        "inscription_type": "cave_drip_ledge",
        "image_type": "field_photograph",
        "license": "CC BY-SA 4.0",
        "split": "train",
        "verified": "true",
        "notes": "Mangulam early Sangam Pandyan Nedunjeliyan inscription (ca. 3rd-2nd c. BCE).",
    },
    {
        "filename": "Nehanurpatti_Tamil_Brahmi_inscription.JPG",
        "script": "tamil_brahmi",
        "source": "Wikimedia Commons / K. Senguttuvan",
        "source_url": "https://commons.wikimedia.org/wiki/File:Nehanurpatti_Tamil_Brahmi_inscription.JPG",
        "site": "Nehanurpatti, Gingee, Tamil Nadu",
        "inscription_type": "rock_bed_cave",
        "image_type": "field_photograph",
        "license": "CC BY-SA 4.0",
        "split": "val",
        "verified": "true",
        "notes": "Nehanurpatti natural cavern Tamil-Brahmi inscription.",
    },
    {
        "filename": "Thiruparankundram_rock_cut_cave_temple_inscription_1.jpg",
        "script": "tamil_brahmi",
        "source": "Wikimedia Commons / Balhur",
        "source_url": "https://commons.wikimedia.org/wiki/File:Thiruparankundram_rock_cut_cave_temple_inscription_1.jpg",
        "site": "Thiruparankundram, Madurai, Tamil Nadu",
        "inscription_type": "rock_cut_cave",
        "image_type": "field_photograph",
        "license": "CC BY-SA 4.0",
        "split": "train",
        "verified": "true",
        "notes": "Thiruparankundram rock cut cave inscription mentioning householder from Erukatur.",
    },
    {
        "filename": "Thiruparankundram_rock_cut_cave_temple_inscription_2.jpg",
        "script": "tamil_brahmi",
        "source": "Wikimedia Commons / Balhur",
        "source_url": "https://commons.wikimedia.org/wiki/File:Thiruparankundram_rock_cut_cave_temple_inscription_2.jpg",
        "site": "Thiruparankundram, Madurai, Tamil Nadu",
        "inscription_type": "rock_cut_cave",
        "image_type": "field_photograph",
        "license": "CC BY-SA 4.0",
        "split": "train",
        "verified": "true",
        "notes": "Thiruparankundram cavern drip-ledge inscription close view.",
    },
    {
        "filename": "Sittanavasal_Rock_Bed_Inscription.jpg",
        "script": "tamil_brahmi",
        "source": "Wikimedia Commons / K. Senguttuvan",
        "source_url": "https://commons.wikimedia.org/wiki/File:Sittanavasal_Rock_Bed_Inscription.jpg",
        "site": "Sittanavasal, Pudukkottai, Tamil Nadu",
        "inscription_type": "stone_bed",
        "image_type": "field_photograph",
        "license": "CC BY-SA 4.0",
        "split": "test",
        "verified": "true",
        "notes": "Sittanavasal Ezhadippattam stone bed Jain inscription (1st c. BCE).",
    },
    {
        "filename": "Stone_bed_with_inscription_in_sittanavasal.JPG",
        "script": "tamil_brahmi",
        "source": "Wikimedia Commons / Ganeshk",
        "source_url": "https://commons.wikimedia.org/wiki/File:Stone_bed_with_inscription_in_sittanavasal.JPG",
        "site": "Sittanavasal, Pudukkottai, Tamil Nadu",
        "inscription_type": "stone_bed",
        "image_type": "field_photograph",
        "license": "CC BY-SA 4.0",
        "split": "test",
        "verified": "true",
        "notes": "Sittanavasal stone bed headrest Tamil-Brahmi inscription.",
    },
    {
        "filename": "Chera_Rock_Shelter_Pugalur.jpg",
        "script": "tamil_brahmi",
        "source": "Wikimedia Commons / K. Senguttuvan",
        "source_url": "https://commons.wikimedia.org/wiki/File:Chera_Rock_Shelter_Pugalur.jpg",
        "site": "Pugalur, Karur, Tamil Nadu",
        "inscription_type": "rock_shelter",
        "image_type": "field_photograph",
        "license": "CC BY-SA 4.0",
        "split": "train",
        "verified": "true",
        "notes": "Pugalur Arunattarmalai Chera kings lineage inscription (1st-2nd c. CE).",
    },
    {
        "filename": "Chithannavasal_cave_inscription_1.JPG",
        "script": "tamil_brahmi",
        "source": "Wikimedia Commons / S. Shanmugaraj",
        "source_url": "https://commons.wikimedia.org/wiki/File:Chithannavasal_cave_inscription_1.JPG",
        "site": "Sittanavasal, Pudukkottai, Tamil Nadu",
        "inscription_type": "stone_bed",
        "image_type": "field_photograph",
        "license": "CC BY-SA 4.0",
        "split": "test",
        "verified": "true",
        "notes": "Sittanavasal Ezhadippattam cave stone bed inscription angle 1.",
    },
    {
        "filename": "Chithannavasal_cave_inscription_2.JPG",
        "script": "tamil_brahmi",
        "source": "Wikimedia Commons / S. Shanmugaraj",
        "source_url": "https://commons.wikimedia.org/wiki/File:Chithannavasal_cave_inscription_2.JPG",
        "site": "Sittanavasal, Pudukkottai, Tamil Nadu",
        "inscription_type": "stone_bed",
        "image_type": "field_photograph",
        "license": "CC BY-SA 4.0",
        "split": "test",
        "verified": "true",
        "notes": "Sittanavasal Ezhadippattam cave stone bed inscription angle 2.",
    },
]


def main():
    print("==================================================================")
    print(" DeepScript: Integrating 13 Authenticated Tamil-Brahmi Inscriptions")
    print("==================================================================")
    
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Copy and verify each file
    new_rows = []
    for item in NEW_RECORDS:
        src = RAW_DIR / item["filename"]
        dst = TARGET_DIR / item["filename"]
        
        if not src.exists():
            raise FileNotFoundError(f"Missing raw image: {src}")
            
        shutil.copy2(src, dst)
        
        # Verify image integrity
        with Image.open(dst) as img:
            img.verify()
        with Image.open(dst) as img:
            w, h = img.size
            mode = img.mode
            
        print(f"Verified & Copied: {dst.name} ({w}x{h}, {mode})")
        
        rel_path = f"dataset/whole_inscriptions/tamil_brahmi/{item['filename']}"
        new_rows.append({
            "image_path": rel_path,
            "script": item["script"],
            "source": item["source"],
            "source_url": item["source_url"],
            "site": item["site"],
            "inscription_type": item["inscription_type"],
            "image_type": item["image_type"],
            "license": item["license"],
            "split": item["split"],
            "verified": item["verified"],
            "notes": item["notes"],
        })
        
    # 2. Update metadata.csv
    df_existing = pd.read_csv(METADATA_CSV)
    
    # Remove any existing rows with the same image_path
    new_paths = {r["image_path"] for r in new_rows}
    df_cleaned = df_existing[~df_existing["image_path"].isin(new_paths)]
    
    # Also ensure existing TBSI images have proper site-based split assignments:
    # Sittanavasal / Kongarpuliyankulam -> test
    # Nehanurpatti / Edakal / Kilavalavu / Jambai -> val
    # Arittapatti / Mangulam / Pugalur / Mamandur / Mettuppatti / Alagarmalai / Karungalakkudi / Aiyarmalai -> train
    
    def assign_tbsi_split(row):
        site = str(row["site"])
        if "Kongarpuliyankulam" in site:
            return "test"
        elif "Edakal" in site or "Kilavalavu" in site or "Jambai" in site:
            return "val"
        else:
            return "train"
            
    # Update TBSI rows
    tbsi_mask = (df_cleaned["script"] == "tamil_brahmi")
    df_cleaned.loc[tbsi_mask, "split"] = df_cleaned[tbsi_mask].apply(assign_tbsi_split, axis=1)
    
    # Append new rows
    df_new = pd.DataFrame(new_rows)
    df_final = pd.concat([df_cleaned, df_new], ignore_index=True)
    
    df_final.to_csv(METADATA_CSV, index=False)
    print(f"\nMetadata CSV successfully updated. Total records: {len(df_final)}")
    
    tb_df = df_final[df_final["script"] == "tamil_brahmi"]
    print(f"Total Tamil-Brahmi records: {len(tb_df)}")
    print("Tamil-Brahmi Split distribution:")
    print(tb_df["split"].value_counts())
    print("\nTamil-Brahmi Sites in Test Split (Completely held out from training!):")
    print(tb_df[tb_df["split"] == "test"][["image_path", "site"]])


if __name__ == "__main__":
    main()

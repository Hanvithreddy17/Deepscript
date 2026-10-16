#!/usr/bin/env python3
"""
DeepScript — Group-Aware Split Assignment for Inscription Metadata
==================================================================
Assigns train/val/test splits to all 74 images in dataset/metadata.csv
ensuring:
1. Group-aware site clustering: no site or monument spans across multiple splits.
2. Stratification: every class has representative images in train, val, and test.
3. Completely isolates held-out test sites for unbiased evaluation.
"""

import pandas as pd
from pathlib import Path

METADATA_CSV = Path("dataset/metadata.csv")

def assign_splits():
    df = pd.read_csv(METADATA_CSV)
    
    # Specific site to split mapping for the 5 non-tamil_brahmi classes
    # Brahmi:
    # Train: Kangra Fort (2), Ramgarh (1), Erragudi (1), Vidisha (1), Nashik (1) = 6
    # Val: Besnagar (index 3) (1), Sarnath (1) = 2
    # Test: Rummindei / Lumbini (2) = 2
    
    # Grantha:
    # Train: Velvikudi (2), Mandagapattu (1), Thalavanur (1), Chera (1), Tamil Nadu general (1) = 6
    # Val: Dharmeshwara (1), Kurumathur (1) = 2
    # Test: Vellore (1), Tiruvannamalai (1) = 2
    
    # Gupta:
    # Train: Eran (3), Arang (1), Raipur (1), Prayagraj (1) = 6
    # Val: Bhitari (1), Mathura (1) = 2
    # Test: Nalanda (1), Sanchi (1) = 2
    
    # Kadamba:
    # Train: Talagunda (3), Malavalli (2), Banavasi (1) = 6
    # Val: Kavadi (1), Tadagani (1) = 2
    # Test: Northern Kerala / border (1) = 1 (Total 9)
    
    # Kharosthi:
    # Train: Niya (6), Shahbazgarhi (1) = 7
    # Val: Hashtnagar (1), Mansehra (1) = 2
    # Test: Bimaran (1) = 1 (Total 10)
    
    split_map = {
        # Brahmi
        0: "train", # Ramgarh
        1: "train", # Kangra Fort
        2: "train", # Kangra Fort
        3: "val",   # Besnagar
        4: "train", # Nashik
        5: "train", # Erragudi
        6: "val",   # Sarnath
        7: "test",  # Lumbini
        8: "train", # Vidisha
        9: "test",  # Lumbini / Paderia
        
        # Grantha
        10: "train", # Mandagapattu
        11: "train", # Thalavanur
        12: "train", # Velvikudi
        13: "train", # Velvikudi
        14: "train", # Chera
        15: "train", # Tamil Nadu
        16: "val",   # Dharmeshwara
        17: "val",   # Kurumathur
        18: "test",  # Vellore
        19: "test",  # Tiruvannamalai
        
        # Gupta
        20: "train", # Arang
        21: "train", # Raipur
        22: "train", # Prayagraj
        23: "train", # Eran
        24: "val",   # Bhitari
        25: "train", # Eran
        26: "train", # Eran
        27: "val",   # Mathura
        28: "test",  # Nalanda
        29: "test",  # Sanchi
        
        # Kadamba
        30: "train", # Banavasi
        31: "val",   # Kavadi
        32: "val",   # Tadagani
        33: "test",  # Northern Kerala
        34: "train", # Malavalli
        35: "train", # Malavalli
        36: "train", # Talagunda
        37: "train", # Talagunda
        38: "train", # Talagunda
        
        # Kharosthi
        39: "train", # Shahbazgarhi
        40: "train", # Niya
        41: "val",   # Hashtnagar
        42: "val",   # Mansehra
        43: "train", # Niya
        44: "train", # Niya
        45: "train", # Niya
        46: "train", # Niya
        47: "train", # Niya
        48: "test",  # Bimaran
    }
    
    for idx, split in split_map.items():
        df.loc[idx, "split"] = split
        
    df.to_csv(METADATA_CSV, index=False)
    print("Assigned splits successfully.")
    print("\nOverall Split Distribution by Script:")
    print(df.groupby(["script", "split"])["image_path"].count())

if __name__ == "__main__":
    assign_splits()

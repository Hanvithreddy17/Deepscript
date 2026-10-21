import csv

src_file = "dataset/metadata.csv"
dst_file = "dataset/metadata_stone_only.csv"

# Read original
with open(src_file, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    original_fields = reader.fieldnames
    rows = list(reader)

audit_fields = [
    "physical_medium",
    "image_modality",
    "is_stone",
    "non_stone_flag",
    "authenticity_status",
    "archaeological_site_normalized",
    "physical_monument",
    "label_confidence",
    "audit_status",
    "audit_notes",
]

all_fields = original_fields + audit_fields

# Define monument and normalization mappings
def audit_row(i, r):
    script = r["script"]
    itype = r["inscription_type"]
    site = r["site"].strip()
    path = r["image_path"]
    notes = r["notes"]
    
    # Defaults
    phys_medium = "stone"
    img_modality = "field_photograph"
    is_stone = True
    non_stone_flag = "none"
    auth_status = "authentic_ancient_epigraph"
    site_norm = site if site else "unknown"
    monument = "unknown"
    label_conf = "high"
    audit_stat = "verified_stone"
    audit_notes = "Authentic monumental stone inscription."

    # 1. Brahmi
    if script == "brahmi":
        if i == 0:
            monument = "Sitabhengra / Jogimara Cave Inscription Panel"
            site_norm = "Ramgarh Hills (Sitabhengra and Jogimara Caves), Chhattisgarh"
        elif i == 1:
            monument = "Kangra Fort Inscribed Stone Slab (Angle 1)"
        elif i == 2:
            monument = "Kangra Fort Inscribed Stone Slab (Angle 2)"
            audit_notes = "Alternate view of Kangra Fort stone slab (same physical monument as row 1)."
        elif i == 3:
            site_norm = "unknown"
            monument = "unknown"
            label_conf = "needs_review"
            audit_stat = "needs_review"
            audit_notes = "Field photograph of rock surface, but archaeological site and physical monument are unstated; requires manual review."
        elif i == 4:
            monument = "Cave 17 Donative Inscription of Indragnidatta"
            site_norm = "Nashik Caves (Pandavleni Cave 17), Maharashtra"
        elif i == 5:
            monument = "Ashoka Major Rock Edict, Erragudi Boulder"
            site_norm = "Erragudi (Yerragudi), Kurnool, Andhra Pradesh"
        elif i == 6:
            monument = "Ashoka Pillar Schism Edict Shaft, Sarnath"
            site_norm = "Sarnath, Varanasi, Uttar Pradesh"
        elif i == 7:
            monument = "Lumbini Rummindei Pillar of Ashoka"
            site_norm = "Lumbini (Rummindei), Rupandehi, Nepal"
        elif i == 8:
            monument = "Heliodorus Monolithic Pillar (Garuda-dhvaja)"
            site_norm = "Vidisha (Besnagar), Madhya Pradesh"
        elif i == 9: # Paderia Pillar.png
            phys_medium = "paper"
            img_modality = "rubbing_facsimile"
            is_stone = False
            non_stone_flag = "paper_rubbing"
            auth_status = "reproduction_rubbing"
            monument = "Lumbini Rummindei Pillar of Ashoka"
            site_norm = "Lumbini (Rummindei / Paderia), Rupandehi, Nepal"
            audit_stat = "flagged_non_stone"
            audit_notes = "2D ink rubbing / paper facsimile reproduction of Lumbini pillar; flagged non-stone reproduction."

    # 2. Grantha
    elif script == "grantha":
        if i == 10:
            monument = "Mandagapattu Trimurti Temple Facade Inscription of Mahendravarman I"
            site_norm = "Mandagapattu, Villupuram, Tamil Nadu"
        elif i == 11:
            monument = "Shatrumalleswaram Cave Temple Pillar Inscription"
            site_norm = "Thalavanur, Villupuram, Tamil Nadu"
        elif i == 12: # Velvikudi Grant
            phys_medium = "copper"
            img_modality = "museum_photograph"
            is_stone = False
            non_stone_flag = "copper_plate"
            monument = "Velvikudi Copper Plate Grant of Nedunjadaiyan (Full Set)"
            site_norm = "Velvikudi, Madurai, Tamil Nadu"
            audit_stat = "flagged_non_stone"
            audit_notes = "Engraved copper-plate charter (metallic substrate); flagged non-stone."
        elif i == 13: # Velvikudi detail
            phys_medium = "copper"
            img_modality = "museum_photograph"
            is_stone = False
            non_stone_flag = "copper_plate"
            monument = "Velvikudi Copper Plate Grant (Detail of Lines 8-14)"
            site_norm = "Velvikudi, Madurai, Tamil Nadu"
            audit_stat = "flagged_non_stone"
            audit_notes = "Detail photograph of engraved copper plate; flagged non-stone."
        elif i == 14:
            site_norm = "unknown (regional: Chera Kingdom)"
            monument = "unknown"
            label_conf = "needs_review"
            audit_stat = "needs_review"
            audit_notes = "Stone epigraph, but specific site and monument are unstated in metadata (broad regional label 'Chera Kingdom'); needs review."
        elif i == 15:
            site_norm = "unknown (regional: Tamil Nadu)"
            monument = "unknown"
            label_conf = "needs_review"
            audit_stat = "needs_review"
            audit_notes = "Temple stone wall inscription, but specific site and temple monument are unstated ('Tamil Nadu'); needs review."
        elif i == 16: # Dharmeshwara
            phys_medium = "copper"
            img_modality = "museum_photograph"
            is_stone = False
            non_stone_flag = "copper_plate"
            monument = "Dharmeshwara Temple Copper Plate Charter"
            site_norm = "Kondarahalli (Dharmeshwara Temple), Karnataka / Tamil Nadu border"
            audit_stat = "flagged_non_stone"
            audit_notes = "Copper plate charter with Grantha script; flagged non-stone."
        elif i == 17:
            monument = "Kurumathur Stone Inscription of Rajendra Chola I"
            site_norm = "Kurumathur, Kannur, Kerala"
        elif i == 18:
            monument = "Vijayanagara Stone Record of Venkata II (1605 CE)"
            site_norm = "Vellore district, Tamil Nadu"
        elif i == 19:
            monument = "Virupaksha Temple Inscription of 1481 CE"
            site_norm = "Tiruvannamalai district, Tamil Nadu"

    # 3. Gupta
    elif script == "gupta":
        if i == 20: # Arang
            phys_medium = "copper"
            img_modality = "museum_photograph"
            is_stone = False
            non_stone_flag = "copper_plate"
            monument = "Arang Copper Plate Grant of King Jayaraja"
            site_norm = "Arang, Raipur, Chhattisgarh"
            audit_stat = "flagged_non_stone"
            audit_notes = "6th c. Sharabhapuriya copper plate grant; flagged non-stone."
        elif i == 21: # Raipur
            phys_medium = "copper"
            img_modality = "museum_photograph"
            is_stone = False
            non_stone_flag = "copper_plate"
            monument = "Raipur Copper Plate Grant of King Sudevaraja"
            site_norm = "Raipur district, Chhattisgarh"
            audit_stat = "flagged_non_stone"
            audit_notes = "6th c. royal copper plate grant; flagged non-stone."
        elif i == 22:
            monument = "Allahabad Pillar Inscription of Samudragupta (Prayag Prashasti)"
            site_norm = "Prayagraj (Allahabad Fort), Uttar Pradesh"
        elif i == 23:
            monument = "Eran Stone Pillar Inscription of Bhanugupta (510 CE)"
            site_norm = "Eran, Sagar, Madhya Pradesh"
        elif i == 24:
            monument = "Bhitari Monolithic Pillar Inscription of Skandagupta"
            site_norm = "Bhitari, Ghazipur, Uttar Pradesh"
        elif i == 25:
            monument = "Eran Budhagupta Pillar Inscription (484 CE)"
            site_norm = "Eran, Sagar, Madhya Pradesh"
        elif i == 26:
            monument = "Eran Stone Slab Inscription of Samudragupta"
            site_norm = "Eran, Sagar, Madhya Pradesh"
        elif i == 27:
            monument = "Mathura Pillar Inscription of Chandragupta II (380 CE)"
            site_norm = "Mathura, Uttar Pradesh"
        elif i == 28: # Nalanda clay seal
            phys_medium = "terracotta"
            img_modality = "museum_photograph"
            is_stone = False
            non_stone_flag = "clay_seal"
            monument = "Nalanda Clay Sealing of Vishnugupta"
            site_norm = "Nalanda Monastic Complex, Bihar"
            audit_stat = "flagged_non_stone"
            audit_notes = "Terracotta monastic sealing (baked clay); flagged non-stone."
        elif i == 29:
            monument = "Sanchi Stupa 1 Railing Inscription of Chandragupta II (412 CE)"
            site_norm = "Sanchi Stupa Complex, Raisen, Madhya Pradesh"

    # 4. Kadamba
    elif script == "kadamba":
        if i == 30:
            monument = "Banavasi Madhukeshwara Temple Stone Fragment 2"
            site_norm = "Banavasi (Madhukeshwara Temple), Uttara Kannada, Karnataka"
        elif i == 31:
            monument = "Kavadi Kadamba Hero Stone (View 1)"
            site_norm = "Kavadi, Soraba, Shivamogga, Karnataka"
        elif i == 32:
            monument = "Tadagani Inscribed Stone Slab"
            site_norm = "Tadagani, Shikaripura, Shivamogga, Karnataka"
        elif i == 33:
            site_norm = "unknown (regional: Northern Kerala / Karnataka border)"
            monument = "unknown"
            label_conf = "needs_review"
            audit_stat = "needs_review"
            audit_notes = "Stone slab with Kadamba script, but specific findspot and monument are unstated; needs review."
        elif i == 34:
            monument = "Malavalli Octagonal Pillar of Shivaskandavarman / Mayurasharma (View 1)"
            site_norm = "Malavalli, Shikaripura, Shivamogga, Karnataka"
        elif i == 35:
            monument = "Malavalli Octagonal Pillar of Shivaskandavarman / Mayurasharma (View 2)"
            site_norm = "Malavalli, Shikaripura, Shivamogga, Karnataka"
            audit_notes = "Alternate full view of Malavalli pillar (same physical monument as row 34)."
        elif i == 36:
            monument = "Pranavalingeshwara Temple Left Door Jamb"
            site_norm = "Talagunda, Shikaripura, Shivamogga, Karnataka"
        elif i == 37:
            monument = "Talagunda Monolithic Pillar Inscription of Shantivarman"
            site_norm = "Talagunda, Shikaripura, Shivamogga, Karnataka"
        elif i == 38:
            monument = "Pranavalingeshwara Temple Right Door Jamb"
            site_norm = "Talagunda, Shikaripura, Shivamogga, Karnataka"

    # 5. Kharosthi
    elif script == "kharosthi":
        if i == 39:
            monument = "Shahbazgarhi Major Rock Edict I (Ashoka)"
            site_norm = "Shahbazgarhi, Mardan, Khyber Pakhtunkhwa"
        elif i == 40: # Wooden tablet
            phys_medium = "wood"
            img_modality = "museum_photograph"
            is_stone = False
            non_stone_flag = "wooden_tablet"
            monument = "Niya Inscribed Wooden Administrative Tablet (Cropped)"
            site_norm = "Niya, Minfeng, Xinjiang (Silk Road)"
            audit_stat = "flagged_non_stone"
            audit_notes = "Timber tablet with inked Gandhari Kharosthi text; flagged non-stone."
        elif i == 41:
            monument = "Hashtnagar Rajar Bodhisattva Inscribed Pedestal (Year 384)"
            site_norm = "Hashtnagar (Charsadda), Khyber Pakhtunkhwa"
            img_modality = "museum_photograph"
            audit_notes = "Authentic carved schist statue pedestal with Kharosthi donative epigraph."
        elif i == 42:
            monument = "Mansehra Major Rock Edicts (Upper Boulder)"
            site_norm = "Mansehra, Khyber Pakhtunkhwa"
        elif i == 43: # Wooden plate 01
            phys_medium = "wood"
            img_modality = "museum_photograph"
            is_stone = False
            non_stone_flag = "wooden_plate"
            monument = "Niya Inscribed Wooden Wedge Plate 01 (Stein Collection)"
            site_norm = "Niya, Minfeng, Xinjiang (Silk Road)"
            audit_stat = "flagged_non_stone"
            audit_notes = "Wooden wedge document excavated by Aurel Stein; flagged non-stone."
        elif i == 44: # Wooden plate 02
            phys_medium = "wood"
            img_modality = "museum_photograph"
            is_stone = False
            non_stone_flag = "wooden_plate"
            monument = "Niya Inscribed Wooden Wedge Plate 02 (Stein Collection)"
            site_norm = "Niya, Minfeng, Xinjiang (Silk Road)"
            audit_stat = "flagged_non_stone"
            audit_notes = "Wooden document obverse face; flagged non-stone."
        elif i == 45: # Wooden plate
            phys_medium = "wood"
            img_modality = "museum_photograph"
            is_stone = False
            non_stone_flag = "wooden_plate"
            monument = "Niya Rectangular Inscribed Wooden Plate"
            site_norm = "Niya, Minfeng, Xinjiang (Silk Road)"
            audit_stat = "flagged_non_stone"
            audit_notes = "Rectangular wooden charter; flagged non-stone."
        elif i == 46: # Niya BLH170
            phys_medium = "wood"
            img_modality = "museum_photograph"
            is_stone = False
            non_stone_flag = "wooden_tablet"
            monument = "Niya Wooden Timber Record BLH170"
            site_norm = "Niya, Minfeng, Xinjiang (Silk Road)"
            audit_stat = "flagged_non_stone"
            audit_notes = "Timber communication record from British Library Stein collection; flagged non-stone."
        elif i == 47: # Niya wooden Karosthi tablets
            phys_medium = "wood"
            img_modality = "museum_photograph"
            is_stone = False
            non_stone_flag = "wooden_tablet"
            monument = "Niya Paired Double-Wooden Tablets with Seal Socket"
            site_norm = "Niya, Minfeng, Xinjiang (Silk Road)"
            audit_stat = "flagged_non_stone"
            audit_notes = "Paired wooden tablets with clay seal groove; flagged non-stone."
        elif i == 48: # Bimaran reliquary casket
            phys_medium = "steatite"
            img_modality = "museum_photograph"
            is_stone = False
            non_stone_flag = "reliquary_casket"
            monument = "Bimaran Stupa Steatite Reliquary Casket"
            site_norm = "Bimaran Stupa, Jalalabad, Nangarhar, Afghanistan"
            audit_stat = "flagged_non_stone"
            audit_notes = "Portable carved steatite reliquary bowl/casket (miniature portable vessel); flagged non-stone."

    # 6. Tamil-Brahmi
    elif script == "tamil_brahmi":
        if i == 49: # TBSI_035
            monument = "Mamandur Cavern Drip Ledge Inscription"
            site_norm = "Mamandur, Tiruvannamalai, Tamil Nadu"
        elif i == 50: # TBSI_036
            monument = "Mangulam Meenakshipuram Cavern Drip Ledge (Nedunjeliyan Record)"
            site_norm = "Mangulam, Madurai, Tamil Nadu"
        elif i == 51: # TBSI_068
            monument = "Pugalur Arunattarmalai Stone Bed Inscription (Chera Lineage)"
            site_norm = "Pugalur, Karur, Tamil Nadu"
        elif i == 52: # TBSI_020
            monument = "Arittapatti Kalinjadimalai Cavern Drip Ledge"
            site_norm = "Arittapatti, Madurai, Tamil Nadu"
        elif i == 53: # TBSI_025
            monument = "Jambai Satyaputra Rock Inscription (Adhiyamaan Nedumaan Anji)"
            site_norm = "Jambai, Kallakurichi, Tamil Nadu"
        elif i == 54: # TBSI_051
            monument = "Mettuppatti Siddharmalai Cavern Stone Bed"
            site_norm = "Mettuppatti, Madurai, Tamil Nadu"
        elif i == 55: # TBSI_002
            monument = "Alagarmalai Natural Cavern Stone Inscription"
            site_norm = "Alagarmalai, Madurai, Tamil Nadu"
        elif i == 56: # TBSI_022
            monument = "Edakal Ambukuthi Hills Cavern Wall Inscription"
            site_norm = "Edakal, Wayanad, Kerala"
        elif i == 57: # TBSI_029
            monument = "Kongarpuliyankulam Natural Cavern Drip Ledge"
            site_norm = "Kongarpuliyankulam, Madurai, Tamil Nadu"
        elif i == 58: # TBSI_026
            monument = "Karungalakkudi Cavern Drip Ledge Inscription"
            site_norm = "Karungalakkudi, Madurai, Tamil Nadu"
        elif i == 59: # TBSI_027
            monument = "Kilavalavu Panchapandavar Malai Cavern Inscription"
            site_norm = "Kilavalavu, Madurai, Tamil Nadu"
        elif i == 60: # TBSI_001
            monument = "Aiyarmalai Natural Cavern Stone Inscription"
            site_norm = "Aiyarmalai, Karur, Tamil Nadu"
        elif i == 61: # 2nd_century_BCE...Arittapatti
            monument = "Arittapatti Cavern Drip Ledge (Capture 1)"
            site_norm = "Arittapatti, Madurai, Tamil Nadu"
        elif i == 62: # Tamil_Brahmi...JEG5886
            monument = "Arittapatti Cavern Drip Ledge (Capture 2)"
            site_norm = "Arittapatti, Madurai, Tamil Nadu"
        elif i == 63: # Arittapatti...Paravar
            monument = "Arittapatti Cavern Drip Ledge (Paravar Dedication, Capture 3)"
            site_norm = "Arittapatti, Madurai, Tamil Nadu"
        elif i == 64: # Jambai
            monument = "Jambai Satyaputra Rock Inscription (Adhiyamaan Nedumaan Anji)"
            site_norm = "Jambai, Kallakurichi, Tamil Nadu"
        elif i == 65: # Mangulam
            monument = "Mangulam Meenakshipuram Cavern Drip Ledge (Nedunjeliyan Record)"
            site_norm = "Mangulam, Madurai, Tamil Nadu"
        elif i == 66: # Nehanurpatti
            monument = "Nehanurpatti Natural Cavern Bed Inscription"
            site_norm = "Nehanurpatti (Nekanurpatti), Gingee, Villupuram, Tamil Nadu"
        elif i == 67: # Thiruparankundram_1
            monument = "Thiruparankundram Rock-Cut Cave Temple Ledge (Capture 1)"
            site_norm = "Thiruparankundram, Madurai, Tamil Nadu"
        elif i == 68: # Thiruparankundram_2
            monument = "Thiruparankundram Rock-Cut Cave Temple Ledge (Capture 2)"
            site_norm = "Thiruparankundram, Madurai, Tamil Nadu"
        elif i == 69: # Sittanavasal SET_A_01
            monument = "Sittanavasal Ezhadippattam Stone Bed #1 (Frontal Angle)"
            site_norm = "Sittanavasal (Sittannavasal), Pudukkottai, Tamil Nadu"
        elif i == 70: # Sittanavasal SET_A_02
            monument = "Sittanavasal Ezhadippattam Stone Bed #1 (High-Angle Overview)"
            site_norm = "Sittanavasal (Sittannavasal), Pudukkottai, Tamil Nadu"
        elif i == 71: # Pugalur Chera Rock Shelter
            monument = "Pugalur Arunattarmalai Rock Shelter (Chera Royal Record)"
            site_norm = "Pugalur, Karur, Tamil Nadu"
        elif i == 72: # Sittanavasal SET_A_03
            monument = "Sittanavasal Ezhadippattam Stone Bed #1 (Cavern Wide View)"
            site_norm = "Sittanavasal (Sittannavasal), Pudukkottai, Tamil Nadu"
        elif i == 73: # Sittanavasal SET_A_04
            monument = "Sittanavasal Ezhadippattam Stone Bed #1 (Close-Up Relief)"
            site_norm = "Sittanavasal (Sittannavasal), Pudukkottai, Tamil Nadu"

    # Assemble complete dictionary preserving all original fields exactly
    out_row = dict(r)
    out_row["physical_medium"] = phys_medium
    out_row["image_modality"] = img_modality
    out_row["is_stone"] = is_stone
    out_row["non_stone_flag"] = non_stone_flag
    out_row["authenticity_status"] = auth_status
    out_row["archaeological_site_normalized"] = site_norm
    out_row["physical_monument"] = monument
    out_row["label_confidence"] = label_conf
    out_row["audit_status"] = audit_stat
    out_row["audit_notes"] = audit_notes
    return out_row

audited_rows = [audit_row(i, r) for i, r in enumerate(rows)]

# Write to dataset/metadata_stone_only.csv
with open(dst_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=all_fields)
    writer.writeheader()
    writer.writerows(audited_rows)

print(f"Successfully generated {dst_file} with {len(audited_rows)} rows and {len(all_fields)} fields.")

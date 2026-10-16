| File | Content |
|---|---|
| `split_inscription_level.csv` | train/val/test (66/12/15), inscription level; columns image_id, site_name, split |
| `split_unseen_site.csv` | train/val/test (65/14/14); whole sites in one partition (test sites unseen in training) |
| `split_sitefold_1.csv` ... `_3.csv` | site-wise 3-fold cross-validation: 54/8/31 per fold, test sites never in train/val, every image tested once |
| `GT Masks v2/<id>_mask.png` | corrected binary ground truth (255 = stroke) for all 93 images |
| `degradation_labels.csv` | main visible degradation type per image (single annotator) |

## How the corrected masks were made (GT Masks v2)
The masks are extracted automatically from the pair (Original Images/<id>_original.jpg,
Expert Annotation/<id>_annotated.jpg). A pixel is a stroke only if the annotated image is BRIGHTER than the
original and white: strong = (annotated - original > 50) and V > 160 and S < 80 (3x3 opening); weak =
(difference > 20) and S < 110 (3x3 closing), kept only within 5 px of a strong pixel; connected regions with
fewer than 40 strong pixels and drawn frames are removed. The first version used the absolute colour
difference, which also marked stone texture as foreground (15 of 93 masks had more than 10 % foreground; now
3). Known issues: TBSI_034 has an empty mask (its annotated file is identical
to the original); a few tracings (e.g. TBSI_029, TBSI_038) are stylised transcriptions rather than exact stroke
outlines. Code: gt_mask.py (extract_gt_mask_v2) in the code release.

## Degradation labels
Counts over the 93 images: Erosion / weathering 26, Surface cracks / fracture 18, Biological growth / mineral staining 16, Uneven illumination / shadow 13, Minimal degradation (clean) 13, Low-contrast / coarse texture 7.

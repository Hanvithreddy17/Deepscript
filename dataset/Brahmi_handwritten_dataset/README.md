This repository contains a handwritten dataset of Brahmi script numerals (0–9) stored as vector stroke data (coordinate sequences), created as part of a structured pipeline designed for low-resource scripts. The dataset includes 200 vector samples per numeral, collected, processed, and organized into class-specific structures.

# Dataset Summary
* Script: Brahmi (Ancient Indian numeral system)
* Classes: 10 (Digits 0 to 9)
* Samples per Class: 200
* Format: Structured Vector Data (.csv / .json coordinate sequences)
* Data Type: X, Y coordinates (with stroke sequence indicators)
* Use Case: Machine Learning / Deep Learning for online handwriting recognition, recurrent neural networks (RNNs), and trajectory analysis.
# Methodology Overview
This dataset was created using a structured, step-by-step pipeline optimized for extracting vector paths from handwriting:
1. Data Collection: Handwritten Brahmi numerals were collected from participants aged 15–60 using a printed template or digital input.
2. Digitization & Segmentation: Sheets were scanned using a high-resolution smartphone camera, and individual characters were isolated using Python and OpenCV contour detection.
3. Skeletonization & Path Extraction: To convert pixels into vectors, the segmented images underwent skeletonization (reducing strokes to 1-pixel thickness). A path-finding algorithm then traced the central pixel lines to determine the sequence of coordinates representing the pen's trajectory.
4. Normalization & Preprocessing: * Resampling: Trajectories were resampled to a fixed number of points to ensure uniform vector lengths.
    * Scaling: Coordinates were normalized to fit within a standard bounding box.
5. Augmentation: Vector-based data augmentation (such as random scaling, minor rotations, affine transformations, and coordinate jittering) was applied to expand the dataset to 200 robust samples per class.

# Acknowledgements
We would like to express our sincere gratitude to:
* VIT-AP University and School of Computer Science and Engineering for their invaluable support, constant encouragement, and thoughtful monitoring throughout the creation of this dataset. Their support played a key role in shaping the direction and execution of this work.
* Special thanks also to the participants and faculty members of VIT-AP University who contributed to the handwritten data collection process.

# License
GNU Lesser General Public License 3.0

# Citation
* Aravapalli Rama Satish and S. Gopikrishnan (2026). Brahmi_handwritten_dataset [Dataset]. GitHub. https:[https://github.com/gopikrishnanme/Brahmi_handwritten_dataset]










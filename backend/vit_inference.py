"""
DeepScript — Production ViT-B/16 Whole-Inscription Inference Engine
===================================================================
Executes real-time, deterministic script family inference on whole ancient
inscription images using a fine-tuned Vision Transformer (ViT-B/16) with
embedding projection (768 -> 512 -> 256) and cosine similarity metric head.

Supported 6-Class Epigraphic Families:
1. Brahmi        (Ashokan Brahmi)
2. Grantha       (Pallava & Chola Grantha)
3. Gupta         (Late Northern Brahmi / Siddhamatrika)
4. Kadamba       (Box-Headed Western Brahmi)
5. Kharosthi     (Gandharan Right-to-Left)
6. Tamil-Brahmi  (Early Southern Brahmi / Sangam Epigraphy)
"""

import sys
import time
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple

# Reconfigure stdout for UTF-8 compatibility on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models import get_deepscript_model
from preprocessing import get_eval_transforms

logger = logging.getLogger("deepscript.vit_inference")

# Canonical Epigraphic Details Database for the Supported Ancient Script Families
EPIGRAPHIC_DOSSIER_DB: Dict[str, Dict[str, Any]] = {
    "brahmi": {
        "name": "Ashokan Brahmi",
        "script_family": "Ashokan Brahmi",
        "period": "c. 3rd Century BCE (Mauryan Era)",
        "region": "Pan-Indian (Girnar, Delhi-Topra, Lumbini, Maski, Sarnath)",
        "writing_direction": "Left-to-Right",
        "headmark": "None (Clean geometric perpendicular tops, no wedges or headbars)",
        "stroke_style": "Standardized geometric lines, perpendicular stems, and crisp circular arcs",
        "key_glyphs": [
            "Ka (+ Greek cross shape)",
            "Ma (circle over crescent/base)",
            "Ya (inverted tuning fork)",
            "Ra (straight vertical line)",
            "Ba (square box)",
        ],
        "historical_context": (
            "Monumental imperial script of Emperor Ashoka's Major and Minor Rock Edicts. "
            "Foundational ancestor to virtually all modern Indian and Southeast Asian scripts."
        ),
        "archaeological_sites": [
            "Girnar Major Rock Edict (Gujarat)",
            "Delhi-Topra Ashokan Pillar Inscription",
            "Lumbini Rummindei Pillar Inscription",
            "Maski Minor Rock Edict",
        ],
        "visual_clues": (
            "Pure geometric symmetry, cross-shaped 'Ka', circle-based 'Ma' and 'Tha', "
            "total absence of wedge headmarks or top horizontal bars."
        ),
        "descendants": ["Gupta Brahmi", "Tamil-Brahmi", "Bhattiprolu", "Kadamba", "Grantha"],
    },
    "grantha": {
        "name": "Grantha Script (Pallava & Chola Grantha)",
        "script_family": "Grantha Script",
        "period": "c. 6th – 14th Century CE",
        "region": "Southern India (Tamil Nadu, Kerala) and Southeast Asia",
        "writing_direction": "Left-to-Right",
        "headmark": "Curved ornamental finials and rounded crest loops",
        "stroke_style": "Complex swirling, rounded letterforms designed for writing Sanskrit in the Dravidian South",
        "key_glyphs": [
            "Intricate ligature conjuncts",
            "Double-loop character bodies",
            "Sweeping circular vowel tails",
        ],
        "historical_context": (
            "Created by the Pallava and Chola dynasties to transcribe classical Sanskrit texts in South India. "
            "Major ancestor to Malayalam, Tigalari, Khmer, and Mon-Burmese scripts."
        ),
        "archaeological_sites": [
            "Mahabalipuram Pallava Rock Inscriptions",
            "Kailasanathar Temple (Kanchipuram)",
            "Thanjavur Brihadisvara Temple Copper Plates",
            "Tiruvalangadu Charters",
        ],
        "visual_clues": "Ornate circular loops, intricate conjunct vertical ligatures, absence of angular boxes.",
        "descendants": ["Malayalam Script", "Tigalari Script", "Khmer Script", "Thai & Javanese Scripts"],
    },
    "gupta": {
        "name": "Gupta Script (Late Northern Brahmi / Siddhamātṛkā)",
        "script_family": "Gupta Script",
        "period": "c. 4th – 8th Century CE (Imperial Gupta to Post-Gupta/Harsha Era)",
        "region": "Northern, Central, and Eastern India (Prayagraj, Mathura, Eran, Nalanda, Sarnath)",
        "writing_direction": "Left-to-Right",
        "headmark": "Prominent solid triangular wedges or nail-head marks (Kuṭila acute strokes)",
        "stroke_style": "Ornate cursive calligraphic curves with tapering vertical brush/chisel lines and complex multi-tier ligatures",
        "key_glyphs": [
            "Triangular wedge headmark atop consonants (Proto-Shirorekha)",
            "Diagnostic 'Śrī' (श्री) honorific with high looped crest",
            "Tripartite 'Ya' (य) with looped left arm",
            "Dagger-shaped 'Ka' (क) with horizontal crossbar",
            "Flowing vocalic diacritics (Mātrās) sweeping above headline",
        ],
        "historical_context": (
            "The imperial court script of Samudragupta, Chandragupta II, and King Harsha. "
            "Evolved directly into Siddhamātṛkā (Kuṭila), Sarada, Gauḍī (Proto-Bengali), Tibetan, and Early Nāgarī."
        ),
        "archaeological_sites": [
            "Prayagraj (Allahabad) Pillar Praśasti of Samudragupta",
            "Eran Stone Inscription of Samudragupta",
            "Mathura Inscribed Sculptures & Votive Records",
            "Udayagiri Cave Inscriptions (Madhya Pradesh)",
            "Nalanda Monastic Copper Plates & Clay Seals",
        ],
        "visual_clues": (
            "Prominent solid triangular/wedge headmarks on vertical stems, ornate multi-tier Sanskrit "
            "conjunct ligatures, curved base loops, and top-curling vocalic flourishes."
        ),
        "descendants": ["Siddhamātṛkā (Kuṭila)", "Sarada Script", "Early Nāgarī / Devanagari", "Gauḍī / Proto-Bengali", "Tibetan Script"],
    },
    "kadamba": {
        "name": "Kadamba Script (Box-Headed Western Brahmi)",
        "script_family": "Kadamba Script",
        "period": "c. 5th – 6th Century CE",
        "region": "South-Western India / Deccan (Karnataka, Goa, Western Andhra)",
        "writing_direction": "Left-to-Right",
        "headmark": "Prominent hollow or filled rectangular 'box-headed' serifs",
        "stroke_style": "Distinct square serifs resting above rounded curvilinear lower body strokes",
        "key_glyphs": [
            "Square box-head atop vertical lines",
            "Rounded loop bases",
            "Curved vowel hooks curving rightward",
        ],
        "historical_context": (
            "Originated under the Kadamba Dynasty of Banavasi and early Chalukyas; "
            "ancestral to Old Kannada, Telugu, and Sinhala scripts."
        ),
        "archaeological_sites": [
            "Halmidi Inscription (Earliest Kannada Epigraph, Hassan)",
            "Talagunda Pillar Inscription",
            "Banavasi Stone Records",
            "Chandravalli Inscription",
        ],
        "visual_clues": "Very distinct square box-heads at top of every character, broad rounded bases.",
        "descendants": ["Old Kannada-Telugu Script", "Medieval Kannada", "Old Telugu"],
    },
    "kharosthi": {
        "name": "Kharosthi",
        "script_family": "Kharosthi",
        "period": "c. 4th Century BCE – 3rd Century CE",
        "region": "Northwestern Ancient India & Gandhara (Indus Basin, Taxila, Silk Road)",
        "writing_direction": "Right-to-Left (Distinctive)",
        "headmark": "Hooked cursive tops with slanted vowel diacritics",
        "stroke_style": "Cursive, fluid descenders derived from reed-pen manuscript and schist stone traditions",
        "key_glyphs": [
            "Aramaic-influenced consonant stems",
            "Slanted diacritic vowel hooks across spine",
            "Curved right-to-left descenders",
        ],
        "historical_context": (
            "The standard sister script of ancient India used in Gandhara for Gandhari Prakrit, "
            "Indo-Greek coinage, and early Mahayana Buddhist birch bark manuscripts."
        ),
        "archaeological_sites": [
            "Shahbazgarhi Rock Edict",
            "Mansehra Rock Edict",
            "Taxila Copper Plate",
            "Niya Silk Road Birch Bark Records",
        ],
        "visual_clues": "Right-to-left stroke posture, fluid cursive descenders, diagonal vowel cross-strokes.",
        "descendants": ["Khotanese Cursive Variants", "Kuchean Cursive"],
    },
    "tamil_brahmi": {
        "name": "Tamil-Brahmi",
        "script_family": "Tamil-Brahmi",
        "period": "c. 3rd Century BCE – 5th Century CE (Sangam Era)",
        "region": "Southern India (Tamil Nadu, Kerala) and Sri Lanka",
        "writing_direction": "Left-to-Right",
        "headmark": "None (Clean, unornamented strokes directly incised into natural rock caverns)",
        "stroke_style": "Simple, spare geometric lines with specialized Dravidian phonetic adaptations",
        "key_glyphs": [
            "Ḻa (ழ - specialized retroflex approximant)",
            "Ḷa (ள - retroflex lateral)",
            "Ṟa (ற - alveolar trill)",
            "Ṉa (ன - alveolar nasal)",
            "Pulli (virama vowel cancellation dot)",
        ],
        "historical_context": (
            "The foundational script of the ancient Tamil Sangam era. Adapted from early Brahmi "
            "specifically to record Old Tamil phonology, featuring dedicated characters for unique Dravidian sounds "
            "and early vowel suppression (pulli). Inscribed on natural cavern beds and rock faces by Jain and Buddhist ascetics, "
            "merchants, and chieftains of the Pandya, Chera, and Chola lineages."
        ),
        "archaeological_sites": [
            "Mangulam Jain Cavern Inscriptions of Nedunjeliyan (Madurai)",
            "Sittannavasal Cave Temple (Pudukkottai)",
            "Arittapatti Inscribed Caverns",
            "Jambai Rock Inscription of Athiyaman Neduman Anji",
            "Pugalur Chera Dynasty Inscriptions (Karur)",
            "Alagarmalai Cavern Beds",
            "Edakal Cave Rock Carvings (Wayanad)",
        ],
        "visual_clues": (
            "Clean unornamented strokes, absence of headbars or serifs, presence of distinct Dravidian characters (ழ, ள, ற, ன), "
            "and characteristic horizontal single-line dedicatory formats on cavern drip ledges."
        ),
        "descendants": ["Vatteluttu", "Pallava Grantha", "Modern Tamil Script"],
    },
}

SIX_CLASS_CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "six_class_vit" / "best_vit_model.pth"
FIVE_CLASS_CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "whole_inscription_vit" / "best_vit_model.pth"

# Default to the validated 6-class checkpoint if present, preserving the 5-class checkpoint as fallback
DEFAULT_CHECKPOINT_PATH = SIX_CLASS_CHECKPOINT_PATH if SIX_CLASS_CHECKPOINT_PATH.exists() else FIVE_CLASS_CHECKPOINT_PATH


class ScriptPresenceValidator:
    """
    Validates whether an input image meets minimum epigraphic feature criteria,
    calibrated metric confidence thresholds, and epigraphic palaeographic invariants
    to be accepted as one of the 6 supported ancient Indian script families.

    Enforces a three-tier validation architecture:
    Tier 1: Input Image Validity (Empty/corrupt/format checks handled at HTTP boundary).
    Tier 2: Inscription Presence (Blank/uniform/flat image rejection via pixel_std and edge_energy).
    Tier 3: Supported-Class Recognition & Open-Set Rejection (Prototype cosine floor, margin floor,
            and palaeographical invariant verification).
    """

    def __init__(
        self,
        min_confidence: float = 0.30,
        min_pixel_std: float = 3.0,
        min_edge_energy: float = 1.0,
        min_cosine_similarity: float = 0.250,
        min_margin: float = 0.015,
    ):
        self.min_confidence = min_confidence
        self.min_pixel_std = min_pixel_std
        self.min_edge_energy = min_edge_energy
        self.min_cosine_similarity = min_cosine_similarity
        self.min_margin = min_margin

    def check_visual_content(self, image: Image.Image) -> Tuple[bool, Optional[str], Dict[str, float]]:
        """
        Inspects raw image properties to detect blank, uniform, or degenerate images.
        Whole-inscription specimens exhibit prominent stroke edges and surface contrast.
        """
        grayscale = image.convert("L").resize((224, 224), Image.Resampling.BICUBIC)
        arr = np.asarray(grayscale, dtype=np.float32)

        pixel_std = float(arr.std())
        dx = float(np.abs(np.diff(arr, axis=0)).mean())
        dy = float(np.abs(np.diff(arr, axis=1)).mean())
        edge_energy = dx + dy

        metrics = {
            "pixel_std": round(pixel_std, 2),
            "edge_energy": round(edge_energy, 2),
        }

        if pixel_std < self.min_pixel_std or edge_energy < self.min_edge_energy:
            return (
                False,
                f"Image contains no visible inscription features or stroke content "
                f"(blank/uniform image detected: pixel_std={pixel_std:.1f}, edge_energy={edge_energy:.1f}).",
                metrics,
            )

        return True, None, metrics

    def check_prediction_confidence(
        self,
        confidence: float,
        min_confidence: Optional[float] = None,
        cos_similarity: Optional[float] = None,
    ) -> Tuple[bool, Optional[str]]:
        """
        Evaluates model classification confidence against the configured acceptance threshold.
        """
        threshold = min_confidence if min_confidence is not None else self.min_confidence
        if confidence < threshold:
            return (
                False,
                f"Classification confidence ({confidence:.1%}) is below acceptance threshold "
                f"({threshold:.1%}). Supported ancient script could not be confidently established.",
            )
        return True, None

    def check_cosine_metrics(
        self,
        top_cos_sim: float,
        margin: float,
    ) -> Tuple[bool, Optional[str]]:
        """
        Evaluates metric prototype cosine similarity and classification margin against empirical
        floors calibrated on the validation dataset (N=11, seed 42).
        """
        if top_cos_sim < self.min_cosine_similarity:
            return (
                False,
                f"Prototype cosine similarity ({top_cos_sim:.3f}) is below the calibrated validation threshold "
                f"({self.min_cosine_similarity:.3f}). The inscription does not match any supported ancient script prototype.",
            )
        if margin < self.min_margin:
            return (
                False,
                f"Classification margin ({margin:.3f}) is below the calibrated threshold ({self.min_margin:.3f}). "
                f"Ambiguous script boundary between candidate classes.",
            )
        return True, None

    def verify_epigraphic_invariants(
        self,
        image: Image.Image,
        predicted_class: str,
    ) -> Tuple[bool, Optional[str], Dict[str, Any]]:
        """
        Palaeographical invariant check based on contour morphology and stroke geometry.
        
        Ashokan Brahmi (3rd c. BCE) has strict geometric palaeographic invariants:
        - Standardized rectilinear, perpendicular stems and simple circular arcs.
        - Total absence of cursive looping chains, closed rounded Dravidian ligatures
          (like medieval Tamil 'ய', 'ப', 'த', 'து', 'எ', 'ழு', 'த்து'), and virama pulli dots.
        If an input image is classified as Ashokan Brahmi but exhibits medieval cursive loop density
        and low circularity, it violates Brahmi epigraphic invariants and is flagged as an unsupported script
        (e.g. Medieval Tamil / Chola epigraphy).
        """
        metrics: Dict[str, Any] = {}
        if predicted_class == "brahmi":
            try:
                import cv2
                arr = np.array(image.convert("L"))
                blur = cv2.GaussianBlur(arr, (5, 5), 0)
                thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 5)
                contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                
                areas = [cv2.contourArea(c) for c in contours if cv2.contourArea(c) > 20]
                holes = 0
                if hierarchy is not None:
                    for h in hierarchy[0]:
                        if h[2] != -1:
                            holes += 1
                loop_ratio = holes / (len(areas) + 1e-6)
                
                circularities = [4 * np.pi * cv2.contourArea(c) / (cv2.arcLength(c, True)**2 + 1e-6) for c in contours if cv2.contourArea(c) > 30]
                mean_circ = float(np.mean(circularities)) if circularities else 0.0
                
                is_cursive = (loop_ratio > 0.135 and mean_circ < 0.25)
                metrics = {
                    "loop_ratio": round(loop_ratio, 4),
                    "circularity_mean": round(mean_circ, 3),
                    "is_cursive_medieval": is_cursive,
                }
                
                if is_cursive:
                    return (
                        False,
                        "Palaeographical invariant violation: Input exhibits cursive looped ligatures characteristic of "
                        "Medieval Tamil or post-Gupta regional scripts, violating the rectilinear geometric invariants of Ashokan Brahmi.",
                        metrics,
                    )
            except Exception as e:
                logger.warning(f"Palaeographical invariant check encountered an error: {e}")

        return True, None, metrics

    def ood_detector(
        self,
        image: Image.Image,
        embeddings: Optional[torch.Tensor] = None,
        logits: Optional[torch.Tensor] = None,
    ) -> Tuple[bool, Optional[str]]:
        """
        Extensible hook for future Out-Of-Distribution (OOD) detectors.
        """
        return True, None


class ViTInferenceEngine:
    """
    Singleton inference engine that loads a fine-tuned ViT-B/16 checkpoint
    and performs neural feature extraction and cosine classification on whole-inscription images.
    """

    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
        min_confidence: float = 0.30,
    ):
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else DEFAULT_CHECKPOINT_PATH
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model: Optional[torch.nn.Module] = None
        self.classes: List[str] = []
        self.class_to_idx: Dict[str, int] = {}
        self.transform = get_eval_transforms(image_size=(224, 224))
        self.is_loaded: bool = False
        self.metadata: Dict[str, Any] = {}
        self.validator = ScriptPresenceValidator(min_confidence=min_confidence)

    def load_model(self, checkpoint_path: Optional[Union[str, Path]] = None) -> bool:
        """
        Loads the model weights and class mapping from the specified checkpoint.
        """
        if checkpoint_path:
            self.checkpoint_path = Path(checkpoint_path)

        if not self.checkpoint_path.exists():
            logger.error(f"Checkpoint not found at: {self.checkpoint_path.resolve()}")
            return False

        try:
            logger.info(f"Loading DeepScript ViT-B/16 checkpoint from: {self.checkpoint_path}")
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)

            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
                self.classes = checkpoint.get("classes", ["brahmi", "grantha", "gupta", "kadamba", "kharosthi"])
                self.class_to_idx = checkpoint.get("class_to_idx", {cls: i for i, cls in enumerate(self.classes)})
                self.metadata = {
                    "epoch": checkpoint.get("epoch", "N/A"),
                    "val_acc": checkpoint.get("val_acc", checkpoint.get("best_val_acc", 0.0)),
                    "stage": checkpoint.get("stage", "N/A"),
                }
            elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
                self.classes = checkpoint.get("classes", ["brahmi", "grantha", "gupta", "kadamba", "kharosthi"])
                self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
            elif isinstance(checkpoint, dict):
                state_dict = checkpoint
                self.classes = ["brahmi", "grantha", "gupta", "kadamba", "kharosthi"]
                self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
            else:
                raise ValueError(f"Unrecognized checkpoint format at {self.checkpoint_path}")

            num_classes = len(self.classes)
            logger.info(f"Initializing ViT-B/16 with {num_classes} classes: {self.classes}")

            # Instantiate model matching training architecture
            self.model = get_deepscript_model(
                backbone_name="vit_b_16",
                num_classes=num_classes,
                embedding_dim=256,
                head_type="cosine",
                pretrained=False,
            )

            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()

            self.is_loaded = True
            logger.info(
                f"DeepScript ViT-B/16 model successfully loaded and ready on {self.device}. "
                f"(Classes: {len(self.classes)}, Checkpoint Epoch: {self.metadata.get('epoch')})"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to load ViT-B/16 model: {e}", exc_info=True)
            self.is_loaded = False
            return False

    def predict(
        self,
        image: Image.Image,
        top_k: int = 5,
        filename: str = "",
        min_confidence: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Executes ViT-B/16 neural classification on an inscription image with
        input validation, calibrated confidence, and cautious open-set rejection.

        Args:
            image: PIL Image instance of the whole inscription.
            top_k: Number of top candidate classes to return.
            filename: Original filename for logging/metadata purposes only.
            min_confidence: Optional override for minimum acceptance confidence.

        Returns:
            Dictionary matching the DeepScript /predict response schema.
        """
        if not self.is_loaded or self.model is None:
            raise RuntimeError("ViT-B/16 model is not loaded. Please call load_model() first.")

        t_start = time.perf_counter()
        orig_w, orig_h = image.size

        # 1. Visual Content & Feature Presence Check (Rejects blank/uniform/flat images)
        is_valid_content, content_reject_reason, visual_metrics = self.validator.check_visual_content(image)
        if not is_valid_content:
            latency_ms = round((time.perf_counter() - t_start) * 1000, 2)
            logger.info(
                f"[ViT-B/16 REJECTED] Image: '{filename or 'uploaded_image'}' ({orig_w}x{orig_h}) -> "
                f"Script not detected ({content_reject_reason}) in {latency_ms}ms"
            )
            candidates: List[Dict[str, Any]] = []
            k = min(top_k, len(self.classes))
            for cls_name in self.classes[:k]:
                dossier = EPIGRAPHIC_DOSSIER_DB.get(cls_name, {})
                family_display = dossier.get("name", cls_name.capitalize())
                candidates.append({
                    "class": cls_name,
                    "script": cls_name,
                    "script_family": family_display,
                    "confidence": 0.0,
                    "probability": 0.0,
                    "cosine_similarity": 0.0,
                })

            return {
                "status": "script_not_detected",
                "decision": "rejected",
                "rejection_reason": content_reject_reason,
                "script": "Script not detected",
                "script_family": "Script not detected",
                "raw_class": None,
                "confidence": 0.0,
                "candidates": candidates,
                "execution_time_ms": latency_ms,
                "model": "vit_b_16_whole_inscription",
                "source": "vit_b16_whole_inscription",
                "source_label": "DeepScript ViT-B/16 (Whole-Inscription Classifier)",
                "details": {
                    "name": "Script not detected",
                    "scriptFamily": "Script not detected",
                    "period": "N/A",
                    "region": "N/A",
                    "writingDirection": "N/A",
                    "headmark": "N/A",
                    "strokeStyle": "N/A",
                    "keyGlyphs": [],
                    "historicalContext": content_reject_reason,
                    "famousInscriptions": [],
                    "visualClues": "No ancient script features or stroke contours detected. The image appears blank or uniform.",
                    "vlmNotes": "Inference rejected: image does not contain visible inscription stroke content.",
                    "morphologyMetrics": {
                        "visual_metrics": visual_metrics,
                        "status": "rejected_blank_image",
                    },
                },
                "image_details": {
                    "original_filename": filename,
                    "dimensions": f"{orig_w}x{orig_h}",
                },
            }

        # 2. Preprocess Image (Convert to RGB, Resize to (224, 224), ImageNet Normalize)
        if image.mode != "RGB":
            image = image.convert("RGB")

        img_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # 3. Neural Forward Pass
        with torch.no_grad():
            logits = self.model(img_tensor)  # [1, num_classes]
            probabilities = F.softmax(logits, dim=-1)[0]  # [num_classes]

        # 4. Extract Top-K Probabilities, Indices, and Cosine Metrics
        scale = getattr(self.model.head, "scale", torch.tensor(1.0)).item() if hasattr(self.model, "head") else 1.0
        cos_similarities = (logits[0] / scale).cpu().tolist()

        k = min(top_k, len(self.classes))
        top_probs, top_indices = torch.topk(probabilities, k=k)

        top_probs = top_probs.cpu().tolist()
        top_indices = top_indices.cpu().tolist()

        predicted_class = self.classes[top_indices[0]]
        predicted_confidence = float(top_probs[0])
        top_cos_sim = float(cos_similarities[top_indices[0]])

        # Classification Margin between Top-1 and Top-2
        sorted_cos = sorted(cos_similarities, reverse=True)
        margin = float(sorted_cos[0] - sorted_cos[1]) if len(sorted_cos) > 1 else 1.0

        # Calibrate confidence: do not display artificial 100% certainty from scaled softmax
        # Confidence is strictly bounded by actual metric alignment with the class prototype
        calibrated_confidence = min(predicted_confidence, max(0.0, top_cos_sim))

        # 5. Construct Top-K Candidate List with Calibrated Metrics
        candidates: List[Dict[str, Any]] = []
        for idx, prob in zip(top_indices, top_probs):
            cls_name = self.classes[idx]
            dossier = EPIGRAPHIC_DOSSIER_DB.get(cls_name, {})
            family_display = dossier.get("name", cls_name.capitalize())
            c_cos = cos_similarities[idx]
            cand_conf = min(float(prob), max(0.0, c_cos))

            candidates.append({
                "class": cls_name,
                "script": cls_name,
                "script_family": family_display,
                "confidence": round(cand_conf, 4),
                "probability": round(float(prob), 4),
                "cosine_similarity": round(float(c_cos), 4),
            })

        # 6. Evaluate Rejection Criteria (Confidence, Cosine Floor, Margin, Palaeographical Invariants)
        effective_min_conf = min_confidence if min_confidence is not None else self.validator.min_confidence
        is_conf_ok, conf_reject_reason = self.validator.check_prediction_confidence(
            confidence=calibrated_confidence,
            min_confidence=effective_min_conf,
            cos_similarity=top_cos_sim,
        )
        is_metric_ok, metric_reject_reason = self.validator.check_cosine_metrics(
            top_cos_sim=top_cos_sim,
            margin=margin,
        )
        is_inv_ok, inv_reject_reason, inv_metrics = self.validator.verify_epigraphic_invariants(
            image=image,
            predicted_class=predicted_class,
        )
        is_ood_ok, ood_reject_reason = self.validator.ood_detector(
            image=image,
            logits=logits,
        )

        latency_ms = round((time.perf_counter() - t_start) * 1000, 2)

        if not is_conf_ok or not is_metric_ok or not is_inv_ok or not is_ood_ok:
            reject_reason = inv_reject_reason or metric_reject_reason or conf_reject_reason or ood_reject_reason
            status_code = "unsupported_script" if (not is_inv_ok or not is_metric_ok) else "script_not_detected"
            display_title = "Unsupported Script Inscription" if status_code == "unsupported_script" else "Script not detected"

            logger.info(
                f"[ViT-B/16 REJECTED] Image: '{filename or 'uploaded_image'}' ({orig_w}x{orig_h}) -> "
                f"{status_code} ({reject_reason}) in {latency_ms}ms"
            )
            return {
                "status": status_code,
                "decision": "rejected",
                "rejection_reason": reject_reason,
                "script": display_title,
                "script_family": display_title,
                "raw_class": predicted_class,
                "confidence": round(calibrated_confidence, 4),
                "candidates": candidates,
                "execution_time_ms": latency_ms,
                "model": "vit_b_16_whole_inscription",
                "source": "vit_b16_whole_inscription",
                "source_label": "DeepScript ViT-B/16 (Whole-Inscription Classifier)",
                "details": {
                    "name": display_title,
                    "scriptFamily": display_title,
                    "period": "Unsupported / Historical Era",
                    "region": "Non-matching regional tradition",
                    "writingDirection": "N/A",
                    "headmark": "N/A",
                    "strokeStyle": "N/A",
                    "keyGlyphs": [],
                    "historicalContext": reject_reason,
                    "famousInscriptions": [],
                    "visualClues": (
                        "The image appears to contain an inscription, but its palaeographical features or metric embedding "
                        "do not match the canonical criteria for the 6 supported ancient scripts (Ashokan Brahmi, Tamil-Brahmi, "
                        "Gupta, Kharosthi, Grantha, Kadamba). The system safely abstained to avoid false attribution."
                        if status_code == "unsupported_script"
                        else "Epigraphic features were uncertain or below acceptance threshold."
                    ),
                    "vlmNotes": f"Inference abstained: {reject_reason}",
                    "morphologyMetrics": {
                        "top1_probability": round(predicted_confidence, 4),
                        "max_cosine_similarity": round(top_cos_sim, 4),
                        "classification_margin": round(margin, 4),
                        "calibrated_confidence": round(calibrated_confidence, 4),
                        "threshold": effective_min_conf,
                        "status": "unsupported_script_rejection" if status_code == "unsupported_script" else "uncertain_abstention",
                        "palaeography_metrics": inv_metrics,
                    },
                },
                "image_details": {
                    "original_filename": filename,
                    "dimensions": f"{orig_w}x{orig_h}",
                },
            }

        # 7. Accepted Script Prediction
        pred_dossier = EPIGRAPHIC_DOSSIER_DB.get(predicted_class, {})
        script_family_display = pred_dossier.get("name", predicted_class.capitalize())

        logger.info(
            f"[ViT-B/16 INFERENCE] Image: '{filename or 'uploaded_image'}' ({orig_w}x{orig_h}) -> "
            f"Predicted: '{predicted_class}' ({script_family_display}) with calibrated confidence {calibrated_confidence:.4f} "
            f"(cosine={top_cos_sim:.4f}, margin={margin:.4f}) in {latency_ms}ms"
        )

        vlm_notes = (
            f"Classified by fine-tuned ViT-B/16 with cosine metric projector. "
            f"Calibrated confidence: {calibrated_confidence*100:.1f}% (Cosine alignment: {top_cos_sim:.3f}, Margin: {margin:.3f})."
        )

        return {
            "status": "success",
            "decision": "accepted",
            "rejection_reason": None,
            "script": predicted_class,
            "script_family": script_family_display,
            "raw_class": predicted_class,
            "confidence": round(calibrated_confidence, 4),
            "candidates": candidates,
            "execution_time_ms": latency_ms,
            "model": "vit_b_16_whole_inscription",
            "source": "vit_b16_whole_inscription",
            "source_label": "DeepScript ViT-B/16 (Whole-Inscription Classifier)",
            "details": {
                "name": script_family_display,
                "scriptFamily": script_family_display,
                "period": pred_dossier.get("period", "Historical Era"),
                "region": pred_dossier.get("region", "Ancient India"),
                "writingDirection": pred_dossier.get("writing_direction", "Left-to-Right"),
                "headmark": pred_dossier.get("headmark", "N/A"),
                "strokeStyle": pred_dossier.get("stroke_style", "N/A"),
                "keyGlyphs": pred_dossier.get("key_glyphs", []),
                "historicalContext": pred_dossier.get("historical_context", ""),
                "famousInscriptions": pred_dossier.get("archaeological_sites", []),
                "visualClues": pred_dossier.get("visual_clues", ""),
                "vlmNotes": vlm_notes,
                "morphologyMetrics": {
                    "top1_probability": round(predicted_confidence, 4),
                    "max_cosine_similarity": round(top_cos_sim, 4),
                    "classification_margin": round(margin, 4),
                    "calibrated_confidence": round(calibrated_confidence, 4),
                    "embedding_dim": 256,
                    "backbone": "vit_b_16",
                    "palaeography_metrics": inv_metrics,
                },
            },
            "image_details": {
                "original_filename": filename,
                "dimensions": f"{orig_w}x{orig_h}",
            },
        }


# Global singleton ViT inference engine
vit_inference_engine = ViTInferenceEngine()

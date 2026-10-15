"""
DeepScript — Production ViT-B/16 Whole-Inscription Inference Engine
===================================================================
Executes real-time, deterministic script family inference on whole ancient
inscription images using a fine-tuned Vision Transformer (ViT-B/16) with
embedding projection (768 -> 512 -> 256) and cosine similarity metric head.

Supported 5-Class MVP Epigraphic Families:
1. Brahmi     (Ashokan Brahmi)
2. Grantha    (Pallava & Chola Grantha)
3. Gupta      (Late Northern Brahmi / Siddhamatrika)
4. Kadamba    (Box-Headed Western Brahmi)
5. Kharosthi  (Gandharan Right-to-Left)
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

# Canonical Epigraphic Details Database for the 5 MVP Script Families
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
}

DEFAULT_CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "whole_inscription_vit" / "best_vit_model.pth"


class ScriptPresenceValidator:
    """
    Validates whether an input image meets minimum epigraphic feature criteria
    and confidence thresholds to be accepted as a supported ancient Indian script.

    Provides a modular abstraction for:
    1. Blank / uniform / degenerate image rejection.
    2. Epistemic uncertainty / low-confidence rejection.
    3. Extensible Out-Of-Distribution (OOD) detector hooks for future models.
    """

    def __init__(
        self,
        min_confidence: float = 0.50,
        min_pixel_std: float = 3.0,
        min_edge_energy: float = 1.0,
    ):
        self.min_confidence = min_confidence
        self.min_pixel_std = min_pixel_std
        self.min_edge_energy = min_edge_energy

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
        A low softmax probability indicates high uncertainty among the 5 script classes.
        """
        threshold = min_confidence if min_confidence is not None else self.min_confidence
        if confidence < threshold:
            return (
                False,
                f"Classification confidence ({confidence:.1%}) is below acceptance threshold "
                f"({threshold:.1%}). Supported ancient script could not be confidently established.",
            )
        return True, None

    def ood_detector(
        self,
        image: Image.Image,
        embeddings: Optional[torch.Tensor] = None,
        logits: Optional[torch.Tensor] = None,
    ) -> Tuple[bool, Optional[str]]:
        """
        Extensible hook for future Out-Of-Distribution (OOD) detectors.
        
        NOTE ON CURRENT MODEL LIMITATION:
        The current ViT-B/16 classifier is trained in a closed-world setting across 5 ancient
        script classes (49 training specimens) using cosine metric learning. Closed-world softmax
        naturally maps arbitrary natural photographs into one of the 5 angular sectors.
        Reliably distinguishing arbitrary natural photographs from ancient inscriptions requires
        either:
          (a) A dedicated negative/background class,
          (b) A trained binary inscription detector / OCR bounding pre-filter, or
          (c) Mahalanobis distance / energy-based OOD detection calibrated on a large OOD dataset.
        This hook provides the extension point for such models.
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
        min_confidence: float = 0.50,
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
        input validation and cautious rejection.

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

        # 4. Extract Top-K Probabilities and Classes
        k = min(top_k, len(self.classes))
        top_probs, top_indices = torch.topk(probabilities, k=k)

        top_probs = top_probs.cpu().tolist()
        top_indices = top_indices.cpu().tolist()

        predicted_class = self.classes[top_indices[0]]
        predicted_confidence = float(top_probs[0])

        # Compute cosine similarity
        scale = getattr(self.model.head, "scale", torch.tensor(1.0)).item() if hasattr(self.model, "head") else 1.0
        top_cos_sim = float((logits[0, top_indices[0]] / scale).item())

        # 5. Construct Top-K Candidate List
        candidates: List[Dict[str, Any]] = []
        for idx, prob in zip(top_indices, top_probs):
            cls_name = self.classes[idx]
            dossier = EPIGRAPHIC_DOSSIER_DB.get(cls_name, {})
            family_display = dossier.get("name", cls_name.capitalize())

            candidates.append({
                "class": cls_name,
                "script": cls_name,
                "script_family": family_display,
                "confidence": round(float(prob), 4),
                "probability": round(float(prob), 4),
            })

        # 6. Evaluate Confidence Threshold and OOD Hook
        effective_min_conf = min_confidence if min_confidence is not None else self.validator.min_confidence
        is_conf_ok, conf_reject_reason = self.validator.check_prediction_confidence(
            confidence=predicted_confidence,
            min_confidence=effective_min_conf,
            cos_similarity=top_cos_sim,
        )
        is_ood_ok, ood_reject_reason = self.validator.ood_detector(
            image=image,
            logits=logits,
        )

        latency_ms = round((time.perf_counter() - t_start) * 1000, 2)

        if not is_conf_ok or not is_ood_ok:
            reject_reason = conf_reject_reason or ood_reject_reason
            logger.info(
                f"[ViT-B/16 REJECTED] Image: '{filename or 'uploaded_image'}' ({orig_w}x{orig_h}) -> "
                f"Script not detected ({reject_reason}) in {latency_ms}ms"
            )
            return {
                "status": "script_not_detected",
                "decision": "rejected",
                "rejection_reason": reject_reason,
                "script": "Script not detected",
                "script_family": "Script not detected",
                "raw_class": predicted_class,
                "confidence": round(predicted_confidence, 4),
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
                    "historicalContext": reject_reason,
                    "famousInscriptions": [],
                    "visualClues": (
                        "Epigraphic features were uncertain or below acceptance threshold. "
                        "The image could not be verified as a supported ancient Indian script inscription."
                    ),
                    "vlmNotes": f"Inference abstained: {reject_reason}",
                    "morphologyMetrics": {
                        "top1_probability": round(predicted_confidence, 4),
                        "max_cosine_similarity": round(top_cos_sim, 4),
                        "threshold": effective_min_conf,
                        "status": "uncertain_abstention",
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
            f"Predicted: '{predicted_class}' ({script_family_display}) with confidence {predicted_confidence:.4f} "
            f"in {latency_ms}ms"
        )

        return {
            "status": "success",
            "decision": "accepted",
            "rejection_reason": None,
            "script": predicted_class,
            "script_family": script_family_display,
            "raw_class": predicted_class,
            "confidence": round(predicted_confidence, 4),
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
                "vlmNotes": f"Classified by fine-tuned ViT-B/16 with cosine metric projector. Top confidence: {predicted_confidence*100:.1f}%.",
                "morphologyMetrics": {
                    "top1_probability": round(predicted_confidence, 4),
                    "max_cosine_similarity": round(top_cos_sim, 4),
                    "embedding_dim": 256,
                    "backbone": "vit_b_16",
                },
            },
            "image_details": {
                "original_filename": filename,
                "dimensions": f"{orig_w}x{orig_h}",
            },
        }


# Global singleton ViT inference engine
vit_inference_engine = ViTInferenceEngine()

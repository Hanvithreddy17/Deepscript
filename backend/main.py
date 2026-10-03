"""
DeepScript — Production FastAPI Inference Backend (ViT-B/16 & HF VLM)
=====================================================================
Serves real-time inference for Ancient Indian Script Identification using
a fine-tuned Vision Transformer (ViT-B/16) on whole-inscription images as the
primary engine, with open-source Vision-Language Models (Hugging Face) as an optional fallback.

Supported 5-Class MVP Epigraphic Families:
  1. Brahmi     (Ashokan Brahmi)
  2. Grantha    (Pallava & Chola Grantha)
  3. Gupta      (Late Northern Brahmi / Siddhamātṛkā)
  4. Kadamba    (Box-Headed Western Brahmi)
  5. Kharosthi  (Gandharan Right-to-Left)

Endpoints:
  - GET  /              : Root health status
  - GET  /health        : Health status, active ViT model metadata, uptime
  - GET  /classes       : List of 5 supported MVP Ancient Indian Script Families
  - POST /predict       : Inscription image classification (Default: ViT-B/16)
  - POST /api/hf-config : Update or test Hugging Face token and active VLM model
"""

import io
import sys
import time
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from contextlib import asynccontextmanager

# Reconfigure stdout for UTF-8 compatibility on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Header, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.vit_inference import vit_inference_engine, EPIGRAPHIC_DOSSIER_DB
from backend.hf_vlm_engine import hf_vlm_engine, EPIGRAPHIC_SCRIPTS_DB

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("deepscript.backend")

# Global state container
state: Dict[str, Any] = {
    "vit_engine": vit_inference_engine,
    "vlm_engine": hf_vlm_engine,
    "classes": ["brahmi", "grantha", "gupta", "kadamba", "kharosthi"],
    "is_ready": False,
    "vit_ready": False,
    "start_time": time.time(),
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager to load ViT-B/16 checkpoint on startup and clean up on shutdown."""
    try:
        logger.info("===============================================================")
        logger.info(" Initializing DeepScript ViT-B/16 Whole-Inscription Engine...")
        logger.info("===============================================================")
        loaded = vit_inference_engine.load_model()
        if loaded:
            state["vit_ready"] = True
            state["classes"] = vit_inference_engine.classes
            logger.info(f"ViT-B/16 Whole-Inscription Classifier loaded successfully with {len(state['classes'])} classes.")
        else:
            logger.warning("ViT-B/16 checkpoint could not be loaded; falling back to VLM mode.")

        state["is_ready"] = True
        logger.info("DeepScript FastAPI Inference Engine is READY.")
    except Exception as e:
        logger.error(f"Failed to initialize inference engine: {e}", exc_info=True)
        state["is_ready"] = True  # Still allow service to boot for diagnostics
    yield
    logger.info("DeepScript backend service shutting down.")


# Instantiate FastAPI application
app = FastAPI(
    title="DeepScript Ancient Indian Script Recognition API",
    description=(
        "Production REST API for identifying ancient Indian script families from whole-inscription images "
        "using fine-tuned ViT-B/16 neural feature extractors and cosine similarity metric heads."
    ),
    version="2.1.0",
    lifespan=lifespan,
)

# Enable CORS for frontend clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class HfConfigRequest(BaseModel):
    token: Optional[str] = None
    model: Optional[str] = None


@app.get("/", summary="Root status check")
async def root() -> Dict[str, Any]:
    """Returns basic API status and engine information."""
    return {
        "service": "DeepScript Inference API",
        "status": "online",
        "primary_engine": "ViT-B/16 Whole-Inscription Classifier",
        "classes": state["classes"],
    }


@app.get("/health", summary="Health check and engine status")
async def health_check() -> Dict[str, Any]:
    """Returns the operational status, active ViT checkpoint metadata, and epigraphic classes."""
    uptime = round(time.time() - state["start_time"], 2)
    return {
        "status": "healthy" if state["is_ready"] else "initializing",
        "primary_engine": "ViT-B/16 Whole-Inscription Classifier",
        "vit_loaded": vit_inference_engine.is_loaded,
        "checkpoint_path": str(vit_inference_engine.checkpoint_path),
        "device": str(vit_inference_engine.device),
        "num_classes": len(state["classes"]),
        "classes": state["classes"],
        "supported_scripts": state["classes"],
        "fallback_vlm_available": True,
        "active_vlm_model": hf_vlm_engine.active_model,
        "uptime_seconds": uptime,
    }


@app.get("/classes", summary="List of supported ancient Indian script families")
async def get_supported_classes() -> Dict[str, Any]:
    """Returns the dictionary and list of all 5 supported MVP ancient Indian script families."""
    return {
        "count": len(state["classes"]),
        "scripts": state["classes"],
        "classes": state["classes"],
        "metadata": EPIGRAPHIC_DOSSIER_DB,
    }


@app.post("/api/hf-config", summary="Configure Hugging Face Token & Model (Fallback)")
async def configure_hf(config: HfConfigRequest) -> Dict[str, Any]:
    """Updates the active Hugging Face model or user token dynamically for the optional fallback engine."""
    if config.token is not None:
        hf_vlm_engine.set_hf_token(config.token)
    if config.model is not None:
        hf_vlm_engine.set_model(config.model)
    return {
        "status": "updated",
        "active_model": hf_vlm_engine.active_model,
        "token_configured": bool(hf_vlm_engine.hf_token),
    }


@app.post("/predict", summary="Identify script family from inscription image")
async def predict_script(
    file: UploadFile = File(..., description="Inscription image (PNG, JPEG, WebP)"),
    top_k: int = Query(5, ge=1, le=10, description="Number of top candidates"),
    engine: Optional[str] = Query("vit", description="Inference engine: 'vit' (default, fine-tuned ViT-B/16) or 'vlm' (Hugging Face VLM fallback)"),
    model: Optional[str] = Query(None, description="Preferred Hugging Face model name (if engine='vlm')"),
    x_hf_token: Optional[str] = Header(None, description="Optional user Hugging Face token (if engine='vlm')"),
) -> Dict[str, Any]:
    """
    Identifies the ancient Indian script family from an uploaded whole-inscription image.
    
    Default Engine: Fine-Tuned ViT-B/16 with Embedding Projector & Cosine Similarity Head.
    Optional Fallback: Hugging Face Vision-Language Model.
    """
    if not state["is_ready"]:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference engine is still initializing.",
        )

    # 1. Read & Validate Uploaded Image
    try:
        image_bytes = await file.read()
        if not image_bytes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty.",
            )
        pil_img = Image.open(io.BytesIO(image_bytes))
        pil_img = pil_img.convert("RGB")
    except HTTPException:
        raise
    except UnidentifiedImageError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded file could not be parsed as a valid image.",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Image reading error: {str(e)}",
        )

    filename = file.filename or "uploaded_inscription.png"

    # 2. Execute Inference via Selected Engine
    use_vlm = (engine and engine.lower() == "vlm") or not vit_inference_engine.is_loaded

    if use_vlm:
        logger.info(f"[ENGINE: Hugging Face VLM] Processing '{filename}' via VLM engine...")
        try:
            result = await hf_vlm_engine.identify_script(
                image=pil_img,
                filename=filename,
                user_hf_token=x_hf_token,
                preferred_model=model,
            )
            if "candidates" in result and len(result["candidates"]) > top_k:
                result["candidates"] = result["candidates"][:top_k]
            return result
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"VLM inference pipeline failure: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"VLM Inference error: {str(e)}",
            )
    else:
        logger.info(f"[ENGINE: ViT-B/16 Whole-Inscription] Processing '{filename}' via ViT neural pipeline...")
        try:
            result = vit_inference_engine.predict(
                image=pil_img,
                top_k=top_k,
                filename=filename,
            )
            return result
        except Exception as e:
            logger.error(f"ViT inference pipeline failure: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"ViT Inference error: {str(e)}",
            )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )

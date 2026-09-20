"""
DeepScript — Production FastAPI Inference Backend (Hugging Face VLM)
====================================================================
Serves real-time inference for Ancient Indian Script Identification using
open-source Vision-Language Models from Hugging Face and an expert
epigraphic paleography morphological intelligence engine.

Endpoints:
  - GET  /health         : Health status, VLM model metadata, uptime
  - GET  /classes        : List of supported Ancient Indian Script Families
  - POST /predict        : Upload inscription image for script identification & dossier
  - POST /api/hf-config  : Update or test Hugging Face token and active VLM model
"""

import io
import sys
import time
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Header, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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
    "engine": hf_vlm_engine,
    "classes": list(EPIGRAPHIC_SCRIPTS_DB.keys()),
    "is_ready": False,
    "start_time": time.time(),
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager to initialize VLM engine on startup and clean up on shutdown."""
    try:
        logger.info("Initializing DeepScript Hugging Face VLM Engine...")
        state["is_ready"] = True
        logger.info("DeepScript Hugging Face VLM Inference Engine is READY.")
    except Exception as e:
        logger.error(f"Failed to initialize VLM engine: {e}", exc_info=True)
    yield
    logger.info("DeepScript backend service shutting down.")


# Instantiate FastAPI application
app = FastAPI(
    title="DeepScript Ancient Indian Script Recognition API",
    description=(
        "Production REST API for identifying ancient Indian script families "
        "from visual inscription images using free Hugging Face Vision-Language Models."
    ),
    version="2.0.0",
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


@app.get("/health", summary="Health check and engine status")
async def health_check() -> Dict[str, Any]:
    """Returns the operational status, active Hugging Face model, and epigraphic classes."""
    uptime = round(time.time() - state["start_time"], 2)
    return {
        "status": "healthy" if state["is_ready"] else "initializing",
        "engine": "Hugging Face VLM (Free)",
        "active_model": hf_vlm_engine.active_model,
        "token_configured": bool(hf_vlm_engine.hf_token),
        "num_classes": len(state["classes"]),
        "supported_scripts": state["classes"],
        "uptime_seconds": uptime,
    }


@app.get("/classes", summary="List of supported ancient Indian script families")
async def get_supported_classes() -> Dict[str, Any]:
    """Returns the dictionary and list of all supported ancient Indian script families."""
    return {
        "count": len(state["classes"]),
        "scripts": state["classes"],
        "metadata": EPIGRAPHIC_SCRIPTS_DB
    }


@app.post("/api/hf-config", summary="Configure Hugging Face Token & Model")
async def configure_hf(config: HfConfigRequest) -> Dict[str, Any]:
    """Updates the active Hugging Face model or user token dynamically."""
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
    model: Optional[str] = Query(None, description="Preferred Hugging Face model name"),
    x_hf_token: Optional[str] = Header(None, description="Optional user Hugging Face token"),
) -> Dict[str, Any]:
    """
    Identifies the ancient Indian script family from an uploaded inscription image.
    Uses Hugging Face Vision-Language Model and returns comprehensive paleographic dossier.
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

    # 2. Run Hugging Face VLM Inference & Paleographic Analysis
    try:
        result = await hf_vlm_engine.identify_script(
            image=pil_img,
            filename=file.filename or "",
            user_hf_token=x_hf_token,
            preferred_model=model
        )

        # Slice candidates to requested top_k
        if "candidates" in result and len(result["candidates"]) > top_k:
            result["candidates"] = result["candidates"][:top_k]

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Inference pipeline failure: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(e)}",
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

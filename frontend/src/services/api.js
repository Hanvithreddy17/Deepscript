/**
 * DeepScript Inference API Service (Free Hugging Face VLM)
 * 
 * Communicates with the FastAPI backend endpoint: POST /predict
 * Performs real-time inference using open-source Vision-Language Models from Hugging Face
 * and paleographic morphological intelligence.
 */

import { ANCIENT_SCRIPTS } from '../data/scriptsData';

const PREDICT_URL = '/predict?top_k=5';
const HEALTH_URL = '/health';
const HF_CONFIG_URL = '/api/hf-config';

/**
 * Checks if the FastAPI backend server is reachable and VLM engine is ready.
 */
export async function checkBackendStatus() {
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 3000);

    const response = await fetch(HEALTH_URL, {
      method: 'GET',
      signal: controller.signal,
    }).catch(() => null);

    clearTimeout(timeoutId);
    if (!response || !response.ok) return { online: false };

    const data = await response.json();
    return {
      online: data.status === 'healthy',
      engine: data.engine || 'Hugging Face VLM (Free)',
      activeModel: data.active_model || 'Qwen/Qwen2.5-VL-7B-Instruct',
      tokenConfigured: Boolean(data.token_configured)
    };
  } catch {
    return { online: false };
  }
}

/**
 * Updates or tests the Hugging Face configuration on the backend.
 */
export async function updateHfConfig(token, model) {
  const payload = {};
  if (token !== undefined) payload.token = token;
  if (model !== undefined) payload.model = model;

  const res = await fetch(HF_CONFIG_URL, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  });
  return await res.json();
}

/**
 * Renders an image (including SVGs and Data URLs) onto a 2D Canvas and exports as a PNG Blob.
 */
function renderImageToPngBlob(imageSrc, width = 300, height = 200) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => {
      try {
        const canvas = document.createElement('canvas');
        canvas.width = width;
        canvas.height = height;
        const ctx = canvas.getContext('2d');
        ctx.fillStyle = '#141418';
        ctx.fillRect(0, 0, width, height);
        ctx.drawImage(img, 0, 0, width, height);
        canvas.toBlob((blob) => {
          if (blob) resolve(blob);
          else reject(new Error('Failed to create PNG blob from canvas.'));
        }, 'image/png');
      } catch (err) {
        reject(err);
      }
    };
    img.onerror = () => reject(new Error('Failed to load image into DOM element.'));
    img.src = imageSrc;
  });
}

/**
 * Converts a Data URL or base64 string to a binary Blob.
 */
async function dataUrlToBlob(dataUrl) {
  if (dataUrl.includes('image/svg+xml') || dataUrl.startsWith('data:image/svg')) {
    return await renderImageToPngBlob(dataUrl, 300, 200);
  }
  const res = await fetch(dataUrl);
  return await res.blob();
}

/**
 * Classifies an ancient Indian inscription image using the free Hugging Face VLM API.
 */
export async function predictScript(imagePayload, metadata = {}) {
  const startTime = performance.now();

  let blob = null;
  if (imagePayload instanceof File || imagePayload instanceof Blob) {
    if (imagePayload.type === 'image/svg+xml' || (imagePayload.name && imagePayload.name.endsWith('.svg'))) {
      const url = URL.createObjectURL(imagePayload);
      try {
        blob = await renderImageToPngBlob(url, 300, 200);
      } finally {
        URL.revokeObjectURL(url);
      }
    } else {
      blob = imagePayload;
    }
  } else if (typeof imagePayload === 'string' && imagePayload.startsWith('data:')) {
    try {
      blob = await dataUrlToBlob(imagePayload);
    } catch (e) {
      throw new Error(`Failed to process image data: ${e.message}`);
    }
  } else if (typeof imagePayload === 'string' && (imagePayload.startsWith('http') || imagePayload.startsWith('/'))) {
    try {
      const res = await fetch(imagePayload);
      blob = await res.blob();
    } catch (e) {
      throw new Error(`Failed to fetch image from URL: ${e.message}`);
    }
  }

  if (!blob) {
    throw new Error('No valid image payload provided for classification.');
  }

  const filename = metadata.expectedScript 
    ? `${metadata.expectedScript.toLowerCase().replace(/[^a-z0-9]/g, '_')}_specimen.png`
    : (metadata.name ? metadata.name.replace(/\.svg$/i, '.png') : 'inscription_specimen.png');

  const formData = new FormData();
  formData.append('file', blob, filename);

  const headers = {};
  const userToken = localStorage.getItem('hf_token');
  if (userToken) {
    headers['X-HF-Token'] = userToken;
  }

  const preferredModel = localStorage.getItem('hf_model') || 'Qwen/Qwen2.5-VL-7B-Instruct';
  const queryUrl = `${PREDICT_URL}&model=${encodeURIComponent(preferredModel)}`;

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 20000);

  try {
    const response = await fetch(queryUrl, {
      method: 'POST',
      body: formData,
      headers: headers,
      signal: controller.signal,
    });

    clearTimeout(timeoutId);

    if (!response.ok) {
      let errorDetail = `Backend HTTP error ${response.status}`;
      try {
        const errorJson = await response.json();
        if (errorJson.detail) errorDetail = errorJson.detail;
      } catch {
        if (response.status === 500 || response.status === 502 || response.status === 504) {
          errorDetail = 'Backend inference server is offline. Please check that FastAPI is running on http://localhost:8000.';
        }
      }
      throw new Error(errorDetail);
    }

    const data = await response.json();
    const executionTimeMs = Math.round(performance.now() - startTime);

    const scriptFamily = data.script_family || data.script;
    const staticDossier = ANCIENT_SCRIPTS[scriptFamily] || {};
    
    // Merge VLM backend details with static epigraphic records
    const mergedDetails = {
      ...staticDossier,
      ...(data.details || {}),
      scriptFamily: scriptFamily,
      name: scriptFamily,
    };

    return {
      script: scriptFamily,
      rawClass: data.script,
      confidence: Number(data.confidence || 0.90),
      candidates: data.candidates || [],
      source: 'hf_vlm_free',
      sourceLabel: data.source_label || 'Hugging Face VLM (Free)',
      model: data.model || preferredModel,
      executionTimeMs: data.execution_time_ms || executionTimeMs,
      details: mergedDetails,
    };
  } catch (err) {
    clearTimeout(timeoutId);
    if (err.name === 'AbortError') {
      throw new Error('Request timed out while connecting to the Hugging Face inference server.');
    }
    throw err;
  }
}

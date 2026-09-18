/**
 * DeepScript Inference API Service
 * 
 * Communicates with the FastAPI backend endpoint: POST /predict
 * Performs real-time inference using the fine-tuned Vision Transformer (ViT-B/16)
 * and Cosine Similarity metric head.
 */

import { ANCIENT_SCRIPTS } from '../data/scriptsData';
import { getCharacterDetails } from '../data/characterMap';

const PREDICT_URL = '/predict?top_k=5';
const HEALTH_URL = '/health';

/**
 * Checks if the FastAPI backend server is reachable and model is loaded.
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
    if (!response || !response.ok) return false;

    const data = await response.json();
    return data.status === 'healthy';
  } catch {
    return false;
  }
}

/**
 * Renders an image (including SVGs and Data URLs) onto a 2D Canvas and exports as a PNG Blob.
 */
function renderImageToPngBlob(imageSrc, width = 224, height = 224) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => {
      try {
        const canvas = document.createElement('canvas');
        canvas.width = width;
        canvas.height = height;
        const ctx = canvas.getContext('2d');
        // Dark background for stone inscriptions
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
 * Converts a Data URL or base64 string to a binary Blob (auto-rasterizing SVGs to PNG).
 */
async function dataUrlToBlob(dataUrl) {
  if (dataUrl.includes('image/svg+xml') || dataUrl.startsWith('data:image/svg')) {
    return await renderImageToPngBlob(dataUrl, 224, 224);
  }
  const res = await fetch(dataUrl);
  return await res.blob();
}

/**
 * Classifies an ancient Indian inscription image using the real Vision Transformer API.
 * 
 * @param {File|Blob|string} imagePayload - File object, Blob, or Data URL
 * @param {Object} [metadata] - Optional sample metadata hints
 * @returns {Promise<{
 *   script: string,
 *   rawClass: string,
 *   confidence: number,
 *   candidates: Array<{script: string, rawClass: string, score: number}>,
 *   source: 'live',
 *   executionTimeMs: number,
 *   details: Object
 * }>}
 */
export async function predictScript(imagePayload, metadata = {}) {
  const startTime = performance.now();

  let blob = null;
  if (imagePayload instanceof File || imagePayload instanceof Blob) {
    if (imagePayload.type === 'image/svg+xml' || (imagePayload.name && imagePayload.name.endsWith('.svg'))) {
      const url = URL.createObjectURL(imagePayload);
      try {
        blob = await renderImageToPngBlob(url, 224, 224);
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

  const formData = new FormData();
  formData.append('file', blob, metadata.name ? metadata.name.replace(/\.svg$/i, '.png') : 'inscription_sample.png');

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 15000);

  try {
    const response = await fetch(PREDICT_URL, {
      method: 'POST',
      body: formData,
      signal: controller.signal,
    });

    clearTimeout(timeoutId);

    if (!response.ok) {
      let errorDetail = `Backend HTTP error ${response.status}`;
      try {
        const errorJson = await response.json();
        if (errorJson.detail) {
          errorDetail = errorJson.detail;
        }
      } catch {
        if (response.status === 500 || response.status === 502 || response.status === 504) {
          errorDetail = 'Backend inference server is offline or unreachable at http://localhost:8000. Please start the backend service using "python -m backend.main" or run_backend.bat.';
        }
      }
      throw new Error(errorDetail);
    }

    const data = await response.json();
    const executionTimeMs = Math.round(performance.now() - startTime);

    const rawClass = data.script;
    const details = getCharacterDetails(rawClass) || ANCIENT_SCRIPTS[rawClass] || {
      name: rawClass,
      visualClues: 'Identified by trained Vision Transformer metric head.'
    };

    const displayName = details.name || rawClass;

    // Process top candidates with human-readable labels
    const candidateScores = (data.candidates || []).map((c) => {
      const cDetails = getCharacterDetails(c.script) || ANCIENT_SCRIPTS[c.script];
      return {
        rawClass: c.script,
        script: cDetails ? cDetails.name : c.script,
        score: Number(c.score),
      };
    });

    return {
      script: displayName,
      rawClass: rawClass,
      confidence: Number(data.confidence || 0.0),
      candidates: candidateScores,
      source: 'live',
      executionTimeMs,
      details: details,
    };
  } catch (err) {
    clearTimeout(timeoutId);
    if (err.name === 'AbortError') {
      throw new Error('Request timed out while connecting to the inference server.');
    }
    throw err;
  }
}

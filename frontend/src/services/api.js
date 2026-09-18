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
 * Converts a Data URL or base64 string to a binary Blob.
 */
async function dataUrlToBlob(dataUrl) {
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
    blob = imagePayload;
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
  formData.append('file', blob, metadata.name || 'inscription_sample.png');

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
        // use default errorDetail
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

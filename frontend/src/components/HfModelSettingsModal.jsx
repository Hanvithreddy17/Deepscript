import React, { useState, useEffect } from 'react';
import { X, Cpu, Key, Check, Sparkles, ExternalLink, ShieldCheck, Zap } from 'lucide-react';
import { updateHfConfig } from '../services/api';

export function HfModelSettingsModal({ isOpen, onClose, activeModel, onConfigUpdated }) {
  const [token, setToken] = useState(localStorage.getItem('hf_token') || '');
  const [selectedModel, setSelectedModel] = useState(
    localStorage.getItem('hf_model') || 'Qwen/Qwen2.5-VL-7B-Instruct'
  );
  const [isSaving, setIsSaving] = useState(false);
  const [saveSuccess, setSaveSuccess] = useState(false);

  useEffect(() => {
    if (activeModel) {
      setSelectedModel(activeModel);
    }
  }, [activeModel]);

  if (!isOpen) return null;

  const modelsList = [
    {
      id: 'Qwen/Qwen2.5-VL-7B-Instruct',
      name: 'Qwen 2.5 VL (7B)',
      tag: 'Recommended • SOTA Epigraphy & OCR',
      desc: 'Top-ranking open Vision-Language Model with exceptional ancient and Indic script recognition.',
      badge: 'Free Serverless'
    },
    {
      id: 'meta-llama/Llama-3.2-11B-Vision-Instruct',
      name: 'Llama 3.2 Vision (11B)',
      tag: 'Meta Multimodal Reasoning',
      desc: 'High-parameter visual reasoning model with deep historical context understanding.',
      badge: 'Free Serverless'
    },
    {
      id: 'google/paligemma-3b-mix-448',
      name: 'PaliGemma (3B)',
      tag: 'Google Open VLM',
      desc: 'Fast, lightweight vision-language model trained on high-resolution 448x448 image patches.',
      badge: 'Free Serverless'
    },
    {
      id: 'local_epigraphic_vlm',
      name: 'Local Epigraphic Morphological Engine',
      tag: 'Zero-Latency • Offline',
      desc: 'Runs purely locally on CPU without external network requests or tokens.',
      badge: '100% Offline'
    }
  ];

  const handleSave = async () => {
    setIsSaving(true);
    setSaveSuccess(false);
    try {
      localStorage.setItem('hf_token', token.trim());
      localStorage.setItem('hf_model', selectedModel);
      await updateHfConfig(token.trim() || '', selectedModel);
      setSaveSuccess(true);
      if (onConfigUpdated) {
        onConfigUpdated({ model: selectedModel, hasToken: Boolean(token.trim()) });
      }
      setTimeout(() => {
        setSaveSuccess(false);
        onClose();
      }, 800);
    } catch (e) {
      console.error('Config save error:', e);
    } finally {
      setIsSaving(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-900/50 backdrop-blur-xs animate-in fade-in duration-150">
      <div
        className="relative w-full max-w-lg bg-white border border-slate-200 rounded-2xl p-6 space-y-5 shadow-2xl overflow-hidden"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between pb-3 border-b border-slate-200">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-lg bg-amber-50 text-amber-800 border border-amber-200 flex items-center justify-center font-bold text-xs">
              ⚡
            </div>
            <div>
              <h3 className="font-display font-bold text-base text-slate-900">
                DeepScript Model Architecture & Settings
              </h3>
              <p className="text-xs text-slate-500">
                Primary: ViT-B/16 • Optional Fallback: Hugging Face VLM
              </p>
            </div>
          </div>

          <button
            onClick={onClose}
            className="p-1.5 rounded-lg text-slate-400 hover:text-slate-800 hover:bg-slate-100 transition-colors"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Primary Model Highlight Card */}
        <div className="p-3 rounded-xl bg-emerald-50/70 border border-emerald-300 ring-1 ring-emerald-500/20 shadow-2xs">
          <div className="flex items-center justify-between">
            <span className="font-bold text-xs text-emerald-950 flex items-center gap-1.5">
              <Cpu className="w-3.5 h-3.5 text-emerald-700" />
              DeepScript ViT-B/16 (Primary Model)
            </span>
            <span className="text-[10px] font-mono px-2 py-0.2 rounded font-semibold bg-emerald-200/80 text-emerald-900">
              Active Primary
            </span>
          </div>
          <p className="text-[11px] font-medium text-emerald-900 mt-0.5">
            Fine-Tuned Vision Transformer + Cosine Similarity Head
          </p>
          <p className="text-[11px] text-emerald-800/80 mt-0.5 leading-relaxed">
            Trained on whole inscriptions for 5 script classes: Brahmi, Grantha, Gupta, Kadamba, Kharosthi.
          </p>
        </div>

        {/* Model Selector */}
        <div className="space-y-2">
          <label className="text-xs font-bold uppercase tracking-wider text-slate-700 flex items-center gap-1.5">
            <Sparkles className="w-3.5 h-3.5 text-amber-600" />
            <span>Hugging Face VLM — Optional Fallback Engine</span>
          </label>

          <div className="space-y-2">
            {modelsList.map((m) => {
              const isSelected = selectedModel === m.id;
              return (
                <div
                  key={m.id}
                  onClick={() => setSelectedModel(m.id)}
                  className={`p-3 rounded-xl border cursor-pointer transition-all ${
                    isSelected
                      ? 'bg-amber-50/60 border-amber-500 ring-1 ring-amber-500/20 shadow-2xs'
                      : 'bg-slate-50/70 border-slate-200 hover:bg-white hover:border-slate-300'
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <span className="font-bold text-xs text-slate-900">{m.name}</span>
                    <span className={`text-[10px] font-mono px-2 py-0.2 rounded font-semibold ${
                      isSelected ? 'bg-amber-200/80 text-amber-900' : 'bg-slate-200 text-slate-600'
                    }`}>
                      {m.badge}
                    </span>
                  </div>
                  <p className="text-[11px] font-medium text-amber-800 mt-0.5">{m.tag}</p>
                  <p className="text-[11px] text-slate-500 mt-0.5 leading-relaxed">{m.desc}</p>
                </div>
              );
            })}
          </div>
        </div>

        {/* Optional HF Token */}
        <div className="space-y-1.5 pt-1">
          <div className="flex items-center justify-between">
            <label className="text-xs font-bold uppercase tracking-wider text-slate-700 flex items-center gap-1.5">
              <Key className="w-3.5 h-3.5 text-amber-600" />
              <span>Hugging Face Access Token (Optional)</span>
            </label>
            <a
              href="https://huggingface.co/settings/tokens"
              target="_blank"
              rel="noreferrer"
              className="text-[11px] font-medium text-amber-800 hover:underline flex items-center gap-0.5"
            >
              <span>Get Free Token</span>
              <ExternalLink className="w-2.5 h-2.5" />
            </a>
          </div>

          <input
            type="password"
            placeholder="hf_xxxxxxxxxxxxxxxxxxxx (Optional for higher rate limits)"
            value={token}
            onChange={(e) => setToken(e.target.value)}
            className="w-full px-3 py-2 text-xs rounded-xl bg-slate-50 border border-slate-200 focus:outline-hidden focus:ring-2 focus:ring-amber-500/20 focus:border-amber-500 font-mono"
          />
          <p className="text-[11px] text-slate-400">
            A free Hugging Face account token allows unlimited requests at zero cost. Leave blank for community tier.
          </p>
        </div>

        {/* Action Controls */}
        <div className="pt-3 border-t border-slate-200 flex justify-end gap-2">
          <button
            onClick={onClose}
            className="btn-clean-secondary text-xs px-4 py-2"
          >
            Cancel
          </button>
          <button
            onClick={handleSave}
            disabled={isSaving}
            className="btn-clean-primary text-xs px-5 py-2 flex items-center gap-1.5"
          >
            {saveSuccess ? (
              <>
                <Check className="w-3.5 h-3.5 text-white" />
                <span>Saved!</span>
              </>
            ) : (
              <>
                <Sparkles className="w-3.5 h-3.5" />
                <span>Save Settings</span>
              </>
            )}
          </button>
        </div>

      </div>
    </div>
  );
}

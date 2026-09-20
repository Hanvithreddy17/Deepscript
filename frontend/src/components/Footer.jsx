import React from 'react';
import { ShieldCheck, BookOpen, Layers, Sparkles } from 'lucide-react';

export function Footer({ onOpenInfoModal, onOpenLibrary }) {
  return (
    <footer className="mt-16 border-t border-slate-200/80 bg-white/60 py-8 px-4 text-xs text-slate-500">
      <div className="max-w-7xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-4">
        
        {/* Left: Branding & Model info */}
        <div className="flex items-center gap-3">
          <div className="w-6 h-6 rounded-md bg-amber-500 text-white font-bold text-xs flex items-center justify-center">
            𑀅
          </div>
          <div>
            <p className="font-semibold text-slate-700">
              DeepScript • Ancient Indian Script Identification System
            </p>
            <p className="text-[11px] text-slate-400">
              Fine-Tuned Vision Transformer (ViT-B/16) + Cosine Similarity Metric Head
            </p>
          </div>
        </div>

        {/* Right: Quick Links */}
        <div className="flex items-center gap-4 text-xs">
          {onOpenLibrary && (
            <button
              onClick={onOpenLibrary}
              className="text-slate-600 hover:text-amber-800 transition-colors flex items-center gap-1 font-medium"
            >
              <BookOpen className="w-3.5 h-3.5" />
              <span>Script Catalogue</span>
            </button>
          )}

          <button
            onClick={onOpenInfoModal}
            className="text-slate-600 hover:text-amber-800 transition-colors flex items-center gap-1 font-medium"
          >
            <ShieldCheck className="w-3.5 h-3.5" />
            <span>Epigraphic Scope & Guidelines</span>
          </button>
        </div>

      </div>
    </footer>
  );
}

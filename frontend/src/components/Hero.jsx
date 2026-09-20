import React from 'react';
import { Layers, Zap, Cpu, Sparkles, Compass } from 'lucide-react';

export function Hero() {
  return (
    <section className="pt-6 pb-2">
      <div className="flex flex-col md:flex-row md:items-end justify-between gap-4 pb-4 border-b border-slate-200/80">
        
        {/* Left: Title & Description */}
        <div className="max-w-2xl">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-amber-50 border border-amber-200/80 text-amber-800 text-xs font-semibold mb-2.5 shadow-2xs">
            <Sparkles className="w-3.5 h-3.5 text-amber-600" />
            <span>Free Hugging Face VLM Engine</span>
            <span className="text-amber-400">•</span>
            <span className="text-amber-700 font-medium">Qwen 2.5 VL & Indic Epigraphy</span>
          </div>

          <h1 className="font-display text-2xl sm:text-3xl lg:text-4xl font-extrabold tracking-tight text-slate-900 leading-tight">
            Ancient Indian Script Identification
          </h1>

          <p className="mt-2 text-sm text-slate-600 leading-relaxed">
            Classify and analyze historical Indian script families from stone edicts, cave brows, and copper charters using open-source Vision-Language Models with zero API cost.
          </p>
        </div>

        {/* Right: Quick Spec Strip / Benchmark Pills */}
        <div className="flex flex-wrap md:flex-col items-start md:items-end gap-2 shrink-0">
          <div className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-white border border-slate-200 text-xs font-medium text-slate-700 shadow-2xs">
            <Cpu className="w-3.5 h-3.5 text-amber-600" />
            <span>Hugging Face Open VLM</span>
          </div>
          <div className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-white border border-slate-200 text-xs font-medium text-slate-700 shadow-2xs">
            <Layers className="w-3.5 h-3.5 text-indigo-600" />
            <span>Multi-Script Epigraphy</span>
          </div>
          <div className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-emerald-50 border border-emerald-200/80 text-xs font-semibold text-emerald-800 shadow-2xs">
            <Zap className="w-3.5 h-3.5 text-emerald-600" />
            <span>100% Free & Open-Source</span>
          </div>
        </div>

      </div>
    </section>
  );
}

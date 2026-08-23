import React from 'react';
import { ShieldAlert, Sparkles } from 'lucide-react';

export function Hero() {
  return (
    <section className="text-center pt-8 pb-4 max-w-2xl mx-auto px-4">
      
      {/* Scope Disclaimer Pill */}
      <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-zinc-900 border border-white/10 text-zinc-400 text-xs font-medium mb-4">
        <span className="w-1.5 h-1.5 rounded-full bg-amber-400" />
        <span>Script Recognition only • Exclusively ancient Indian scripts</span>
      </div>

      {/* Main Title */}
      <h1 className="font-display text-2xl sm:text-4xl font-bold tracking-tight text-zinc-100 mb-3">
        Ancient Indian Script Identification
      </h1>

      {/* Subtitle */}
      <p className="text-sm text-zinc-400 leading-relaxed max-w-lg mx-auto">
        Classify historical Indian scripts from stone edicts, cave carvings, and copper charters using Vision Transformers and Few-Shot Learning.
      </p>

    </section>
  );
}

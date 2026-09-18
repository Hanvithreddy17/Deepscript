import React, { useState } from 'react';
import { ChevronRight, ChevronDown, Clock, MapPin, Sparkles, AlertCircle, BookOpen } from 'lucide-react';

export function PredictionResult({
  result,
  onOpenDetails
}) {
  const [showTechnicalDetails, setShowTechnicalDetails] = useState(false);

  if (!result) return null;

  const { script, confidence, candidates = [], source, executionTimeMs, details } = result;
  const scriptFamily = details?.scriptFamily || 'Ashokan Brahmi Script';
  const characterName = details?.name || script;

  return (
    <div className="w-full clean-card p-5 sm:p-6 space-y-5">
      
      {/* Primary Result Banner: Identified Script */}
      <div className="pb-5 border-b border-white/[0.06]">
        <div className="flex flex-wrap items-center justify-between gap-3 mb-2">
          <span className="text-xs uppercase tracking-wider font-semibold text-amber-400 bg-amber-500/10 px-2.5 py-1 rounded-md border border-amber-500/20 inline-flex items-center gap-1.5">
            <Sparkles className="w-3.5 h-3.5 text-amber-400" />
            Identified Ancient Script
          </span>

          <span className="text-[11px] font-mono text-zinc-500">
            {source === 'live' ? `ViT Engine (${executionTimeMs}ms)` : 'Reference Model'}
          </span>
        </div>

        {/* Large Prominent Script Headline */}
        <h2 className="font-display text-2xl sm:text-3xl font-bold text-zinc-100 tracking-tight">
          {scriptFamily}
        </h2>

        {/* Glyph and Epigraphic Metadata Badges */}
        <div className="flex flex-wrap items-center gap-2 mt-3">
          <span className="px-2.5 py-1 rounded-md bg-amber-500/10 border border-amber-500/25 text-xs font-semibold text-amber-300">
            Glyph: {characterName}
          </span>
          {details?.category && (
            <span className="px-2.5 py-1 rounded-md bg-zinc-900 border border-white/5 text-xs text-zinc-300">
              {details.category}
            </span>
          )}
          {details?.period && (
            <span className="px-2.5 py-1 rounded-md bg-zinc-900 border border-white/5 text-xs text-zinc-400 flex items-center gap-1.5">
              <Clock className="w-3.5 h-3.5 text-zinc-500" />
              {details.period}
            </span>
          )}
          {details?.region && (
            <span className="px-2.5 py-1 rounded-md bg-zinc-900 border border-white/5 text-xs text-zinc-400 flex items-center gap-1.5">
              <MapPin className="w-3.5 h-3.5 text-zinc-500" />
              {details.region}
            </span>
          )}
        </div>
      </div>

      {/* Epigraphic Dossier Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        
        {/* Left: Paleographic Features */}
        <div className="rounded-xl bg-zinc-950/60 border border-white/[0.06] p-4 space-y-2.5">
          <h4 className="text-xs font-semibold text-zinc-300 uppercase tracking-wider flex items-center gap-1.5">
            <BookOpen className="w-3.5 h-3.5 text-amber-400" />
            Paleographic Characteristics
          </h4>
          <p className="text-xs text-zinc-400 leading-relaxed">
            {details?.visualClues || 'Visual feature embeddings matched against reference historical Indian script classes.'}
          </p>
          {details?.phonetic && (
            <div className="text-xs text-zinc-400 pt-1 border-t border-white/5">
              <span className="text-zinc-500">Phonetic Value:</span> <span className="font-mono text-zinc-300">{details.phonetic}</span>
            </div>
          )}
        </div>

        {/* Right: Historical Inscription Records */}
        <div className="rounded-xl bg-zinc-950/60 border border-white/[0.06] p-4 flex flex-col justify-between">
          <div>
            <h4 className="text-xs font-semibold text-zinc-300 uppercase tracking-wider mb-2">
              Historical Context
            </h4>
            <p className="text-xs text-zinc-400 leading-relaxed mb-3">
              {details?.historicalContext || 'Ancient Indian epigraphic specimen identified via DeepScript Vision Transformer.'}
            </p>
          </div>

          <button
            onClick={() => onOpenDetails(details || { name: script })}
            className="w-full inline-flex items-center justify-between py-2 px-3 rounded-lg text-xs font-medium text-zinc-200 bg-zinc-900 hover:bg-zinc-800 border border-white/5 transition-colors"
          >
            <span>Explore Script Dossier</span>
            <ChevronRight className="w-3.5 h-3.5 text-zinc-500" />
          </button>
        </div>

      </div>

      {/* Optional Collapsible: Raw Model Probabilities */}
      <div className="pt-1">
        <button
          onClick={() => setShowTechnicalDetails(!showTechnicalDetails)}
          className="text-xs text-zinc-500 hover:text-zinc-300 inline-flex items-center gap-1.5 transition-colors"
        >
          {showTechnicalDetails ? <ChevronDown className="w-3.5 h-3.5" /> : <ChevronRight className="w-3.5 h-3.5" />}
          <span>{showTechnicalDetails ? 'Hide' : 'View'} Raw Candidate Probabilities</span>
        </button>

        {showTechnicalDetails && (
          <div className="mt-3 p-3.5 rounded-xl bg-zinc-950/80 border border-white/[0.06] space-y-2">
            {candidates.map((cand, idx) => {
              const candScore = Math.round(cand.score * 100);
              return (
                <div key={cand.script || idx} className="space-y-1">
                  <div className="flex justify-between text-xs">
                    <span className={idx === 0 ? 'text-zinc-200 font-medium' : 'text-zinc-400'}>
                      {cand.script}
                    </span>
                    <span className="font-mono text-zinc-400">
                      {candScore}%
                    </span>
                  </div>
                  <div className="w-full h-1 rounded-full bg-zinc-900 overflow-hidden">
                    <div
                      className={`h-full rounded-full transition-all duration-300 ${
                        idx === 0 ? 'bg-amber-400' : 'bg-zinc-700'
                      }`}
                      style={{ width: `${candScore}%` }}
                    />
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* Scope Disclaimer */}
      <div className="p-3 rounded-lg bg-zinc-950/40 border border-white/[0.04] flex items-center gap-2 text-xs text-zinc-500">
        <AlertCircle className="w-3.5 h-3.5 text-zinc-600 shrink-0" />
        <span>DeepScript identifies ancient Indian script classes. It is not an OCR or linguistic translation system.</span>
      </div>

    </div>
  );
}


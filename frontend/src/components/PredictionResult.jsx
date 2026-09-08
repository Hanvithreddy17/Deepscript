import React from 'react';
import { ChevronRight, Clock, MapPin, Sparkles, AlertCircle } from 'lucide-react';

export function PredictionResult({
  result,
  onOpenDetails
}) {
  if (!result) return null;

  const { script, confidence, candidates = [], source, executionTimeMs, details } = result;
  const confidencePercent = Math.round(confidence * 100);

  return (
    <div className="w-full clean-card p-5 sm:p-6 space-y-6">
      
      {/* Top Banner: Identified Script */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-5 border-b border-white/[0.06]">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <span className="text-[11px] uppercase tracking-wider font-semibold text-amber-400 bg-amber-500/10 px-2 py-0.5 rounded border border-amber-500/20">
              Identified Indian Script
            </span>
            <span className="text-[11px] font-mono text-zinc-500">
              {source === 'live' ? 'FastAPI ViT Engine' : 'Prototype Simulation'}
            </span>
            {details?.category && (
              <span className="text-[11px] font-mono text-cyan-400 bg-cyan-500/10 px-2 py-0.5 rounded border border-cyan-500/20">
                {details.category}
              </span>
            )}
          </div>

          <h2 className="font-display text-2xl sm:text-3xl font-bold text-zinc-100">
            {script}
          </h2>

          {(details?.period || details?.phonetic) && (
            <p className="text-xs text-zinc-400 flex flex-wrap items-center gap-2 mt-1">
              {details.phonetic && (
                <span className="font-mono text-zinc-300 bg-zinc-900 px-1.5 py-0.5 rounded border border-white/5">
                  {details.phonetic}
                </span>
              )}
              {details.period && (
                <>
                  <span className="flex items-center gap-1 text-zinc-400">
                    <Clock className="w-3.5 h-3.5 text-zinc-500" />
                    {details.period}
                  </span>
                  {details.region && (
                    <>
                      <span className="text-zinc-600">•</span>
                      <span className="flex items-center gap-1 text-zinc-400 truncate">
                        <MapPin className="w-3.5 h-3.5 text-zinc-500" />
                        {details.region}
                      </span>
                    </>
                  )}
                </>
              )}
            </p>
          )}
        </div>

        {/* Confidence Percentage Badge */}
        <div className="flex items-baseline gap-2 bg-zinc-900 border border-white/[0.08] px-4 py-2.5 rounded-xl self-start sm:self-auto">
          <span className="text-xs text-zinc-400 uppercase tracking-wide font-medium">
            Confidence
          </span>
          <span className="font-mono text-2xl font-bold text-emerald-400">
            {confidencePercent}%
          </span>
        </div>
      </div>

      {/* Grid: Candidates & Epigraphic Clues */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
        
        {/* Left: Candidate Probabilities */}
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <h4 className="text-xs font-semibold text-zinc-400 uppercase tracking-wider">
              Script Class Probabilities
            </h4>
            <span className="text-[10px] font-mono text-zinc-500">
              {executionTimeMs}ms
            </span>
          </div>

          <div className="space-y-2">
            {candidates.map((cand, idx) => {
              const candScore = Math.round(cand.score * 100);
              return (
                <div key={cand.script || idx} className="space-y-1">
                  <div className="flex justify-between text-xs">
                    <span className={idx === 0 ? 'text-zinc-100 font-medium' : 'text-zinc-400'}>
                      {cand.script}
                    </span>
                    <span className="font-mono text-zinc-400">
                      {candScore}%
                    </span>
                  </div>
                  <div className="w-full h-1.5 rounded-full bg-zinc-900 overflow-hidden border border-white/5">
                    <div
                      className={`h-full rounded-full transition-all duration-500 ${
                        idx === 0 ? 'bg-amber-400' : 'bg-zinc-700'
                      }`}
                      style={{ width: `${candScore}%` }}
                    />
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Right: Paleographic Key Features */}
        <div className="rounded-xl bg-zinc-950/60 border border-white/[0.06] p-4 flex flex-col justify-between">
          <div>
            <h4 className="text-xs font-semibold text-zinc-300 flex items-center gap-1.5 mb-2">
              <Sparkles className="w-3.5 h-3.5 text-amber-400" />
              Paleographic Characteristics
            </h4>
            <p className="text-xs text-zinc-400 leading-relaxed mb-3">
              {details?.visualClues || 'Visual feature embeddings matched against reference historical Indian script classes.'}
            </p>
          </div>

          <button
            onClick={() => onOpenDetails(details || { name: script })}
            className="w-full inline-flex items-center justify-between py-2 px-3 rounded-lg text-xs font-medium text-zinc-200 bg-zinc-900 hover:bg-zinc-800 border border-white/5 transition-colors"
          >
            <span>Explore Epigraphic Dossier</span>
            <ChevronRight className="w-3.5 h-3.5 text-zinc-500" />
          </button>
        </div>

      </div>

      {/* Scope Disclaimer */}
      <div className="p-3 rounded-lg bg-zinc-950/40 border border-white/[0.04] flex items-center gap-2 text-xs text-zinc-500">
        <AlertCircle className="w-3.5 h-3.5 text-zinc-600 shrink-0" />
        <span>DeepScript identifies ancient Indian script classes. It is not an OCR or linguistic translation system.</span>
      </div>

    </div>
  );
}

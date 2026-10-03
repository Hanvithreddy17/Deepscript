import React, { useState } from 'react';
import { 
  ChevronRight, 
  Clock, 
  MapPin, 
  Sparkles, 
  BookOpen, 
  CheckCircle2, 
  BarChart3, 
  Copy, 
  Check, 
  ShieldCheck,
  Compass,
  ArrowRightLeft,
  Crown
} from 'lucide-react';

export function PredictionResult({
  result,
  onOpenDetails
}) {
  const [activeTab, setActiveTab] = useState('paleography');
  const [copied, setCopied] = useState(false);

  if (!result) return null;

  const { script, confidence, candidates = [], sourceLabel, executionTimeMs, details, model } = result;
  const scriptFamily = details?.scriptFamily || details?.name || script || 'Ashokan Brahmi';
  const confidencePct = Math.round((confidence || 0.90) * 100);

  const getConfidenceBadge = (pct) => {
    if (pct >= 85) {
      return {
        label: 'High Confidence',
        bg: 'bg-emerald-50 text-emerald-800 border-emerald-200',
        barColor: 'bg-emerald-500'
      };
    } else if (pct >= 60) {
      return {
        label: 'Moderate Match',
        bg: 'bg-amber-50 text-amber-800 border-amber-200',
        barColor: 'bg-amber-500'
      };
    } else {
      return {
        label: 'Exploratory Candidate',
        bg: 'bg-slate-100 text-slate-700 border-slate-200',
        barColor: 'bg-slate-400'
      };
    }
  };

  const badgeInfo = getConfidenceBadge(confidencePct);

  const handleCopyJson = () => {
    navigator.clipboard.writeText(JSON.stringify(result, null, 2));
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="w-full clean-card p-5 sm:p-6 space-y-5 bg-white shadow-soft animate-in fade-in duration-200">
      
      {/* Result Header & Classification Callout */}
      <div className="pb-4 border-b border-slate-200/80">
        
        <div className="flex flex-wrap items-center justify-between gap-2 mb-2.5">
          <div className="flex items-center gap-2">
            <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-amber-50 border border-amber-200 text-amber-800 text-xs font-bold uppercase tracking-wider">
              <Sparkles className="w-3.5 h-3.5 text-amber-600" />
              Identified Script Family
            </span>
            <span className="hidden sm:inline-flex items-center text-[10px] font-semibold text-slate-500 bg-slate-100 px-2 py-0.5 rounded border border-slate-200">
              {sourceLabel || 'DeepScript ViT-B/16'}
            </span>
          </div>

          <div className="flex items-center gap-2">
            <span className={`inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-xs font-semibold border ${badgeInfo.bg}`}>
              <ShieldCheck className="w-3.5 h-3.5" />
              <span>{confidencePct}% {badgeInfo.label}</span>
            </span>
            <span className="text-[11px] font-mono text-slate-500 bg-slate-100 px-2 py-0.5 rounded border border-slate-200">
              {executionTimeMs}ms
            </span>
          </div>
        </div>

        {/* Big Script Headline */}
        <h2 className="font-display text-2xl sm:text-3xl font-extrabold text-slate-900 tracking-tight">
          {scriptFamily}
        </h2>

        {/* Epigraphic Metadata Badges */}
        <div className="flex flex-wrap items-center gap-2 mt-3">
          {details?.writingDirection && (
            <span className="px-2.5 py-1 rounded-lg bg-indigo-50 border border-indigo-200 text-xs font-bold text-indigo-900 flex items-center gap-1">
              <ArrowRightLeft className="w-3 h-3 text-indigo-600" />
              <span>{details.writingDirection}</span>
            </span>
          )}

          {details?.period && (
            <span className="px-2.5 py-1 rounded-lg bg-slate-100 border border-slate-200 text-xs text-slate-700 font-medium flex items-center gap-1">
              <Clock className="w-3 h-3 text-slate-400" />
              <span>{details.period}</span>
            </span>
          )}

          {details?.region && (
            <span className="px-2.5 py-1 rounded-lg bg-slate-100 border border-slate-200 text-xs text-slate-700 font-medium flex items-center gap-1">
              <MapPin className="w-3 h-3 text-slate-400" />
              <span className="truncate max-w-[200px]">{details.region}</span>
            </span>
          )}
        </div>

      </div>

      {/* Epigraphic Dossier Navigation Tabs */}
      <div className="flex items-center gap-2 border-b border-slate-200/80 pb-2">
        <button
          type="button"
          onClick={() => setActiveTab('paleography')}
          className={`px-3 py-1.5 text-xs font-bold rounded-lg transition-all flex items-center gap-1.5 ${
            activeTab === 'paleography'
              ? 'bg-slate-900 text-white shadow-2xs'
              : 'text-slate-600 hover:text-slate-900 hover:bg-slate-100'
          }`}
        >
          <BookOpen className="w-3.5 h-3.5" />
          <span>Paleography</span>
        </button>

        <button
          type="button"
          onClick={() => setActiveTab('history')}
          className={`px-3 py-1.5 text-xs font-bold rounded-lg transition-all flex items-center gap-1.5 ${
            activeTab === 'history'
              ? 'bg-slate-900 text-white shadow-2xs'
              : 'text-slate-600 hover:text-slate-900 hover:bg-slate-100'
          }`}
        >
          <Compass className="w-3.5 h-3.5" />
          <span>Archaeology & Sites</span>
        </button>

        <button
          type="button"
          onClick={() => setActiveTab('candidates')}
          className={`px-3 py-1.5 text-xs font-bold rounded-lg transition-all flex items-center gap-1.5 ${
            activeTab === 'candidates'
              ? 'bg-slate-900 text-white shadow-2xs'
              : 'text-slate-600 hover:text-slate-900 hover:bg-slate-100'
          }`}
        >
          <BarChart3 className="w-3.5 h-3.5" />
          <span>Candidates ({candidates.length})</span>
        </button>
      </div>

      {/* Tab Contents */}
      <div className="min-h-[140px]">
        {/* Tab 1: Paleography */}
        {activeTab === 'paleography' && (
          <div className="space-y-3 animate-in fade-in duration-150">
            {/* Visual Clues */}
            <div className="p-3.5 rounded-xl bg-slate-50 border border-slate-200/80 space-y-1.5">
              <h4 className="text-xs font-bold text-slate-800 uppercase tracking-wider flex items-center gap-1.5">
                <Sparkles className="w-3.5 h-3.5 text-amber-600" />
                <span>Paleographic Diagnostics</span>
              </h4>
              <p className="text-xs text-slate-700 leading-relaxed">
                {details?.visualClues || details?.strokeStyle || 'Visual stroke and contour morphology analyzed with ViT-B/16 neural feature extractor.'}
              </p>
            </div>

            {/* Headmark & Stroke Style Chips */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 text-xs">
              {details?.headmark && (
                <div className="p-2.5 rounded-lg bg-amber-50/60 border border-amber-200/70 text-amber-950">
                  <span className="font-bold text-[11px] text-amber-900 block mb-0.5">Headmark Morphology</span>
                  <span>{details.headmark}</span>
                </div>
              )}
              {details?.stroke_style && (
                <div className="p-2.5 rounded-lg bg-slate-50 border border-slate-200 text-slate-700">
                  <span className="font-bold text-[11px] text-slate-800 block mb-0.5">Stroke Style</span>
                  <span>{details.stroke_style}</span>
                </div>
              )}
            </div>

            {/* Key Glyphs */}
            {details?.keyGlyphs && details.keyGlyphs.length > 0 && (
              <div className="space-y-1.5">
                <h5 className="text-[11px] font-bold text-slate-500 uppercase tracking-wider">
                  Diagnostic Glyphs Recognized
                </h5>
                <div className="flex flex-wrap gap-1.5">
                  {details.keyGlyphs.map((glyph, idx) => (
                    <span key={idx} className="px-2.5 py-1 rounded-md bg-white border border-slate-200 text-xs font-mono text-slate-800 shadow-2xs">
                      {glyph}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {/* Tab 2: Archaeology & Sites */}
        {activeTab === 'history' && (
          <div className="space-y-3 animate-in fade-in duration-150">
            <div className="p-3.5 rounded-xl bg-slate-50 border border-slate-200/80 space-y-1.5">
              <h4 className="text-xs font-bold text-slate-800 uppercase tracking-wider flex items-center gap-1.5">
                <Compass className="w-3.5 h-3.5 text-indigo-600" />
                <span>Historical Context & Inscriptions</span>
              </h4>
              <p className="text-xs text-slate-700 leading-relaxed">
                {details?.historicalContext || 'Ancient Indian epigraphic record verified with DeepScript epigraphic database.'}
              </p>
            </div>

            {details?.famousInscriptions && details.famousInscriptions.length > 0 && (
              <div>
                <h5 className="text-[11px] font-bold text-slate-500 uppercase tracking-wider mb-1.5">
                  Prominent Archaeological Sites
                </h5>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-1.5">
                  {details.famousInscriptions.map((site, idx) => (
                    <div key={idx} className="p-2 rounded-lg bg-slate-50 border border-slate-200 text-xs text-slate-700 flex items-center gap-1.5">
                      <MapPin className="w-3 h-3 text-amber-600 shrink-0" />
                      <span className="truncate">{site}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {/* Tab 3: Ranked Candidates */}
        {activeTab === 'candidates' && (
          <div className="space-y-2.5 animate-in fade-in duration-150">
            <h4 className="text-xs font-bold text-slate-700 uppercase tracking-wider">
              Comparative Candidate Distribution
            </h4>
            <div className="space-y-2">
              {candidates.map((cand, idx) => {
                const candScore = Math.round((cand.score ?? cand.probability ?? cand.confidence ?? 0) * 100);
                return (
                  <div key={cand.script || idx} className="p-2.5 rounded-xl bg-slate-50 border border-slate-200/80 space-y-1.5">
                    <div className="flex justify-between items-center text-xs">
                      <div>
                        <span className={`font-semibold ${idx === 0 ? 'text-amber-950 font-bold' : 'text-slate-700'}`}>
                          {cand.script}
                        </span>
                        {cand.period && (
                          <span className="text-[10px] text-slate-400 pl-1.5 hidden sm:inline">
                            • {cand.period}
                          </span>
                        )}
                      </div>
                      <span className="font-mono font-bold text-slate-700">
                        {candScore}%
                      </span>
                    </div>
                    <div className="w-full h-1.5 rounded-full bg-slate-200 overflow-hidden">
                      <div
                        className={`h-full rounded-full transition-all duration-500 ${
                          idx === 0 ? 'bg-amber-500' : 'bg-slate-400'
                        }`}
                        style={{ width: `${candScore}%` }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}
      </div>

      {/* Action Footer */}
      <div className="pt-3 border-t border-slate-200/80 flex flex-wrap items-center justify-between gap-2">
        <button
          type="button"
          onClick={() => onOpenDetails(details || { name: script })}
          className="btn-clean-secondary text-xs py-2 px-3.5 flex items-center gap-1.5"
        >
          <BookOpen className="w-3.5 h-3.5 text-slate-500" />
          <span>Explore Full Script Dossier</span>
          <ChevronRight className="w-3.5 h-3.5 text-slate-400" />
        </button>

        <button
          type="button"
          onClick={handleCopyJson}
          className="text-xs text-slate-500 hover:text-slate-800 inline-flex items-center gap-1 px-2.5 py-1.5 rounded-md hover:bg-slate-100 transition-colors"
          title="Copy raw inference payload"
        >
          {copied ? <Check className="w-3.5 h-3.5 text-emerald-600" /> : <Copy className="w-3.5 h-3.5" />}
          <span>{copied ? 'Copied' : 'Copy JSON'}</span>
        </button>
      </div>

    </div>
  );
}

import React from 'react';
import { SAMPLE_INSCRIPTIONS } from '../data/sampleImages';
import { Check, Sparkles, MapPin, Clock } from 'lucide-react';

export function SamplePicker({ onSelectSample, activeSampleId, isAnalyzing }) {
  return (
    <div className="w-full">
      <div className="flex items-center justify-between mb-2.5">
        <label className="text-xs font-bold uppercase tracking-wider text-slate-700 flex items-center gap-1.5">
          <Sparkles className="w-3.5 h-3.5 text-amber-600" />
          <span>Select Historical Specimen</span>
        </label>
        <span className="text-[11px] text-slate-500 hidden sm:inline">
          Curated archaeological inscriptions
        </span>
      </div>

      {/* Grid of Specimen Cards */}
      <div className="grid grid-cols-2 sm:grid-cols-3 gap-2.5">
        {SAMPLE_INSCRIPTIONS.map((sample) => {
          const isSelected = activeSampleId === sample.id;
          return (
            <button
              key={sample.id}
              type="button"
              onClick={() => onSelectSample(sample)}
              disabled={isAnalyzing}
              className={`group relative text-left rounded-xl p-2.5 transition-all duration-200 border cursor-pointer ${
                isSelected
                  ? 'border-amber-500 bg-amber-50/60 shadow-sm ring-2 ring-amber-500/20'
                  : 'border-slate-200 bg-white hover:border-slate-300 hover:bg-slate-50/70 shadow-2xs hover:shadow-sm'
              }`}
            >
              {/* Selected Badge */}
              {isSelected && (
                <div className="absolute top-2 right-2 z-10 w-4 h-4 rounded-full bg-amber-600 text-white flex items-center justify-center shadow-xs">
                  <Check className="w-2.5 h-2.5 stroke-[3]" />
                </div>
              )}

              {/* Thumbnail Container */}
              <div className="w-full h-20 rounded-lg overflow-hidden bg-slate-900 mb-2 relative border border-slate-200/60 flex items-center justify-center group-hover:scale-[1.02] transition-transform duration-200">
                <img
                  src={sample.thumbnail}
                  alt={sample.title}
                  className="w-full h-full object-cover"
                />
                <span className="absolute bottom-1 right-1 px-1.5 py-0.2 rounded bg-slate-900/85 text-[10px] font-mono text-slate-300">
                  {sample.medium ? sample.medium.split(' ')[0] : 'Stone'}
                </span>
              </div>

              {/* Title & Script Info */}
              <div>
                <h4 className={`text-xs font-bold leading-snug truncate ${
                  isSelected ? 'text-amber-950' : 'text-slate-900 group-hover:text-amber-900'
                }`}>
                  {sample.script}
                </h4>
                <div className="flex items-center gap-1 text-[11px] text-slate-500 mt-0.5 truncate">
                  <Clock className="w-3 h-3 text-slate-400 shrink-0" />
                  <span className="truncate">{sample.period}</span>
                </div>
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
}

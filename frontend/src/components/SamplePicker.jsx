import React from 'react';
import { SAMPLE_INSCRIPTIONS } from '../data/sampleImages';

export function SamplePicker({ onSelectSample, activeSampleId, isAnalyzing }) {
  return (
    <div className="w-full">
      <div className="flex items-center justify-between mb-3 px-1">
        <h3 className="text-xs font-semibold text-zinc-400 uppercase tracking-wider">
          Sample Indian Inscriptions
        </h3>
        <span className="text-[11px] text-zinc-500 hidden sm:inline">
          Click any specimen to test
        </span>
      </div>

      {/* Grid of Minimalist Sample Cards */}
      <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-6 gap-2.5">
        {SAMPLE_INSCRIPTIONS.map((sample) => {
          const isSelected = activeSampleId === sample.id;
          return (
            <button
              key={sample.id}
              onClick={() => onSelectSample(sample)}
              disabled={isAnalyzing}
              className={`group text-left rounded-xl p-2 transition-all border ${
                isSelected
                  ? 'border-amber-500/50 bg-zinc-900 shadow-sm ring-1 ring-amber-500/20'
                  : 'border-white/[0.06] bg-zinc-925 hover:border-white/15 hover:bg-zinc-900'
              }`}
            >
              {/* Thumbnail Container */}
              <div className="w-full h-18 sm:h-20 rounded-lg overflow-hidden bg-zinc-950 mb-2 relative border border-white/[0.04]">
                <img
                  src={sample.thumbnail}
                  alt={sample.title}
                  className="w-full h-full object-cover group-hover:scale-102 transition-transform duration-200"
                />
              </div>

              {/* Title & Script Label */}
              <div>
                <h4 className={`text-xs font-medium truncate ${isSelected ? 'text-amber-300 font-semibold' : 'text-zinc-200 group-hover:text-zinc-100'}`}>
                  {sample.script}
                </h4>
                <p className="text-[11px] text-zinc-500 truncate mt-0.5">
                  {sample.period}
                </p>
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
}

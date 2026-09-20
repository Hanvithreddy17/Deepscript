import React, { useState } from 'react';
import { X, BookOpen, Clock, MapPin, Search, ChevronRight, CheckCircle2, Landmark } from 'lucide-react';
import { ANCIENT_SCRIPTS } from '../data/scriptsData';

export function ScriptLibraryModal({ isOpen, onClose, onSelectScript }) {
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedScriptKey, setSelectedScriptKey] = useState(Object.keys(ANCIENT_SCRIPTS)[0] || null);

  if (!isOpen) return null;

  const scriptKeys = Object.keys(ANCIENT_SCRIPTS).filter((key) => {
    const item = ANCIENT_SCRIPTS[key];
    const q = searchQuery.toLowerCase();
    return (
      key.toLowerCase().includes(q) ||
      (item.region && item.region.toLowerCase().includes(q)) ||
      (item.period && item.period.toLowerCase().includes(q))
    );
  });

  const activeScript = ANCIENT_SCRIPTS[selectedScriptKey] || ANCIENT_SCRIPTS[scriptKeys[0]];

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 sm:p-6 bg-slate-900/50 backdrop-blur-xs animate-in fade-in duration-150">
      <div
        className="relative w-full max-w-4xl h-[85vh] bg-white border border-slate-200 rounded-2xl flex flex-col shadow-2xl overflow-hidden"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="p-5 sm:p-6 border-b border-slate-200 flex items-center justify-between bg-slate-50/70">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-amber-100 text-amber-800 border border-amber-200 flex items-center justify-center">
              <BookOpen className="w-5 h-5" />
            </div>
            <div>
              <h2 className="font-display text-lg font-bold text-slate-900">
                Ancient Indian Script Library
              </h2>
              <p className="text-xs text-slate-500">
                Reference paleographic catalogue of historical Indian scripts
              </p>
            </div>
          </div>

          <button
            onClick={onClose}
            className="p-2 rounded-xl text-slate-400 hover:text-slate-800 hover:bg-slate-200/60 transition-colors"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Search Bar */}
        <div className="p-4 border-b border-slate-200 bg-white">
          <div className="relative">
            <Search className="w-4 h-4 text-slate-400 absolute left-3 top-3" />
            <input
              type="text"
              placeholder="Search scripts by name, historical period, or region..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full pl-9 pr-4 py-2 text-xs rounded-xl bg-slate-50 border border-slate-200 focus:outline-hidden focus:ring-2 focus:ring-amber-500/20 focus:border-amber-500 transition-all"
            />
          </div>
        </div>

        {/* 2-Column Split: Script List (Left) & Full Dossier (Right) */}
        <div className="flex-1 grid grid-cols-1 md:grid-cols-12 overflow-hidden">
          
          {/* Script List Sidebar */}
          <div className="md:col-span-4 border-r border-slate-200 overflow-y-auto p-3 space-y-1.5 bg-slate-50/50">
            {scriptKeys.map((key) => {
              const script = ANCIENT_SCRIPTS[key];
              const isSelected = key === selectedScriptKey;
              return (
                <button
                  key={key}
                  onClick={() => setSelectedScriptKey(key)}
                  className={`w-full text-left p-3 rounded-xl transition-all border ${
                    isSelected
                      ? 'bg-white border-amber-400 text-amber-950 shadow-sm ring-1 ring-amber-500/20'
                      : 'bg-white/80 border-slate-200/80 hover:bg-white text-slate-700 hover:border-slate-300'
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <h4 className="text-xs font-bold truncate">{script.name}</h4>
                    <ChevronRight className={`w-3.5 h-3.5 ${isSelected ? 'text-amber-600' : 'text-slate-300'}`} />
                  </div>
                  <p className="text-[11px] text-slate-500 truncate mt-0.5">{script.period}</p>
                </button>
              );
            })}
          </div>

          {/* Active Script Details Panel */}
          <div className="md:col-span-8 overflow-y-auto p-6 space-y-4">
            {activeScript ? (
              <>
                <div>
                  <span className="text-[11px] uppercase tracking-wider font-bold text-amber-700 bg-amber-50 px-2.5 py-1 rounded-md border border-amber-200">
                    Epigraphic Family Profile
                  </span>
                  <h3 className="font-display text-2xl font-extrabold text-slate-900 mt-2">
                    {activeScript.name}
                  </h3>
                  {activeScript.alternateNames && (
                    <p className="text-xs text-slate-500 mt-0.5">
                      Alternate names: <span className="font-medium text-slate-700">{activeScript.alternateNames.join(', ')}</span>
                    </p>
                  )}
                </div>

                <div className="grid grid-cols-2 gap-3">
                  <div className="p-3 rounded-xl bg-slate-50 border border-slate-200">
                    <span className="text-[11px] font-semibold text-slate-500 block mb-0.5">Chronology</span>
                    <p className="text-xs font-bold text-slate-800">{activeScript.period}</p>
                  </div>
                  <div className="p-3 rounded-xl bg-slate-50 border border-slate-200">
                    <span className="text-[11px] font-semibold text-slate-500 block mb-0.5">Primary Geography</span>
                    <p className="text-xs font-bold text-slate-800">{activeScript.region}</p>
                  </div>
                </div>

                {activeScript.historicalContext && (
                  <div className="p-3.5 rounded-xl bg-slate-50 border border-slate-200 text-xs text-slate-600 space-y-1">
                    <span className="font-bold text-slate-800 uppercase tracking-wider text-[11px] block">Historical Overview</span>
                    <p className="leading-relaxed">{activeScript.historicalContext}</p>
                  </div>
                )}

                {activeScript.keyFeatures && (
                  <div>
                    <h5 className="text-[11px] font-bold text-slate-500 uppercase tracking-wider mb-2">
                      Key Paleographic Traits
                    </h5>
                    <ul className="space-y-1.5 text-xs text-slate-700">
                      {activeScript.keyFeatures.map((feat, i) => (
                        <li key={i} className="flex items-start gap-2 bg-slate-50 p-2.5 rounded-lg border border-slate-200/80">
                          <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600 shrink-0 mt-0.5" />
                          <span>{feat}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}

                {activeScript.famousInscriptions && (
                  <div>
                    <h5 className="text-[11px] font-bold text-slate-500 uppercase tracking-wider mb-1.5">
                      Archaeological Sites & Monumental Inscriptions
                    </h5>
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-1.5 text-xs text-slate-700">
                      {activeScript.famousInscriptions.map((site, idx) => (
                        <div key={idx} className="p-2 rounded-lg bg-slate-50 border border-slate-200 flex items-center gap-1.5">
                          <MapPin className="w-3 h-3 text-amber-600 shrink-0" />
                          <span>{site}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </>
            ) : (
              <div className="text-center py-16 text-slate-400 text-xs">
                Select a script from the list to view its complete epigraphic dossier.
              </div>
            )}
          </div>

        </div>

        {/* Footer */}
        <div className="p-4 border-t border-slate-200 bg-slate-50/70 flex justify-end">
          <button
            onClick={onClose}
            className="btn-clean-primary text-xs px-5 py-2"
          >
            Close Library
          </button>
        </div>

      </div>
    </div>
  );
}

import React from 'react';
import { X, BookOpen, Clock, MapPin, CheckCircle2, Landmark, GitCommit, Layers } from 'lucide-react';
import { ANCIENT_SCRIPTS } from '../data/scriptsData';

export function ScriptModal({ scriptData, isOpen, onClose }) {
  if (!isOpen) return null;

  const scriptKey = typeof scriptData === 'string' ? scriptData : scriptData?.name;
  const script = ANCIENT_SCRIPTS[scriptKey] || scriptData || {};

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-900/50 backdrop-blur-xs animate-in fade-in duration-150">
      <div
        className="relative w-full max-w-xl max-h-[88vh] overflow-y-auto bg-white border border-slate-200 rounded-2xl p-6 sm:p-7 space-y-5 shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        
        {/* Close Button */}
        <button
          onClick={onClose}
          className="absolute top-5 right-5 p-2 rounded-xl text-slate-400 hover:text-slate-800 hover:bg-slate-100 transition-colors border border-transparent hover:border-slate-200"
          aria-label="Close modal"
        >
          <X className="w-4 h-4" />
        </button>

        {/* Header */}
        <div>
          <span className="text-[11px] uppercase tracking-wider font-bold text-amber-700 bg-amber-50 px-2.5 py-1 rounded-md border border-amber-200/80">
            Epigraphic Dossier
          </span>
          <h2 className="font-display text-xl sm:text-2xl font-extrabold text-slate-900 mt-2">
            {script.name || 'Ancient Indian Script Profile'}
          </h2>
          {script.alternateNames && (
            <p className="text-xs text-slate-500 mt-0.5">
              Also known as: <span className="font-medium text-slate-700">{script.alternateNames.join(', ')}</span>
            </p>
          )}
        </div>

        {/* Quick Facts Grid */}
        <div className="grid grid-cols-2 gap-3">
          <div className="p-3.5 rounded-xl bg-slate-50 border border-slate-200/80">
            <div className="flex items-center gap-1.5 text-xs font-semibold text-slate-500 mb-1">
              <Clock className="w-3.5 h-3.5 text-amber-600" />
              <span>Historical Period</span>
            </div>
            <p className="text-xs text-slate-800 font-bold">{script.period || 'Ancient Era'}</p>
          </div>

          <div className="p-3.5 rounded-xl bg-slate-50 border border-slate-200/80">
            <div className="flex items-center gap-1.5 text-xs font-semibold text-slate-500 mb-1">
              <MapPin className="w-3.5 h-3.5 text-indigo-600" />
              <span>Geographical Region</span>
            </div>
            <p className="text-xs text-slate-800 font-bold">{script.region || 'Pan-Indian'}</p>
          </div>
        </div>

        {/* Phonetic & Classification Details */}
        {(script.category || script.phonetic || script.scriptFamily) && (
          <div className="p-3.5 rounded-xl bg-amber-50/50 border border-amber-200/70 text-xs space-y-1.5">
            {script.category && (
              <p>
                <strong className="text-slate-700">Epigraphic Category:</strong>{' '}
                <span className="text-amber-900 font-semibold">{script.category}</span>
              </p>
            )}
            {script.phonetic && (
              <p>
                <strong className="text-slate-700">Phonetic Value:</strong>{' '}
                <span className="font-mono font-bold text-amber-800 bg-white px-1.5 py-0.5 rounded border border-amber-200">{script.phonetic}</span>
              </p>
            )}
            {script.scriptFamily && (
              <p>
                <strong className="text-slate-700">Script Family:</strong>{' '}
                <span className="text-slate-900 font-semibold">{script.scriptFamily}</span>
              </p>
            )}
          </div>
        )}

        {/* Visual Recognition Clues */}
        {script.visualClues && (
          <div>
            <h4 className="text-xs font-bold text-slate-700 uppercase tracking-wider mb-1.5">
              Visual Recognition Characteristics
            </h4>
            <p className="text-xs text-slate-700 bg-slate-50 p-3 rounded-xl border border-slate-200/80 leading-relaxed">
              {script.visualClues}
            </p>
          </div>
        )}

        {/* Historical Context */}
        {script.historicalContext && (
          <div>
            <h4 className="text-xs font-bold text-slate-700 uppercase tracking-wider mb-1.5">
              Historical Context & Royal Patronage
            </h4>
            <p className="text-xs sm:text-sm text-slate-600 leading-relaxed bg-slate-50 p-3.5 rounded-xl border border-slate-200/80">
              {script.historicalContext}
            </p>
          </div>
        )}

        {/* Key Paleographic Characteristics */}
        {script.keyFeatures && (
          <div>
            <h4 className="text-xs font-bold text-slate-700 uppercase tracking-wider mb-1.5">
              Key Paleographic Characteristics
            </h4>
            <ul className="space-y-1.5 text-xs text-slate-700">
              {script.keyFeatures.map((feat, idx) => (
                <li key={idx} className="flex items-start gap-2 bg-slate-50 p-2.5 rounded-lg border border-slate-200/80">
                  <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600 shrink-0 mt-0.5" />
                  <span>{feat}</span>
                </li>
              ))}
            </ul>
          </div>
        )}

        {/* Exemplar Sites */}
        {script.famousInscriptions && (
          <div>
            <h4 className="text-xs font-bold text-slate-700 uppercase tracking-wider mb-1.5">
              Archaeological Inscriptions & Sites
            </h4>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
              {script.famousInscriptions.map((site, idx) => (
                <div key={idx} className="p-2.5 rounded-lg bg-slate-50 border border-slate-200 text-xs text-slate-700 flex items-center gap-1.5">
                  <MapPin className="w-3.5 h-3.5 text-amber-600 shrink-0" />
                  <span className="truncate">{site}</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Lineage */}
        {(script.ancestor || script.descendants) && (
          <div className="p-3.5 rounded-xl bg-slate-50 border border-slate-200 text-xs text-slate-600 space-y-1">
            {script.ancestor && (
              <p>
                <strong className="text-slate-800">Ancestor:</strong> {script.ancestor}
              </p>
            )}
            {script.descendants && (
              <p>
                <strong className="text-slate-800">Descendants:</strong> {script.descendants.join(', ')}
              </p>
            )}
          </div>
        )}

        {/* Footer */}
        <div className="pt-2 flex justify-end">
          <button onClick={onClose} className="btn-clean-primary text-xs px-5 py-2">
            Close Dossier
          </button>
        </div>

      </div>
    </div>
  );
}

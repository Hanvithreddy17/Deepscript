import React from 'react';
import { X, BookOpen, Clock, MapPin, CheckCircle2, Landmark, GitCommit } from 'lucide-react';
import { ANCIENT_SCRIPTS } from '../data/scriptsData';

export function ScriptModal({ scriptData, isOpen, onClose }) {
  if (!isOpen) return null;

  const scriptKey = typeof scriptData === 'string' ? scriptData : scriptData?.name;
  const script = ANCIENT_SCRIPTS[scriptKey] || scriptData || {};

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/70 backdrop-blur-sm animate-in fade-in duration-150">
      <div
        className="relative w-full max-w-xl max-h-[85vh] overflow-y-auto bg-zinc-925 border border-white/10 rounded-2xl p-6 space-y-5 shadow-xl"
        onClick={(e) => e.stopPropagation()}
      >
        
        {/* Close Button */}
        <button
          onClick={onClose}
          className="absolute top-5 right-5 p-1.5 rounded-lg text-zinc-400 hover:text-zinc-100 hover:bg-zinc-800 transition-colors"
        >
          <X className="w-4 h-4" />
        </button>

        {/* Header */}
        <div>
          <span className="text-[11px] uppercase tracking-wider font-semibold text-amber-400">
            Epigraphic Dossier
          </span>
          <h2 className="font-display text-xl sm:text-2xl font-bold text-zinc-100 mt-1">
            {script.name || 'Indian Script Profile'}
          </h2>
          {script.alternateNames && (
            <p className="text-xs text-zinc-500 mt-0.5">
              Also known as: {script.alternateNames.join(', ')}
            </p>
          )}
        </div>

        {/* Quick Facts */}
        <div className="grid grid-cols-2 gap-3">
          <div className="p-3 rounded-xl bg-zinc-900/60 border border-white/5">
            <div className="flex items-center gap-1.5 text-xs text-zinc-400 mb-1">
              <Clock className="w-3.5 h-3.5 text-amber-400/80" />
              <span>Historical Period</span>
            </div>
            <p className="text-xs text-zinc-200 font-medium">{script.period || 'Ancient Era'}</p>
          </div>

          <div className="p-3 rounded-xl bg-zinc-900/60 border border-white/5">
            <div className="flex items-center gap-1.5 text-xs text-zinc-400 mb-1">
              <MapPin className="w-3.5 h-3.5 text-cyan-400/80" />
              <span>Geographical Region</span>
            </div>
            <p className="text-xs text-zinc-200 font-medium">{script.region || 'India'}</p>
          </div>
        </div>

        {/* Phonetic & Classification Details (if character dossier) */}
        {(script.category || script.phonetic || script.scriptFamily) && (
          <div className="p-3.5 rounded-xl bg-zinc-900/40 border border-white/5 text-xs space-y-1.5">
            {script.category && (
              <p>
                <strong className="text-zinc-300">Epigraphic Category:</strong>{' '}
                <span className="text-cyan-400 font-medium">{script.category}</span>
              </p>
            )}
            {script.phonetic && (
              <p>
                <strong className="text-zinc-300">Phonetic Value:</strong>{' '}
                <span className="font-mono text-amber-300">{script.phonetic}</span>
              </p>
            )}
            {script.scriptFamily && (
              <p>
                <strong className="text-zinc-300">Script Family:</strong>{' '}
                <span className="text-zinc-300">{script.scriptFamily}</span>
              </p>
            )}
          </div>
        )}

        {/* Visual Clues */}
        {script.visualClues && (
          <div>
            <h4 className="text-xs font-semibold text-zinc-300 uppercase tracking-wider mb-2">
              Visual Recognition Clues
            </h4>
            <p className="text-xs text-zinc-300 bg-zinc-950/60 p-3 rounded-xl border border-white/5">
              {script.visualClues}
            </p>
          </div>
        )}

        {/* Historical Context */}
        {script.historicalContext && (
          <div>
            <h4 className="text-xs font-semibold text-zinc-300 uppercase tracking-wider mb-2">
              Historical Context
            </h4>
            <p className="text-xs sm:text-sm text-zinc-400 leading-relaxed bg-zinc-950/60 p-3.5 rounded-xl border border-white/5">
              {script.historicalContext}
            </p>
          </div>
        )}

        {/* Paleographic Features */}
        {script.keyFeatures && (
          <div>
            <h4 className="text-xs font-semibold text-zinc-300 uppercase tracking-wider mb-2">
              Key Paleographic Characteristics
            </h4>
            <ul className="space-y-2 text-xs text-zinc-300">
              {script.keyFeatures.map((feat, idx) => (
                <li key={idx} className="flex items-start gap-2 bg-zinc-900/40 p-2.5 rounded-lg border border-white/5">
                  <CheckCircle2 className="w-3.5 h-3.5 text-amber-400 shrink-0 mt-0.5" />
                  <span>{feat}</span>
                </li>
              ))}
            </ul>
          </div>
        )}

        {/* Exemplar Sites */}
        {script.famousInscriptions && (
          <div>
            <h4 className="text-xs font-semibold text-zinc-300 uppercase tracking-wider mb-2">
              Exemplar Archaeological Inscriptions
            </h4>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
              {script.famousInscriptions.map((site, idx) => (
                <div key={idx} className="p-2.5 rounded-lg bg-zinc-950/60 border border-white/5 text-xs text-zinc-300">
                  📍 {site}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Lineage */}
        {(script.ancestor || script.descendants) && (
          <div className="p-3.5 rounded-xl bg-zinc-900/40 border border-white/5 text-xs text-zinc-400 space-y-1">
            {script.ancestor && (
              <p>
                <strong className="text-zinc-200">Ancestor:</strong> {script.ancestor}
              </p>
            )}
            {script.descendants && (
              <p>
                <strong className="text-zinc-200">Descendants:</strong> {script.descendants.join(', ')}
              </p>
            )}
          </div>
        )}

        {/* Footer */}
        <div className="pt-2 flex justify-end">
          <button onClick={onClose} className="btn-clean-secondary text-xs px-4 py-1.5">
            Close
          </button>
        </div>

      </div>
    </div>
  );
}

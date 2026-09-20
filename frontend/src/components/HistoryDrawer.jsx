import React from 'react';
import { X, Trash2, History, Clock, ChevronRight, Sparkles } from 'lucide-react';

export function HistoryDrawer({
  isOpen,
  onClose,
  history = [],
  onSelectHistoryItem,
  onClearHistory
}) {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 overflow-hidden">
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-slate-900/40 backdrop-blur-xs transition-opacity"
        onClick={onClose}
      />

      <div className="fixed inset-y-0 right-0 max-w-full flex pl-10">
        <div className="w-screen max-w-md bg-white border-l border-slate-200 p-5 sm:p-6 flex flex-col justify-between shadow-2xl">
          
          {/* Header */}
          <div>
            <div className="flex items-center justify-between pb-4 border-b border-slate-200">
              <div className="flex items-center gap-2">
                <div className="w-8 h-8 rounded-lg bg-amber-50 text-amber-800 border border-amber-200 flex items-center justify-center">
                  <History className="w-4 h-4" />
                </div>
                <div>
                  <h3 className="font-display text-base font-bold text-slate-900">
                    Session Inscription History
                  </h3>
                  <p className="text-xs text-slate-500">
                    Previous epigraphic classifications
                  </p>
                </div>
              </div>

              <button
                onClick={onClose}
                className="p-1.5 rounded-lg text-slate-400 hover:text-slate-800 hover:bg-slate-100 transition-colors border border-transparent hover:border-slate-200"
                aria-label="Close history"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            {/* Inscriptions History List */}
            <div className="mt-4 space-y-2.5 max-h-[72vh] overflow-y-auto pr-1">
              {history.length === 0 ? (
                <div className="text-center py-16 px-4 rounded-2xl border-2 border-dashed border-slate-200 bg-slate-50/60">
                  <Clock className="w-8 h-8 text-slate-400 mx-auto mb-2.5" />
                  <h4 className="text-xs font-bold text-slate-700 uppercase tracking-wider">No Inscriptions Yet</h4>
                  <p className="text-xs text-slate-500 mt-1 max-w-xs mx-auto">
                    Classify an inscription from the studio to record its epigraphic profile.
                  </p>
                </div>
              ) : (
                history.map((item) => {
                  const confPct = Math.round((item?.result?.confidence ?? 0) * 100);
                  const scriptTitle = item?.result?.details?.scriptFamily || item?.result?.script || 'Unknown Script';
                  const imageName = item?.image?.name || 'Specimen';
                  const imageUrl = item?.image?.url || '';
                  const timestamp = item?.timestamp || '';

                  return (
                    <div
                      key={item.id}
                      onClick={() => {
                        onSelectHistoryItem(item);
                        onClose();
                      }}
                      className="group cursor-pointer rounded-xl bg-white border border-slate-200 hover:border-amber-400 hover:bg-amber-50/30 p-3 flex items-center gap-3 transition-all shadow-2xs hover:shadow-sm"
                    >
                      {/* Image Thumbnail */}
                      <div className="w-14 h-14 rounded-lg overflow-hidden bg-slate-950 shrink-0 border border-slate-200 flex items-center justify-center group-hover:scale-105 transition-transform duration-200">
                        {imageUrl ? (
                          <img
                            src={imageUrl}
                            alt="Thumbnail"
                            className="w-full h-full object-cover"
                          />
                        ) : (
                          <div className="w-full h-full flex items-center justify-center text-slate-400 text-xs font-mono">
                            IMG
                          </div>
                        )}
                      </div>

                      {/* Details */}
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center justify-between">
                          <h4 className="text-xs font-bold text-slate-900 truncate group-hover:text-amber-900">
                            {scriptTitle}
                          </h4>
                          <span className="text-[11px] font-mono font-bold text-emerald-800 bg-emerald-50 px-1.5 py-0.2 rounded border border-emerald-200/80">
                            {confPct}%
                          </span>
                        </div>
                        <p className="text-[11px] text-slate-500 truncate mt-0.5">
                          {imageName}
                        </p>
                        {timestamp && (
                          <p className="text-[10px] text-slate-400 font-mono mt-0.5 flex items-center gap-1">
                            <Clock className="w-2.5 h-2.5" />
                            <span>{timestamp}</span>
                          </p>
                        )}
                      </div>

                      <ChevronRight className="w-4 h-4 text-slate-300 group-hover:text-amber-600 transition-colors shrink-0" />
                    </div>
                  );
                })
              )}
            </div>
          </div>

          {/* Bottom Controls */}
          {history.length > 0 && (
            <div className="pt-4 border-t border-slate-200 flex justify-between items-center text-xs">
              <span className="text-slate-500 font-mono font-medium">
                {history.length} {history.length === 1 ? 'specimen' : 'specimens'} recorded
              </span>
              <button
                onClick={onClearHistory}
                className="text-xs text-slate-500 hover:text-rose-600 inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg hover:bg-rose-50 transition-colors border border-transparent hover:border-rose-200 font-medium"
              >
                <Trash2 className="w-3.5 h-3.5" />
                <span>Clear History</span>
              </button>
            </div>
          )}

        </div>
      </div>
    </div>
  );
}

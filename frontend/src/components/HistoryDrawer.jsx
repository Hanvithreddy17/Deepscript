import React from 'react';
import { X, Trash2, History, Clock } from 'lucide-react';

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
        className="absolute inset-0 bg-black/60 backdrop-blur-xs transition-opacity"
        onClick={onClose}
      />

      <div className="fixed inset-y-0 right-0 max-w-full flex pl-10">
        <div className="w-screen max-w-sm bg-zinc-925 border-l border-white/10 p-5 flex flex-col justify-between shadow-2xl">
          
          {/* Header */}
          <div>
            <div className="flex items-center justify-between pb-3 border-b border-white/10">
              <div className="flex items-center gap-2">
                <History className="w-4 h-4 text-amber-400" />
                <h3 className="font-display text-sm font-semibold text-zinc-100">
                  Classification History
                </h3>
              </div>
              <button
                onClick={onClose}
                className="p-1 rounded-md text-zinc-400 hover:text-white hover:bg-zinc-800 transition-colors"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            {/* List */}
            <div className="mt-4 space-y-2.5 max-h-[72vh] overflow-y-auto pr-1">
              {history.length === 0 ? (
                <div className="text-center py-12 px-4 rounded-xl border border-dashed border-white/5 bg-zinc-950/40">
                  <Clock className="w-6 h-6 text-zinc-600 mx-auto mb-2" />
                  <p className="text-xs text-zinc-400">No inscriptions analyzed yet</p>
                </div>
              ) : (
                history.map((item) => {
                  const confPct = Math.round((item?.result?.confidence ?? 0) * 100);
                  const scriptTitle = item?.result?.script || 'Unknown Script';
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
                      className="cursor-pointer rounded-xl bg-zinc-900/60 border border-white/5 hover:border-white/15 p-2.5 flex items-center gap-3 transition-all hover:bg-zinc-900"
                    >
                      {/* Image Thumbnail */}
                      <div className="w-12 h-12 rounded-lg overflow-hidden bg-zinc-950 shrink-0 border border-white/5">
                        {imageUrl ? (
                          <img
                            src={imageUrl}
                            alt="Thumbnail"
                            className="w-full h-full object-cover"
                          />
                        ) : (
                          <div className="w-full h-full flex items-center justify-center text-zinc-600 text-xs font-mono">
                            IMG
                          </div>
                        )}
                      </div>

                      {/* Details */}
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center justify-between">
                          <h4 className="text-xs font-semibold text-zinc-200 truncate">
                            {scriptTitle}
                          </h4>
                          <span className="text-[10px] font-mono text-emerald-400">
                            {confPct}%
                          </span>
                        </div>
                        <p className="text-[11px] text-zinc-500 truncate mt-0.5">
                          {imageName}
                        </p>
                        {timestamp && (
                          <p className="text-[10px] text-zinc-600 font-mono mt-0.5">
                            {timestamp}
                          </p>
                        )}
                      </div>
                    </div>
                  );
                })
              )}
            </div>
          </div>

          {/* Footer */}
          {history.length > 0 && (
            <div className="pt-3 border-t border-white/10 flex justify-between items-center text-xs">
              <span className="text-zinc-500 font-mono">
                {history.length} {history.length === 1 ? 'record' : 'records'}
              </span>
              <button
                onClick={onClearHistory}
                className="text-xs text-zinc-400 hover:text-rose-400 inline-flex items-center gap-1 transition-colors"
              >
                <Trash2 className="w-3.5 h-3.5" />
                <span>Clear</span>
              </button>
            </div>
          )}

        </div>
      </div>
    </div>
  );
}

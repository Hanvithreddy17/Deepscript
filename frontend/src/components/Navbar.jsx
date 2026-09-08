import React from 'react';
import { History, HelpCircle } from 'lucide-react';

export function Navbar({
  onOpenHistory,
  onOpenInfoModal,
  historyCount = 0,
  isLiveApi = false,
}) {
  return (
    <header className="sticky top-0 z-40 w-full border-b border-white/[0.08] bg-zinc-950/80 backdrop-blur-md">
      <div className="max-w-4xl mx-auto px-4 sm:px-6 h-15 flex items-center justify-between">
        
        {/* Clean Logo */}
        <div className="flex items-center gap-2.5">
          <div className="flex items-center justify-center w-7 h-7 rounded-lg bg-zinc-900 border border-white/10 text-amber-400 font-semibold text-xs">
            𑀅
          </div>
          <div className="flex items-center gap-2">
            <span className="font-display font-semibold text-sm sm:text-base tracking-tight text-zinc-100">
              DeepScript
            </span>
            <span className="text-[10px] font-medium text-zinc-500 bg-zinc-900 px-1.5 py-0.5 rounded border border-white/5">
              Indian Epigraphy
            </span>
          </div>
        </div>

        {/* Action Controls */}
        <div className="flex items-center gap-2">
          {/* Live Status Badge */}
          <div className={`hidden sm:flex items-center gap-1.5 px-2 py-1 rounded-full text-[11px] font-medium border ${
            isLiveApi 
              ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20' 
              : 'bg-zinc-900 text-zinc-400 border-white/5'
          }`}>
            <span className={`w-1.5 h-1.5 rounded-full ${isLiveApi ? 'bg-emerald-400 animate-pulse' : 'bg-zinc-500'}`} />
            <span>{isLiveApi ? 'ViT Model Live' : 'Simulation Mode'}</span>
          </div>
          
          {/* History */}
          <button
            onClick={onOpenHistory}
            className="btn-clean-secondary text-xs py-1 px-2.5 flex items-center gap-1.5"
            title="Classification history"
          >
            <History className="w-3.5 h-3.5 text-zinc-400" />
            <span>History</span>
            {historyCount > 0 && (
              <span className="text-[10px] bg-zinc-800 text-zinc-300 px-1.5 py-0.2 rounded font-mono font-medium">
                {historyCount}
              </span>
            )}
          </button>

          {/* About */}
          <button
            onClick={onOpenInfoModal}
            className="p-1.5 rounded-lg text-zinc-400 hover:text-zinc-100 hover:bg-zinc-900 transition-colors border border-transparent hover:border-white/5"
            title="Epigraphic guidelines and scope"
          >
            <HelpCircle className="w-4 h-4" />
          </button>
        </div>

      </div>
    </header>
  );
}

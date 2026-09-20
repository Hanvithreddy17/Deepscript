import React from 'react';
import { History, HelpCircle, BookOpen, Sparkles, Cpu, Settings } from 'lucide-react';

export function Navbar({
  onOpenHistory,
  onOpenInfoModal,
  onOpenLibrary,
  onOpenSettings,
  historyCount = 0,
  isLiveApi = false,
  activeModel = 'Qwen2.5-VL',
}) {
  const modelShortName = activeModel.includes('Qwen') 
    ? 'Qwen2.5-VL (Free)' 
    : (activeModel.includes('Llama') ? 'Llama 3.2 Vision' : 'HF VLM Engine');

  return (
    <header className="sticky top-0 z-40 w-full border-b border-slate-200/80 bg-white/85 backdrop-blur-md transition-all">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
        
        {/* Brand Logo & Title */}
        <div className="flex items-center gap-3">
          <div className="flex items-center justify-center w-9 h-9 rounded-xl bg-gradient-to-br from-amber-500 to-amber-600 text-white font-bold text-sm shadow-sm shadow-amber-500/20 border border-amber-400/40">
            𑀅
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="font-display font-bold text-base sm:text-lg tracking-tight text-slate-900">
                DeepScript
              </span>
              <span className="hidden sm:inline-flex items-center gap-1 text-[11px] font-semibold text-amber-800 bg-amber-50 border border-amber-200/70 px-2 py-0.5 rounded-full">
                <Sparkles className="w-3 h-3 text-amber-600" />
                Hugging Face VLM
              </span>
            </div>
            <p className="text-[11px] text-slate-500 hidden md:block">
              Free Multimodal Paleography & Ancient Indian Script Recognition
            </p>
          </div>
        </div>

        {/* Action Controls */}
        <div className="flex items-center gap-2.5">
          {/* Live VLM Status Badge / Settings Trigger */}
          <button
            type="button"
            onClick={onOpenSettings}
            className={`flex items-center gap-2 px-2.5 py-1 rounded-full text-xs font-medium border transition-all cursor-pointer hover:scale-[1.02] ${
              isLiveApi 
                ? 'bg-emerald-50 text-emerald-800 border-emerald-200/80 shadow-2xs hover:bg-emerald-100/70' 
                : 'bg-slate-100 text-slate-600 border-slate-200 hover:bg-slate-200/70'
            }`}
            title="Configure Hugging Face VLM model & token"
          >
            <span className="relative flex h-2 w-2">
              {isLiveApi && (
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
              )}
              <span className={`relative inline-flex rounded-full h-2 w-2 ${isLiveApi ? 'bg-emerald-500' : 'bg-slate-400'}`}></span>
            </span>
            <span className="font-semibold text-[11px] sm:text-xs">
              {isLiveApi ? modelShortName : 'Connecting Engine...'}
            </span>
            <Settings className="w-3 h-3 text-slate-400 hover:text-slate-700" />
          </button>

          {/* Script Library Button */}
          {onOpenLibrary && (
            <button
              onClick={onOpenLibrary}
              className="btn-clean-secondary text-xs py-1.5 px-3 flex items-center gap-1.5"
              title="Browse supported Indian script families"
            >
              <BookOpen className="w-3.5 h-3.5 text-slate-500" />
              <span className="hidden md:inline">Script Library</span>
            </button>
          )}

          {/* History Drawer Button */}
          <button
            onClick={onOpenHistory}
            className="btn-clean-secondary text-xs py-1.5 px-3 flex items-center gap-1.5"
            title="Classification history"
          >
            <History className="w-3.5 h-3.5 text-slate-500" />
            <span>History</span>
            {historyCount > 0 && (
              <span className="text-[10px] bg-amber-100 text-amber-800 px-1.5 py-0.2 rounded-full font-mono font-semibold">
                {historyCount}
              </span>
            )}
          </button>

          {/* Epigraphic Scope & Help */}
          <button
            onClick={onOpenInfoModal}
            className="p-2 rounded-lg text-slate-500 hover:text-slate-800 hover:bg-slate-100 transition-colors border border-slate-200 shadow-2xs"
            title="Epigraphic guidelines and scope"
            aria-label="Epigraphic guidelines"
          >
            <HelpCircle className="w-4 h-4" />
          </button>
        </div>

      </div>
    </header>
  );
}

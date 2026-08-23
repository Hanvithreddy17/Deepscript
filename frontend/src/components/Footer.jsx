import React from 'react';

export function Footer({ onOpenInfoModal }) {
  return (
    <footer className="mt-16 border-t border-white/[0.06] py-8 px-4 text-center text-xs text-zinc-500">
      <div className="max-w-4xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-3">
        <p>
          DeepScript • Ancient Indian Script Classification System
        </p>
        <button
          onClick={onOpenInfoModal}
          className="hover:text-zinc-300 transition-colors underline underline-offset-4"
        >
          Scope & Supported Indian Scripts
        </button>
      </div>
    </footer>
  );
}

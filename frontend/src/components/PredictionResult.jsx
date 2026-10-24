import React, { useState } from 'react';
import { 
  ChevronRight, 
  Clock, 
  MapPin, 
  Sparkles, 
  BookOpen, 
  CheckCircle2, 
  Copy, 
  Check, 
  Compass, 
  ArrowRightLeft, 
  AlertTriangle, 
  SearchX, 
  Info, 
  ShieldAlert 
} from 'lucide-react';
import { getCleanBasename } from '../utils/formatters';

const SCRIPT_NAME_MAP = {
  brahmi: 'Ashokan Brahmi',
  ashokan_brahmi: 'Ashokan Brahmi',
  tamil_brahmi: 'Tamil-Brahmi',
  grantha: 'Grantha Script',
  gupta: 'Gupta Script',
  kadamba: 'Kadamba Script',
  kharosthi: 'Kharosthi',
};

/**
 * Dynamically resolves the canonical human-readable script name from backend prediction payload.
 */
export function getCanonicalScriptName(result) {
  if (!result) return 'Unknown Script';

  const rawKey = (result.raw_class || result.script || '').toLowerCase().replace(/[\s-]+/g, '_').trim();
  if (SCRIPT_NAME_MAP[rawKey]) {
    return SCRIPT_NAME_MAP[rawKey];
  }

  return (
    result.details?.scriptFamily ||
    result.details?.name ||
    result.script_family ||
    result.script ||
    'Unknown Script'
  );
}

export function PredictionResult({
  result,
  specimenName,
  onOpenDetails
}) {
  const [activeTab, setActiveTab] = useState('paleography');
  const [copied, setCopied] = useState(false);

  if (!result) return null;

  const { script, candidates = [], sourceLabel, executionTimeMs, details } = result;
  const predictedScriptDisplay = getCanonicalScriptName(result);
  const scriptFamily = predictedScriptDisplay;

  // Normalized specimen basename (e.g. "faculty_tamil_brahmi.png")
  const rawFilename = result.image_details?.original_filename || result.filename || specimenName || '';
  const specimenBasename = getCleanBasename(rawFilename);

  const handleCopyJson = () => {
    navigator.clipboard.writeText(JSON.stringify(result, null, 2));
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  // 1. Invalid Input Image State
  if (result.status === 'invalid_input' || scriptFamily === 'Invalid input image') {
    return (
      <div className="w-full clean-card p-5 sm:p-6 space-y-5 bg-white shadow-soft animate-in fade-in duration-200">
        <div className="pb-4 border-b border-slate-200/80">
          <div className="flex flex-wrap items-center justify-between gap-2 mb-2.5">
            <div className="flex items-center gap-2">
              <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-rose-50 border border-rose-200 text-rose-800 text-xs font-bold uppercase tracking-wider">
                <AlertTriangle className="w-3.5 h-3.5 text-rose-600" />
                Input Validation
              </span>
              <span className="text-[10px] font-semibold text-slate-500 bg-slate-100 px-2 py-0.5 rounded border border-slate-200">
                {sourceLabel || 'DeepScript Input Validator'}
              </span>
            </div>
            {executionTimeMs !== undefined && (
              <span className="text-[11px] font-mono text-slate-500 bg-slate-100 px-2 py-0.5 rounded border border-slate-200">
                {executionTimeMs}ms
              </span>
            )}
          </div>

          {specimenBasename && (
            <div className="flex items-center gap-1.5 text-xs text-slate-500 mb-2">
              <span className="font-semibold text-slate-700">Specimen File:</span>
              <span className="font-mono bg-slate-100 text-slate-800 px-2 py-0.5 rounded border border-slate-200">
                {specimenBasename}
              </span>
            </div>
          )}

          <div className="py-2">
            <span className="text-xs font-bold uppercase tracking-widest text-rose-700 block mb-1">
              Validation Result
            </span>
            <h1 className="font-display text-2xl sm:text-3xl font-extrabold text-slate-900 tracking-tight">
              Invalid input image
            </h1>
          </div>
          <p className="text-xs text-rose-700 mt-2 leading-relaxed bg-rose-50/70 border border-rose-200/80 rounded-xl p-3">
            {result.rejectionReason || details?.historicalContext || 'The uploaded file could not be decoded as a valid, non-empty image.'}
          </p>
        </div>

        <div className="space-y-3">
          <h4 className="text-xs font-bold text-slate-800 uppercase tracking-wider flex items-center gap-1.5">
            <Info className="w-3.5 h-3.5 text-slate-500" />
            <span>Image Requirements</span>
          </h4>
          <div className="p-3.5 rounded-xl bg-slate-50 border border-slate-200/80 space-y-2 text-xs text-slate-600">
            <div className="flex items-start gap-2">
              <CheckCircle2 className="w-4 h-4 text-slate-400 shrink-0 mt-0.5" />
              <span>Supported formats: standard raster images (PNG, JPEG, WebP, TIFF).</span>
            </div>
            <div className="flex items-start gap-2">
              <CheckCircle2 className="w-4 h-4 text-slate-400 shrink-0 mt-0.5" />
              <span>Ensure the file is not empty (0 bytes) or corrupted.</span>
            </div>
            <div className="flex items-start gap-2">
              <CheckCircle2 className="w-4 h-4 text-slate-400 shrink-0 mt-0.5" />
              <span>Ensure image resolution is at least 10×10 pixels.</span>
            </div>
          </div>
        </div>

        <div className="pt-3 border-t border-slate-200/80 flex items-center justify-end">
          <button
            type="button"
            onClick={handleCopyJson}
            className="text-xs text-slate-500 hover:text-slate-800 inline-flex items-center gap-1 px-2.5 py-1.5 rounded-md hover:bg-slate-100 transition-colors"
            title="Copy raw validation payload"
          >
            {copied ? <Check className="w-3.5 h-3.5 text-emerald-600" /> : <Copy className="w-3.5 h-3.5" />}
            <span>{copied ? 'Copied' : 'Copy JSON'}</span>
          </button>
        </div>
      </div>
    );
  }

  // 2. Unsupported Script State (Tier 3 Open-Set Rejection)
  if (result.status === 'unsupported_script' || scriptFamily.toLowerCase().includes('unsupported script')) {
    return (
      <div className="w-full clean-card p-5 sm:p-6 space-y-5 bg-white shadow-soft animate-in fade-in duration-200">
        <div className="pb-4 border-b border-slate-200/80">
          <div className="flex flex-wrap items-center justify-between gap-2 mb-2.5">
            <div className="flex items-center gap-2">
              <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-purple-50 border border-purple-200 text-purple-800 text-xs font-bold uppercase tracking-wider">
                <ShieldAlert className="w-3.5 h-3.5 text-purple-600" />
                Unsupported Script
              </span>
              <span className="text-[10px] font-semibold text-slate-500 bg-slate-100 px-2 py-0.5 rounded border border-slate-200">
                {sourceLabel || 'DeepScript Open-Set Rejector'}
              </span>
            </div>

            <div className="flex items-center gap-2">
              <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-xs font-semibold border bg-purple-50 text-purple-700 border-purple-200">
                <AlertTriangle className="w-3.5 h-3.5 text-purple-600" />
                <span>Safely Abstained</span>
              </span>
              {executionTimeMs !== undefined && (
                <span className="text-[11px] font-mono text-slate-500 bg-slate-100 px-2 py-0.5 rounded border border-slate-200">
                  {executionTimeMs}ms
                </span>
              )}
            </div>
          </div>

          {specimenBasename && (
            <div className="flex items-center gap-1.5 text-xs text-slate-500 mb-2">
              <span className="font-semibold text-slate-700">Specimen File:</span>
              <span className="font-mono bg-slate-100 text-slate-800 px-2 py-0.5 rounded border border-slate-200">
                {specimenBasename}
              </span>
            </div>
          )}

          <div className="py-2">
            <span className="text-xs font-bold uppercase tracking-widest text-purple-700 block mb-1">
              Predicted Script
            </span>
            <h1 className="font-display text-2xl sm:text-3xl font-extrabold text-slate-900 tracking-tight">
              Unsupported Script Inscription
            </h1>
          </div>
          <p className="text-xs text-purple-900 mt-2 leading-relaxed bg-purple-50/70 border border-purple-200/80 rounded-xl p-3">
            {result.rejectionReason || details?.historicalContext || 'The inscription features do not match the canonical metric prototypes for the 6 supported ancient scripts.'}
          </p>
        </div>

        {/* Visible Limitation Notice */}
        <div className="p-3.5 rounded-xl bg-amber-50/90 border border-amber-200/90 text-amber-950 text-xs flex items-start gap-2.5 shadow-2xs">
          <AlertTriangle className="w-4 h-4 text-amber-600 shrink-0 mt-0.5" />
          <div className="leading-relaxed">
            <span className="font-bold block sm:inline text-amber-900">Limitation Notice: </span>
            <span>This is a model prediction, not a verified epigraphic identification. Historical period and script attribution require expert confirmation.</span>
          </div>
        </div>

        {/* Diagnostic Assessment */}
        <div className="space-y-3">
          <div className="p-3.5 rounded-xl bg-slate-50 border border-slate-200/80 space-y-1.5">
            <h4 className="text-xs font-bold text-slate-800 uppercase tracking-wider flex items-center gap-1.5">
              <Compass className="w-3.5 h-3.5 text-purple-600" />
              <span>Open-Set Epigraphic Assessment</span>
            </h4>
            <p className="text-xs text-slate-600 leading-relaxed">
              {details?.visualClues || 'The metric embedding distance to all supported class prototypes exceeded calibrated acceptance thresholds. DeepScript safely abstained to prevent false historical attribution.'}
            </p>
          </div>

          {candidates && candidates.length > 0 && (
            <div className="space-y-2 pt-1">
              <h5 className="text-[11px] font-bold text-slate-500 uppercase tracking-wider">
                Nearest Candidate Scripts
              </h5>
              <div className="flex flex-wrap gap-1.5">
                {candidates.map((cand, idx) => (
                  <span key={cand.script || idx} className="px-2.5 py-1 rounded-md bg-slate-100 border border-slate-200 text-xs font-medium text-slate-700">
                    {cand.script_family || cand.script}
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>

        <div className="pt-3 border-t border-slate-200/80 flex items-center justify-end">
          <button
            type="button"
            onClick={handleCopyJson}
            className="text-xs text-slate-500 hover:text-slate-800 inline-flex items-center gap-1 px-2.5 py-1.5 rounded-md hover:bg-slate-100 transition-colors"
            title="Copy raw rejection payload"
          >
            {copied ? <Check className="w-3.5 h-3.5 text-emerald-600" /> : <Copy className="w-3.5 h-3.5" />}
            <span>{copied ? 'Copied' : 'Copy JSON'}</span>
          </button>
        </div>
      </div>
    );
  }

  // 3. Script Not Detected State (Tier 2 Visual Presence Rejection)
  if (result.status === 'script_not_detected' || scriptFamily === 'Script not detected') {
    return (
      <div className="w-full clean-card p-5 sm:p-6 space-y-5 bg-white shadow-soft animate-in fade-in duration-200">
        <div className="pb-4 border-b border-slate-200/80">
          <div className="flex flex-wrap items-center justify-between gap-2 mb-2.5">
            <div className="flex items-center gap-2">
              <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-amber-50 border border-amber-200 text-amber-800 text-xs font-bold uppercase tracking-wider">
                <SearchX className="w-3.5 h-3.5 text-amber-600" />
                Detection Status
              </span>
              <span className="text-[10px] font-semibold text-slate-500 bg-slate-100 px-2 py-0.5 rounded border border-slate-200">
                {sourceLabel || 'DeepScript ViT-B/16'}
              </span>
            </div>

            <div className="flex items-center gap-2">
              <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-xs font-semibold border bg-slate-100 text-slate-700 border-slate-200">
                <AlertTriangle className="w-3.5 h-3.5 text-slate-500" />
                <span>Uncertain Match</span>
              </span>
              {executionTimeMs !== undefined && (
                <span className="text-[11px] font-mono text-slate-500 bg-slate-100 px-2 py-0.5 rounded border border-slate-200">
                  {executionTimeMs}ms
                </span>
              )}
            </div>
          </div>

          {specimenBasename && (
            <div className="flex items-center gap-1.5 text-xs text-slate-500 mb-2">
              <span className="font-semibold text-slate-700">Specimen File:</span>
              <span className="font-mono bg-slate-100 text-slate-800 px-2 py-0.5 rounded border border-slate-200">
                {specimenBasename}
              </span>
            </div>
          )}

          <div className="py-2">
            <span className="text-xs font-bold uppercase tracking-widest text-amber-700 block mb-1">
              Predicted Script
            </span>
            <h1 className="font-display text-2xl sm:text-3xl font-extrabold text-slate-900 tracking-tight">
              Script not detected
            </h1>
          </div>
          <p className="text-xs text-amber-900 mt-2 leading-relaxed bg-amber-50/70 border border-amber-200/80 rounded-xl p-3">
            {result.rejectionReason || details?.historicalContext || 'The system could not confidently establish that the image contains a supported ancient Indian script inscription.'}
          </p>
        </div>

        {/* Visible Limitation Notice */}
        <div className="p-3.5 rounded-xl bg-amber-50/90 border border-amber-200/90 text-amber-950 text-xs flex items-start gap-2.5 shadow-2xs">
          <AlertTriangle className="w-4 h-4 text-amber-600 shrink-0 mt-0.5" />
          <div className="leading-relaxed">
            <span className="font-bold block sm:inline text-amber-900">Limitation Notice: </span>
            <span>This is a model prediction, not a verified epigraphic identification. Historical period and script attribution require expert confirmation.</span>
          </div>
        </div>

        {/* Diagnostic Assessment */}
        <div className="space-y-3">
          <div className="p-3.5 rounded-xl bg-slate-50 border border-slate-200/80 space-y-1.5">
            <h4 className="text-xs font-bold text-slate-800 uppercase tracking-wider flex items-center gap-1.5">
              <Compass className="w-3.5 h-3.5 text-indigo-600" />
              <span>Epigraphic Assessment</span>
            </h4>
            <p className="text-xs text-slate-600 leading-relaxed">
              {details?.visualClues || 'Input features do not meet canonical characteristics for Ashokan Brahmi, Tamil-Brahmi, Gupta, Kharosthi, Grantha, or Kadamba inscriptions. The model abstained from making an authoritative prediction.'}
            </p>
          </div>

          {candidates && candidates.length > 0 && (
            <div className="space-y-2 pt-1">
              <h5 className="text-[11px] font-bold text-slate-500 uppercase tracking-wider">
                Candidate Classes Evaluated
              </h5>
              <div className="flex flex-wrap gap-1.5">
                {candidates.map((cand, idx) => (
                  <span key={cand.script || idx} className="px-2.5 py-1 rounded-md bg-slate-100 border border-slate-200 text-xs font-medium text-slate-700">
                    {cand.script_family || cand.script}
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>

        <div className="pt-3 border-t border-slate-200/80 flex items-center justify-end">
          <button
            type="button"
            onClick={handleCopyJson}
            className="text-xs text-slate-500 hover:text-slate-800 inline-flex items-center gap-1 px-2.5 py-1.5 rounded-md hover:bg-slate-100 transition-colors"
            title="Copy raw rejection payload"
          >
            {copied ? <Check className="w-3.5 h-3.5 text-emerald-600" /> : <Copy className="w-3.5 h-3.5" />}
            <span>{copied ? 'Copied' : 'Copy JSON'}</span>
          </button>
        </div>
      </div>
    );
  }

  // 4. Standard Accepted Prediction State — Prominently Displaying Only the Predicted Script
  return (
    <div className="w-full clean-card p-5 sm:p-6 space-y-5 bg-white shadow-soft animate-in fade-in duration-200">
      
      {/* 1. MODEL CLASSIFIER OUTPUT SECTION */}
      <div className="pb-4 border-b border-slate-200/80 space-y-3">
        
        {/* Top Meta Bar */}
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div className="flex items-center gap-2">
            <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-amber-50 border border-amber-200/90 text-amber-950 text-xs font-bold uppercase tracking-wider">
              <Sparkles className="w-3.5 h-3.5 text-amber-700" />
              Script Classification Result
            </span>
            <span className="hidden sm:inline-flex items-center text-[10px] font-semibold text-slate-500 bg-slate-100 px-2 py-0.5 rounded border border-slate-200">
              {sourceLabel || 'DeepScript ViT-B/16 (6-Class Model)'}
            </span>
          </div>

          <div className="flex items-center gap-2">
            <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-xs font-semibold border bg-emerald-50 text-emerald-800 border-emerald-200">
              <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600" />
              <span>Identified</span>
            </span>
            {executionTimeMs !== undefined && (
              <span className="text-[11px] font-mono text-slate-500 bg-slate-100 px-2 py-0.5 rounded border border-slate-200">
                {executionTimeMs}ms
              </span>
            )}
          </div>
        </div>

        {/* Specimen Basename */}
        {specimenBasename && (
          <div className="flex items-center gap-2 text-xs text-slate-600 bg-slate-50 border border-slate-200/80 px-3 py-1.5 rounded-lg">
            <span className="font-semibold text-slate-700 shrink-0">Analyzed Specimen:</span>
            <span className="font-mono font-medium text-slate-900 truncate">
              {specimenBasename}
            </span>
            {result.image_details?.dimensions && (
              <span className="text-[10px] text-slate-400 font-mono pl-1.5 border-l border-slate-200 shrink-0">
                {result.image_details.dimensions}
              </span>
            )}
          </div>
        )}

        {/* Clean, Prominent Predicted Script Display */}
        <div className="pt-2 pb-1">
          <span className="text-xs font-bold uppercase tracking-widest text-amber-800/90 block mb-1">
            Predicted Script
          </span>
          <h1 className="font-display text-3xl sm:text-4xl md:text-5xl font-extrabold text-slate-950 tracking-tight">
            {predictedScriptDisplay}
          </h1>
        </div>

      </div>

      {/* 2. MANDATORY LIMITATION NOTICE */}
      <div className="p-3.5 rounded-xl bg-amber-50/90 border border-amber-200/90 text-amber-950 text-xs flex items-start gap-2.5 shadow-2xs">
        <AlertTriangle className="w-4 h-4 text-amber-600 shrink-0 mt-0.5" />
        <div className="leading-relaxed">
          <span className="font-bold block sm:inline text-amber-900">Limitation Notice: </span>
          <span>This is a model prediction, not a verified epigraphic identification. Historical period and script attribution require expert confirmation.</span>
        </div>
      </div>

      {/* 3. GENERAL SCRIPT FAMILY REFERENCE SECTION */}
      <div className="p-3.5 rounded-xl bg-slate-50/90 border border-slate-200/80 space-y-2">
        <div className="flex items-center gap-1.5 text-xs font-bold uppercase tracking-wider text-slate-700">
          <BookOpen className="w-3.5 h-3.5 text-slate-500" />
          <span>Script Family Reference Dossier (General Historical Context)</span>
        </div>
        <p className="text-[11px] text-slate-500 leading-relaxed">
          The dates, geographic regions, and paleographical archetypes below describe the general <strong className="text-slate-700">{scriptFamily}</strong> script family in archaeological records. They are reference context only, not facts established about the uploaded specimen.
        </p>

        {/* General Script Metadata Badges */}
        <div className="flex flex-wrap items-center gap-2 pt-1">
          {details?.writingDirection && (
            <span className="px-2.5 py-1 rounded-lg bg-indigo-50 border border-indigo-200 text-xs font-bold text-indigo-900 flex items-center gap-1" title="General script family writing direction">
              <ArrowRightLeft className="w-3 h-3 text-indigo-600" />
              <span>Direction: {details.writingDirection}</span>
            </span>
          )}

          {details?.period && (
            <span className="px-2.5 py-1 rounded-lg bg-white border border-slate-200 text-xs text-slate-700 font-medium flex items-center gap-1" title="Historical era associated with this script family">
              <Clock className="w-3 h-3 text-slate-400" />
              <span>Historical Era: {details.period}</span>
            </span>
          )}

          {details?.region && (
            <span className="px-2.5 py-1 rounded-lg bg-white border border-slate-200 text-xs text-slate-700 font-medium flex items-center gap-1" title="Geographical distribution of this script tradition">
              <MapPin className="w-3 h-3 text-slate-400" />
              <span className="truncate max-w-[240px]">Geographic Tradition: {details.region}</span>
            </span>
          )}
        </div>
      </div>

      {/* 4. DOSSIER NAVIGATION TABS */}
      <div className="flex items-center gap-2 border-b border-slate-200/80 pb-2">
        <button
          type="button"
          onClick={() => setActiveTab('paleography')}
          className={`px-3 py-1.5 text-xs font-bold rounded-lg transition-all flex items-center gap-1.5 ${
            activeTab === 'paleography'
              ? 'bg-slate-900 text-white shadow-2xs'
              : 'text-slate-600 hover:text-slate-900 hover:bg-slate-100'
          }`}
        >
          <BookOpen className="w-3.5 h-3.5" />
          <span>Reference Paleography</span>
        </button>

        <button
          type="button"
          onClick={() => setActiveTab('history')}
          className={`px-3 py-1.5 text-xs font-bold rounded-lg transition-all flex items-center gap-1.5 ${
            activeTab === 'history'
              ? 'bg-slate-900 text-white shadow-2xs'
              : 'text-slate-600 hover:text-slate-900 hover:bg-slate-100'
          }`}
        >
          <Compass className="w-3.5 h-3.5" />
          <span>Historical Context & Sites</span>
        </button>
      </div>

      {/* 5. TAB CONTENTS */}
      <div className="min-h-[140px]">
        {/* Tab 1: Reference Paleography */}
        {activeTab === 'paleography' && (
          <div className="space-y-3 animate-in fade-in duration-150">
            {/* Script Family Archetype */}
            <div className="p-3.5 rounded-xl bg-slate-50 border border-slate-200/80 space-y-1.5">
              <h4 className="text-xs font-bold text-slate-800 uppercase tracking-wider flex items-center gap-1.5">
                <Sparkles className="w-3.5 h-3.5 text-amber-600" />
                <span>Script Family Typological Archetype</span>
              </h4>
              <p className="text-xs text-slate-700 leading-relaxed">
                {details?.visualClues || details?.strokeStyle || 'Standard stylistic archetype documented in the epigraphic reference database.'}
              </p>
            </div>

            {/* Headmark & Stroke Style Chips */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 text-xs">
              {details?.headmark && (
                <div className="p-2.5 rounded-lg bg-amber-50/60 border border-amber-200/70 text-amber-950">
                  <span className="font-bold text-[11px] text-amber-900 block mb-0.5">Canonical Headmark Typology</span>
                  <span>{details.headmark}</span>
                </div>
              )}
              {details?.stroke_style && (
                <div className="p-2.5 rounded-lg bg-slate-50 border border-slate-200 text-slate-700">
                  <span className="font-bold text-[11px] text-slate-800 block mb-0.5">Stroke Style Tradition</span>
                  <span>{details.stroke_style}</span>
                </div>
              )}
            </div>

            {/* Canonical Reference Glyphs */}
            {details?.keyGlyphs && details.keyGlyphs.length > 0 && (
              <div className="space-y-1.5">
                <div className="flex items-center justify-between">
                  <h5 className="text-[11px] font-bold text-slate-500 uppercase tracking-wider">
                    Canonical Reference Glyphs for {scriptFamily}
                  </h5>
                </div>
                <p className="text-[11px] text-slate-500 leading-relaxed">
                  Representative glyph archetypes from the canonical epigraphic reference database. DeepScript analyzes whole-inscription visual embeddings; individual characters in uploaded photographs are not segmentally transcribed or verified without manual paleographic review.
                </p>
                <div className="flex flex-wrap gap-1.5 pt-1">
                  {details.keyGlyphs.map((glyph, idx) => (
                    <span key={idx} className="px-2.5 py-1 rounded-md bg-white border border-slate-200 text-xs font-mono text-slate-800 shadow-2xs">
                      {glyph}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {/* Tab 2: Historical Context & Sites */}
        {activeTab === 'history' && (
          <div className="space-y-3 animate-in fade-in duration-150">
            <div className="p-3.5 rounded-xl bg-slate-50 border border-slate-200/80 space-y-1.5">
              <h4 className="text-xs font-bold text-slate-800 uppercase tracking-wider flex items-center gap-1.5">
                <Compass className="w-3.5 h-3.5 text-indigo-600" />
                <span>Script Family Historical Context</span>
              </h4>
              <p className="text-xs text-slate-700 leading-relaxed">
                {details?.historicalContext || 'Ancient Indian epigraphic record verified with DeepScript epigraphic reference database.'}
              </p>
            </div>

            {details?.famousInscriptions && details.famousInscriptions.length > 0 && (
              <div>
                <h5 className="text-[11px] font-bold text-slate-500 uppercase tracking-wider mb-1.5">
                  Historically Documented Archaeological Sites (Script Family)
                </h5>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-1.5">
                  {details.famousInscriptions.map((site, idx) => (
                    <div key={idx} className="p-2 rounded-lg bg-slate-50 border border-slate-200 text-xs text-slate-700 flex items-center gap-1.5">
                      <MapPin className="w-3 h-3 text-amber-600 shrink-0" />
                      <span className="truncate">{site}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      {/* 6. ACTION FOOTER */}
      <div className="pt-3 border-t border-slate-200/80 flex flex-wrap items-center justify-between gap-2">
        <button
          type="button"
          onClick={() => onOpenDetails(details || { name: script })}
          className="btn-clean-secondary text-xs py-2 px-3.5 flex items-center gap-1.5"
        >
          <BookOpen className="w-3.5 h-3.5 text-slate-500" />
          <span>Explore Full Script Dossier</span>
          <ChevronRight className="w-3.5 h-3.5 text-slate-400" />
        </button>

        <button
          type="button"
          onClick={handleCopyJson}
          className="text-xs text-slate-500 hover:text-slate-800 inline-flex items-center gap-1 px-2.5 py-1.5 rounded-md hover:bg-slate-100 transition-colors"
          title="Copy raw inference payload"
        >
          {copied ? <Check className="w-3.5 h-3.5 text-emerald-600" /> : <Copy className="w-3.5 h-3.5" />}
          <span>{copied ? 'Copied' : 'Copy JSON'}</span>
        </button>
      </div>

    </div>
  );
}

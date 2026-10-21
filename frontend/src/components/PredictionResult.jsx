import React, { useState } from 'react';
import { 
  ChevronRight, 
  Clock, 
  MapPin, 
  Sparkles, 
  BookOpen, 
  CheckCircle2, 
  BarChart3, 
  Copy, 
  Check, 
  ShieldCheck,
  Compass,
  ArrowRightLeft,
  AlertTriangle,
  SearchX,
  Info,
  ShieldAlert,
  FileText
} from 'lucide-react';
import { getCleanBasename } from '../utils/formatters';

export function PredictionResult({
  result,
  specimenName,
  onOpenDetails
}) {
  const [activeTab, setActiveTab] = useState('paleography');
  const [copied, setCopied] = useState(false);

  if (!result) return null;

  const { script, confidence, candidates = [], sourceLabel, executionTimeMs, details, model } = result;
  const scriptFamily = details?.scriptFamily || details?.name || script || 'Ashokan Brahmi';

  // 1. Calibrated confidence (bounded by prototype cosine similarity)
  const calibratedConfidence = confidence ?? 0;
  const calibratedConfidencePct = Math.round(calibratedConfidence * 100);

  // 2. Softmax probability (from morphologyMetrics or candidates[0])
  const softmaxProbability = 
    details?.morphologyMetrics?.top1_probability ?? 
    candidates?.[0]?.probability ?? 
    calibratedConfidence;
  const softmaxProbabilityPct = (softmaxProbability * 100).toFixed(1);

  // 3. Raw metric cosine & authoritative margin
  const cosineSimilarity = typeof details?.morphologyMetrics?.max_cosine_similarity === 'number'
    ? details.morphologyMetrics.max_cosine_similarity
    : (typeof candidates?.[0]?.cosine_similarity === 'number' ? candidates[0].cosine_similarity : null);

  const top1Candidate = candidates?.[0] ?? null;
  const top2Candidate = candidates?.[1] ?? null;

  const top1Cos = typeof top1Candidate?.cosine_similarity === 'number' ? top1Candidate.cosine_similarity : null;
  const top2Cos = typeof top2Candidate?.cosine_similarity === 'number' ? top2Candidate.cosine_similarity : null;

  const top1Name = top1Candidate?.script_family || top1Candidate?.script || scriptFamily || 'Top-1';
  const top2Name = top2Candidate?.script_family || top2Candidate?.script || 'Runner-up';

  // Authoritative margin from backend (calculated from full class list sorted_cos[0] - sorted_cos[1])
  const backendMargin = typeof details?.morphologyMetrics?.classification_margin === 'number'
    ? details.morphologyMetrics.classification_margin
    : null;

  // Fallback calculation: difference between top two cosine similarities from candidate list
  let fallbackMargin = null;
  if (top1Cos !== null && top2Cos !== null) {
    fallbackMargin = Number((top1Cos - top2Cos).toFixed(4));
  }

  const authoritativeMargin = backendMargin !== null ? backendMargin : fallbackMargin;

  // 4. Normalized specimen basename (e.g. "faculty_tamil_brahmi.png")
  const rawFilename = result.image_details?.original_filename || result.filename || specimenName || '';
  const specimenBasename = getCleanBasename(rawFilename);

  const getConfidenceBadge = (pct) => {
    if (pct >= 85) {
      return {
        label: 'High Confidence',
        bg: 'bg-emerald-50 text-emerald-800 border-emerald-200',
        barColor: 'bg-emerald-500'
      };
    } else if (pct >= 60) {
      return {
        label: 'Moderate Match',
        bg: 'bg-amber-50 text-amber-800 border-amber-200',
        barColor: 'bg-amber-500'
      };
    } else {
      return {
        label: 'Exploratory Candidate',
        bg: 'bg-slate-100 text-slate-700 border-slate-200',
        barColor: 'bg-slate-400'
      };
    }
  };

  // Match indicator badge is strictly driven by calibrated confidence
  const badgeInfo = getConfidenceBadge(calibratedConfidencePct);

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

          <h2 className="font-display text-2xl sm:text-3xl font-extrabold text-slate-900 tracking-tight">
            Invalid input image
          </h2>
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

          <h2 className="font-display text-2xl sm:text-3xl font-extrabold text-slate-900 tracking-tight">
            Unsupported Script Inscription
          </h2>
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

        {/* Diagnostic Breakdown */}
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
                Nearest Candidate Alignment (Abstained)
              </h5>
              <div className="space-y-1.5">
                {candidates.map((cand, idx) => {
                  const candProb = cand.probability !== undefined ? (cand.probability * 100).toFixed(1) : '0.0';
                  const candCos = cand.cosine_similarity !== undefined ? cand.cosine_similarity.toFixed(4) : 'N/A';
                  return (
                    <div key={cand.script || idx} className="p-2 rounded-lg bg-slate-50 border border-slate-200/80 flex items-center justify-between text-xs">
                      <span className="font-medium text-slate-700">{cand.script_family || cand.script}</span>
                      <div className="flex items-center gap-3 font-mono text-[11px] text-slate-500">
                        <span>Prob: {candProb}%</span>
                        <span className="text-slate-400">|</span>
                        <span>Cos: {candCos}</span>
                      </div>
                    </div>
                  );
                })}
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
                <ShieldAlert className="w-3.5 h-3.5 text-slate-500" />
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

          <h2 className="font-display text-2xl sm:text-3xl font-extrabold text-slate-900 tracking-tight">
            Script not detected
          </h2>
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

        {/* Diagnostic Breakdown */}
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
                Exploratory Class Alignment (Abstained)
              </h5>
              <div className="space-y-1.5">
                {candidates.map((cand, idx) => {
                  const candProb = cand.probability !== undefined ? (cand.probability * 100).toFixed(1) : '0.0';
                  const candCos = cand.cosine_similarity !== undefined ? cand.cosine_similarity.toFixed(4) : 'N/A';
                  return (
                    <div key={cand.script || idx} className="p-2 rounded-lg bg-slate-50 border border-slate-200/80 flex items-center justify-between text-xs">
                      <span className="font-medium text-slate-700">{cand.script_family || cand.script}</span>
                      <div className="flex items-center gap-3 font-mono text-[11px] text-slate-500">
                        <span>Prob: {candProb}%</span>
                        <span className="text-slate-400">|</span>
                        <span>Cos: {candCos}</span>
                      </div>
                    </div>
                  );
                })}
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

  // 4. Standard Accepted Prediction State
  return (
    <div className="w-full clean-card p-5 sm:p-6 space-y-5 bg-white shadow-soft animate-in fade-in duration-200">
      
      {/* 1. MODEL CLASSIFIER OUTPUT SECTION */}
      <div className="pb-4 border-b border-slate-200/80 space-y-3">
        
        {/* Top Meta Bar */}
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div className="flex items-center gap-2">
            <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-indigo-50 border border-indigo-200 text-indigo-900 text-xs font-bold uppercase tracking-wider">
              <Sparkles className="w-3.5 h-3.5 text-indigo-600" />
              Classifier Output
            </span>
            <span className="hidden sm:inline-flex items-center text-[10px] font-semibold text-slate-500 bg-slate-100 px-2 py-0.5 rounded border border-slate-200">
              {sourceLabel || 'DeepScript ViT-B/16 (6-Class Model)'}
            </span>
          </div>

          <div className="flex items-center gap-2">
            {/* Match Indicator Badge strictly based on calibrated confidence */}
            <span className={`inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-xs font-semibold border ${badgeInfo.bg}`}>
              <ShieldCheck className="w-3.5 h-3.5" />
              <span>{calibratedConfidencePct}% {badgeInfo.label}</span>
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

        {/* Script Class Heading */}
        <div>
          <span className="text-[11px] font-bold uppercase tracking-wider text-slate-500 block">
            Predicted Script Class
          </span>
          <h2 className="font-display text-2xl sm:text-3xl font-extrabold text-slate-900 tracking-tight mt-0.5">
            {scriptFamily}
          </h2>
        </div>

        {/* Metrics Grid: Separate Softmax Probability vs Calibrated Confidence */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 pt-1 text-xs">
          <div className="p-2.5 rounded-xl bg-amber-50/70 border border-amber-200/80">
            <span className="text-[10px] font-bold text-amber-900 uppercase tracking-wider block">
              Calibrated Confidence
            </span>
            <span className="text-base font-extrabold font-mono text-amber-950 block mt-0.5">
              {calibratedConfidencePct}%
            </span>
            <span className="text-[10px] text-amber-800/80 block mt-0.5">
              Cosine alignment bound
            </span>
          </div>

          <div className="p-2.5 rounded-xl bg-slate-50 border border-slate-200">
            <span className="text-[10px] font-bold text-slate-600 uppercase tracking-wider block">
              Softmax Probability
            </span>
            <span className="text-base font-extrabold font-mono text-slate-900 block mt-0.5">
              {softmaxProbabilityPct}%
            </span>
            <span className="text-[10px] text-slate-500 block mt-0.5">
              Posterior distribution
            </span>
          </div>

          <div className="p-2.5 rounded-xl bg-slate-50 border border-slate-200">
            <span className="text-[10px] font-bold text-slate-600 uppercase tracking-wider block">
              Cosine Similarity
            </span>
            <span className="text-base font-extrabold font-mono text-slate-900 block mt-0.5">
              {cosineSimilarity !== null ? `${Number(cosineSimilarity) > 0 ? '+' : ''}${Number(cosineSimilarity).toFixed(4)}` : 'N/A'}
            </span>
            <span className="text-[10px] text-slate-500 block mt-0.5">
              Prototype vector align
            </span>
          </div>

          <div 
            className="p-2.5 rounded-xl bg-slate-50 border border-slate-200"
            title={top2Candidate && top1Cos !== null && top2Cos !== null ? `Top-1 (${top1Name}: ${top1Cos >= 0 ? '+' : ''}${top1Cos.toFixed(4)}) minus Top-2 (${top2Name}: ${top2Cos >= 0 ? '+' : ''}${top2Cos.toFixed(4)})` : 'Cosine separation between top two candidate classes'}
          >
            <span className="text-[10px] font-bold text-slate-600 uppercase tracking-wider block">
              Cosine Margin (Top-1 vs Top-2)
            </span>
            <span className="text-base font-extrabold font-mono text-slate-900 block mt-0.5">
              {authoritativeMargin !== null
                ? `${authoritativeMargin >= 0 ? '+' : ''}${Number(authoritativeMargin).toFixed(4)}`
                : 'Unavailable'}
            </span>
            <span className="text-[10px] text-slate-500 block mt-0.5 truncate">
              {top2Candidate ? `${top1Name.split(' ')[0]} vs ${top2Name.split(' ')[0]} Δ` : 'Cosine separation over runner-up'}
            </span>
          </div>
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
              <span>Historical Era (Script Family): {details.period}</span>
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

        <button
          type="button"
          onClick={() => setActiveTab('candidates')}
          className={`px-3 py-1.5 text-xs font-bold rounded-lg transition-all flex items-center gap-1.5 ${
            activeTab === 'candidates'
              ? 'bg-slate-900 text-white shadow-2xs'
              : 'text-slate-600 hover:text-slate-900 hover:bg-slate-100'
          }`}
        >
          <BarChart3 className="w-3.5 h-3.5" />
          <span>Comparative Candidates ({candidates.length})</span>
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

        {/* Tab 3: Comparative Candidates */}
        {activeTab === 'candidates' && (
          <div className="space-y-2.5 animate-in fade-in duration-150">
            <div>
              <h4 className="text-xs font-bold text-slate-700 uppercase tracking-wider">
                Comparative Script Family Candidate Distribution
              </h4>
              <p className="text-[11px] text-slate-500 mt-0.5">
                Softmax probability reflects the model's scaled posterior distribution. Calibrated confidence is bounded by metric cosine alignment with each script prototype.
              </p>
            </div>
            
            <div className="space-y-2 pt-1">
              {candidates.map((cand, idx) => {
                const candProb = cand.probability !== undefined ? (cand.probability * 100).toFixed(1) : '0.0';
                const candCos = cand.cosine_similarity !== undefined ? cand.cosine_similarity.toFixed(4) : null;
                const candConf = cand.confidence !== undefined ? Math.round(cand.confidence * 100) : Math.round(Number(candProb));
                const barWidth = Math.max(2, Math.min(100, Math.round(Number(candProb))));

                return (
                  <div key={cand.script || idx} className="p-2.5 rounded-xl bg-slate-50 border border-slate-200/80 space-y-1.5">
                    <div className="flex justify-between items-center text-xs">
                      <div>
                        <span className={`font-semibold ${idx === 0 ? 'text-amber-950 font-bold' : 'text-slate-700'}`}>
                          {cand.script_family || cand.script}
                        </span>
                        {cand.period && (
                          <span className="text-[10px] text-slate-400 pl-1.5 hidden sm:inline">
                            • {cand.period}
                          </span>
                        )}
                      </div>
                      
                      <div className="flex items-center gap-2.5 font-mono text-[11px]">
                        <span className="text-slate-600" title="Softmax Probability">
                          Prob: <strong className="text-slate-900">{candProb}%</strong>
                        </span>
                        {candCos !== null && (
                          <span className="text-slate-500 hidden sm:inline" title="Prototype Cosine Alignment">
                            Cos: <strong>{candCos}</strong>
                          </span>
                        )}
                        <span className="text-amber-700" title="Calibrated Confidence">
                          Conf: <strong>{candConf}%</strong>
                        </span>
                      </div>
                    </div>
                    
                    <div className="w-full h-1.5 rounded-full bg-slate-200 overflow-hidden">
                      <div
                        className={`h-full rounded-full transition-all duration-500 ${
                          idx === 0 ? 'bg-amber-500' : 'bg-slate-400'
                        }`}
                        style={{ width: `${barWidth}%` }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
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

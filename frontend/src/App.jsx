import React, { useState, useEffect } from 'react';
import { Navbar } from './components/Navbar';
import { Hero } from './components/Hero';
import { UploadZone } from './components/UploadZone';
import { PredictionResult } from './components/PredictionResult';
import { ScriptModal } from './components/ScriptModal';
import { ScriptLibraryModal } from './components/ScriptLibraryModal';
import { HfModelSettingsModal } from './components/HfModelSettingsModal';
import { HistoryDrawer } from './components/HistoryDrawer';
import { Footer } from './components/Footer';
import { predictScript, checkBackendStatus } from './services/api';
import { ANCIENT_SCRIPTS } from './data/scriptsData';
import { SAMPLE_INSCRIPTIONS } from './data/sampleImages';
import { 
  X, 
  CheckCircle2, 
  ShieldAlert, 
  Sparkles, 
  Cpu, 
  BookOpen, 
  ArrowRight, 
  Compass, 
  Layers,
  ScanLine
} from 'lucide-react';

export default function App() {
  const [selectedImage, setSelectedImage] = useState(null);
  const [activeSampleId, setActiveSampleId] = useState(null);
  const [filterMode, setFilterMode] = useState('normal'); // 'normal' | 'high-contrast' | 'invert' | 'monochrome'
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [predictionResult, setPredictionResult] = useState(null);
  const [history, setHistory] = useState([]);
  
  const [isHistoryOpen, setIsHistoryOpen] = useState(false);
  const [isLibraryOpen, setIsLibraryOpen] = useState(false);
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [selectedScriptForModal, setSelectedScriptForModal] = useState(null);
  const [isInfoModalOpen, setIsInfoModalOpen] = useState(false);
  
  const [isLiveApi, setIsLiveApi] = useState(false);
  const [activeModel, setActiveModel] = useState(localStorage.getItem('hf_model') || 'Qwen/Qwen2.5-VL-7B-Instruct');
  const [errorMessage, setErrorMessage] = useState(null);

  // Check backend server status on mount
  useEffect(() => {
    async function initCheck() {
      const status = await checkBackendStatus();
      setIsLiveApi(status.online);
      if (status.activeModel) {
        setActiveModel(status.activeModel);
      }
    }
    initCheck();
  }, []);

  // Pre-select first Indian script sample on initial load
  useEffect(() => {
    if (SAMPLE_INSCRIPTIONS.length > 0) {
      handleSelectSample(SAMPLE_INSCRIPTIONS[0]);
    }
  }, []);

  const handleSelectSample = (sample) => {
    setActiveSampleId(sample.id);
    setSelectedImage({
      url: sample.thumbnail,
      name: sample.title,
      size: 'Curated Archaeological Specimen',
      source: 'sample',
      expectedScript: sample.expectedScript,
      medium: sample.medium
    });
    setPredictionResult(null);
    setErrorMessage(null);
  };

  const handleImageUpload = (imageData) => {
    setActiveSampleId(null);
    setSelectedImage(imageData);
    setPredictionResult(null);
    setErrorMessage(null);
  };

  const handleClearImage = () => {
    setSelectedImage(null);
    setActiveSampleId(null);
    setPredictionResult(null);
    setErrorMessage(null);
  };

  const handleAnalyze = async () => {
    if (!selectedImage) return;

    setIsAnalyzing(true);
    setErrorMessage(null);
    try {
      const payload = selectedImage.file || selectedImage.url;
      const metadata = {
        expectedScript: selectedImage.expectedScript,
        name: selectedImage.name
      };

      const result = await predictScript(payload, metadata);
      setPredictionResult(result);
      setIsLiveApi(true);

      const historyItem = {
        id: `scan-${Date.now()}`,
        image: {
          url: selectedImage.url,
          name: selectedImage.name || 'Specimen Inscription'
        },
        result,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      };
      setHistory((prev) => [historyItem, ...prev]);

    } catch (err) {
      console.error('Analysis failed:', err);
      setErrorMessage(err.message || 'An error occurred while communicating with the DeepScript inference engine.');
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleSelectHistoryItem = (item) => {
    setSelectedImage(item.image);
    setPredictionResult(item.result);
    setActiveSampleId(null);
    setErrorMessage(null);
  };

  const handleClearHistory = () => {
    setHistory([]);
  };

  return (
    <div className="min-h-screen bg-slate-50 flex flex-col justify-between text-slate-900 antialiased selection:bg-amber-500/20 selection:text-amber-900">
      
      {/* Top Navbar */}
      <Navbar
        onOpenHistory={() => setIsHistoryOpen(true)}
        onOpenInfoModal={() => setIsInfoModalOpen(true)}
        onOpenLibrary={() => setIsLibraryOpen(true)}
        onOpenSettings={() => setIsSettingsOpen(true)}
        historyCount={history.length}
        isLiveApi={isLiveApi}
        activeModel={activeModel}
      />

      {/* Main Studio Container */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 w-full space-y-6 flex-1 py-4">
        
        {/* Hero Section */}
        <Hero />

        {/* Error Alert Notice (if any) */}
        {errorMessage && (
          <div className="bg-rose-50 border border-rose-200 rounded-2xl p-4 text-rose-900 flex items-start justify-between shadow-soft">
            <div className="flex items-start space-x-3">
              <div className="w-8 h-8 rounded-lg bg-rose-100 text-rose-700 flex items-center justify-center shrink-0">
                <ShieldAlert className="w-5 h-5" />
              </div>
              <div>
                <h4 className="font-bold text-rose-900 text-sm">Inference Pipeline Notice</h4>
                <p className="text-xs text-rose-700 mt-0.5 leading-relaxed">{errorMessage}</p>
              </div>
            </div>
            <button
              onClick={() => setErrorMessage(null)}
              className="text-rose-400 hover:text-rose-700 p-1.5 rounded-lg hover:bg-rose-100 transition-colors"
              aria-label="Dismiss error"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        )}

        {/* Modern 2-Column Responsive Studio Layout */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
          
          {/* Left Column (7 cols): Specimen Studio & Inscription Canvas */}
          <div className="lg:col-span-7 space-y-6">
            <UploadZone
              selectedImage={selectedImage}
              onImageSelect={handleImageUpload}
              onClearImage={handleClearImage}
              onAnalyze={handleAnalyze}
              isAnalyzing={isAnalyzing}
              filterMode={filterMode}
              onChangeFilterMode={setFilterMode}
              activeSampleId={activeSampleId}
              onSelectSample={handleSelectSample}
            />
          </div>

          {/* Right Column (5 cols): Real-Time Epigraphic Intelligence Dashboard */}
          <div className="lg:col-span-5 space-y-6">
            
            {/* State 1: Active Neural Inference in Progress */}
            {isAnalyzing && (
              <div className="clean-card p-6 bg-white shadow-card space-y-5 text-center relative overflow-hidden">
                <div className="absolute inset-x-0 top-0 h-1 bg-gradient-to-r from-amber-400 via-amber-600 to-indigo-600 animate-pulse" />
                
                <div className="w-16 h-16 rounded-2xl bg-amber-50 border border-amber-200/80 mx-auto flex items-center justify-center text-amber-600 shadow-soft relative">
                  <ScanLine className="w-8 h-8 animate-bounce" />
                </div>

                <div>
                  <span className="text-[11px] font-bold uppercase tracking-wider text-amber-700 bg-amber-50 px-2.5 py-1 rounded-full border border-amber-200">
                    DeepScript ViT-B/16
                  </span>
                  <h3 className="font-display font-extrabold text-lg text-slate-900 mt-2">
                    Analyzing Epigraphic Features
                  </h3>
                  <p className="text-xs text-slate-500 mt-1 max-w-xs mx-auto">
                    Evaluating stroke curvature, headmarks, writing direction, and Indic script prototypes...
                  </p>
                </div>

                <div className="space-y-2 text-left bg-slate-50 p-3.5 rounded-xl border border-slate-200/80 text-xs font-mono text-slate-600">
                  <div className="flex items-center gap-2 text-emerald-700">
                    <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
                    <span>Preprocessing & bicubic resize (224x224): OK</span>
                  </div>
                  <div className="flex items-center gap-2 text-amber-700">
                    <span className="w-1.5 h-1.5 rounded-full bg-amber-500 animate-pulse" />
                    <span>ViT-B/16 neural embedding & cosine projection...</span>
                  </div>
                  <div className="flex items-center gap-2 text-slate-400">
                    <span className="w-1.5 h-1.5 rounded-full bg-slate-300" />
                    <span>Paleographic comparative ranking</span>
                  </div>
                </div>
              </div>
            )}

            {/* State 2: Prediction Result Ready */}
            {!isAnalyzing && predictionResult && (
              <div id="results-view" className="space-y-6">
                <PredictionResult
                  result={predictionResult}
                  onOpenDetails={(scriptData) => setSelectedScriptForModal(scriptData)}
                />
              </div>
            )}

            {/* State 3: Ready / Idle Guide */}
            {!isAnalyzing && !predictionResult && (
              <div className="clean-card p-6 bg-white shadow-soft space-y-4">
                <div className="flex items-center gap-2">
                  <div className="w-8 h-8 rounded-lg bg-amber-50 text-amber-800 border border-amber-200 flex items-center justify-center">
                    <Compass className="w-4 h-4" />
                  </div>
                  <div>
                    <h3 className="font-display font-bold text-base text-slate-900">
                      Epigraphic Intelligence
                    </h3>
                    <p className="text-xs text-slate-500">
                      DeepScript ViT-B/16 (Primary) • Optional VLM
                    </p>
                  </div>
                </div>

                <div className="p-4 rounded-xl bg-slate-50 border border-slate-200/80 space-y-2.5 text-xs text-slate-600 leading-relaxed">
                  <div className="flex items-start gap-2">
                    <CheckCircle2 className="w-4 h-4 text-emerald-600 shrink-0 mt-0.5" />
                    <span>
                      <strong>Select any specimen</strong> on the left or upload an image to run the ViT-B/16 classifier.
                    </span>
                  </div>
                  <div className="flex items-start gap-2">
                    <CheckCircle2 className="w-4 h-4 text-emerald-600 shrink-0 mt-0.5" />
                    <span>
                      Primary 5 MVP Classes: <strong>Ashokan Brahmi, Gupta, Kharosthi, Grantha, Kadamba</strong>.
                    </span>
                  </div>
                  <div className="flex items-start gap-2">
                    <CheckCircle2 className="w-4 h-4 text-emerald-600 shrink-0 mt-0.5" />
                    <span>
                      Provides writing direction, headmark morphology, diagnostic glyph markers, and archaeological sites.
                    </span>
                  </div>
                </div>

                {selectedImage && (
                  <button
                    type="button"
                    onClick={handleAnalyze}
                    className="w-full btn-clean-primary py-2.5 text-xs font-bold flex items-center justify-center gap-2 shadow-sm"
                  >
                    <Sparkles className="w-4 h-4" />
                    <span>Identify Active Specimen Now</span>
                  </button>
                )}
              </div>
            )}

          </div>

        </div>

        {/* Full-Width Script Family Reference Strip */}
        <section className="clean-card p-5 sm:p-6 bg-white shadow-soft space-y-3">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
            <div>
              <h3 className="font-display font-bold text-sm text-slate-900 flex items-center gap-2">
                <Layers className="w-4 h-4 text-amber-600" />
                <span>Ancient Indian Script Lineage & Families</span>
              </h3>
              <p className="text-xs text-slate-500">
                Click any script family to inspect its historical archaeology and visual characteristics
              </p>
            </div>

            <button
              onClick={() => setIsLibraryOpen(true)}
              className="text-xs font-bold text-amber-800 hover:text-amber-900 inline-flex items-center gap-1 self-start sm:self-auto"
            >
              <span>Explore Complete Library</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </button>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-6 gap-2 pt-1">
            {Object.keys(ANCIENT_SCRIPTS).map((scriptName) => {
              const script = ANCIENT_SCRIPTS[scriptName];
              return (
                <button
                  key={scriptName}
                  onClick={() => setSelectedScriptForModal(script)}
                  className="text-left p-2.5 rounded-xl bg-slate-50 hover:bg-amber-50/60 border border-slate-200/80 hover:border-amber-300 transition-all group cursor-pointer"
                >
                  <h4 className="text-xs font-bold text-slate-800 group-hover:text-amber-900 truncate">
                    {script.name}
                  </h4>
                  <p className="text-[11px] text-slate-500 truncate mt-0.5">
                    {script.period ? script.period.split('–')[0] : 'Ancient'}
                  </p>
                </button>
              );
            })}
          </div>
        </section>

      </main>

      {/* Footer */}
      <Footer 
        onOpenInfoModal={() => setIsInfoModalOpen(true)}
        onOpenLibrary={() => setIsLibraryOpen(true)}
      />

      {/* Script Epigraphic Profile Modal */}
      <ScriptModal
        scriptData={selectedScriptForModal}
        isOpen={Boolean(selectedScriptForModal)}
        onClose={() => setSelectedScriptForModal(null)}
      />

      {/* Script Library Explorer Modal */}
      <ScriptLibraryModal
        isOpen={isLibraryOpen}
        onClose={() => setIsLibraryOpen(false)}
        onSelectScript={(script) => {
          setSelectedScriptForModal(script);
          setIsLibraryOpen(false);
        }}
      />

      {/* Hugging Face VLM Model Settings Modal */}
      <HfModelSettingsModal
        isOpen={isSettingsOpen}
        onClose={() => setIsSettingsOpen(false)}
        activeModel={activeModel}
        onConfigUpdated={(config) => {
          if (config.model) setActiveModel(config.model);
        }}
      />

      {/* Session Inscription History Drawer */}
      <HistoryDrawer
        isOpen={isHistoryOpen}
        onClose={() => setIsHistoryOpen(false)}
        history={history}
        onSelectHistoryItem={handleSelectHistoryItem}
        onClearHistory={handleClearHistory}
      />

      {/* Epigraphic Scope & Guidelines Modal */}
      {isInfoModalOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-900/50 backdrop-blur-xs animate-in fade-in duration-150">
          <div className="relative w-full max-w-lg max-h-[85vh] overflow-y-auto bg-white border border-slate-200 rounded-2xl p-6 sm:p-7 space-y-4 shadow-2xl">
            <button
              onClick={() => setIsInfoModalOpen(false)}
              className="absolute top-5 right-5 p-1.5 rounded-lg text-slate-400 hover:text-slate-800 hover:bg-slate-100 transition-colors"
              aria-label="Close modal"
            >
              <X className="w-4 h-4" />
            </button>

            <span className="text-[11px] uppercase tracking-wider font-bold text-amber-700 bg-amber-50 px-2.5 py-1 rounded-md border border-amber-200">
              Epigraphic Scope & Guidelines
            </span>

            <h3 className="font-display text-xl font-extrabold text-slate-900">
              DeepScript System Scope
            </h3>

            <div className="p-3.5 rounded-xl bg-amber-50/70 border border-amber-200 text-amber-950 text-xs leading-relaxed flex items-start gap-2.5">
              <ShieldAlert className="w-4 h-4 text-amber-600 shrink-0 mt-0.5" />
              <div>
                <strong>DeepScript ViT-B/16 Architecture:</strong> DeepScript utilizes a fine-tuned Vision Transformer (ViT-B/16) with embedding projection and cosine similarity head to perform <strong>Script Family Classification</strong> on whole ancient inscriptions, with optional VLM paleographic assistance.
                <p className="mt-1 text-[11px] text-amber-900/80">
                  Initial evaluation: 100% on a 6-image held-out test split (small MVP dataset of 49 whole inscriptions; ongoing research).
                </p>
              </div>
            </div>

            <div className="space-y-2 pt-1">
              <h4 className="text-xs font-bold text-slate-700 uppercase tracking-wider">
                Supported Ancient Indian Script Families
              </h4>
              <div className="grid grid-cols-2 gap-2 text-xs text-slate-700">
                {Object.keys(ANCIENT_SCRIPTS).map((scriptName) => (
                  <div key={scriptName} className="p-2.5 rounded-lg bg-slate-50 border border-slate-200 flex items-center gap-2">
                    <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600" />
                    <span className="font-medium">{scriptName}</span>
                  </div>
                ))}
              </div>
            </div>

            <div className="pt-3 flex justify-end">
              <button
                onClick={() => setIsInfoModalOpen(false)}
                className="btn-clean-primary text-xs px-5 py-2"
              >
                Understood
              </button>
            </div>
          </div>
        </div>
      )}

    </div>
  );
}

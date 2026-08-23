import React, { useState, useEffect } from 'react';
import { Navbar } from './components/Navbar';
import { Hero } from './components/Hero';
import { SamplePicker } from './components/SamplePicker';
import { UploadZone } from './components/UploadZone';
import { PredictionResult } from './components/PredictionResult';
import { ScriptModal } from './components/ScriptModal';
import { HistoryDrawer } from './components/HistoryDrawer';
import { Footer } from './components/Footer';
import { predictScript, checkBackendStatus } from './services/api';
import { ANCIENT_SCRIPTS } from './data/scriptsData';
import { SAMPLE_INSCRIPTIONS } from './data/sampleImages';
import { X, CheckCircle2, ShieldAlert } from 'lucide-react';

export default function App() {
  const [selectedImage, setSelectedImage] = useState(null);
  const [activeSampleId, setActiveSampleId] = useState(null);
  const [filterMode, setFilterMode] = useState('normal'); // 'normal' | 'high-contrast' | 'invert'
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [predictionResult, setPredictionResult] = useState(null);
  const [history, setHistory] = useState([]);
  
  const [isHistoryOpen, setIsHistoryOpen] = useState(false);
  const [selectedScriptForModal, setSelectedScriptForModal] = useState(null);
  const [isInfoModalOpen, setIsInfoModalOpen] = useState(false);
  
  const [isLiveApi, setIsLiveApi] = useState(false);

  useEffect(() => {
    async function initCheck() {
      const isOnline = await checkBackendStatus();
      setIsLiveApi(isOnline);
    }
    initCheck();
  }, []);

  // Pre-select first Indian script sample
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
      size: 'Sample Specimen',
      source: 'sample',
      expectedScript: sample.expectedScript,
      medium: sample.medium
    });
    setPredictionResult(null);
  };

  const handleImageUpload = (imageData) => {
    setActiveSampleId(null);
    setSelectedImage(imageData);
    setPredictionResult(null);
  };

  const handleClearImage = () => {
    setSelectedImage(null);
    setActiveSampleId(null);
    setPredictionResult(null);
  };

  const handleAnalyze = async () => {
    if (!selectedImage) return;

    setIsAnalyzing(true);
    try {
      const payload = selectedImage.file || selectedImage.url;
      const metadata = {
        expectedScript: selectedImage.expectedScript,
        name: selectedImage.name
      };

      const result = await predictScript(payload, metadata);
      setPredictionResult(result);

      const historyItem = {
        id: `scan-${Date.now()}`,
        image: {
          url: selectedImage.url,
          name: selectedImage.name || 'Custom Inscription'
        },
        result,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      };
      setHistory((prev) => [historyItem, ...prev]);

    } catch (err) {
      console.error('Analysis failed:', err);
      alert('An error occurred while identifying the script.');
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleSelectHistoryItem = (item) => {
    setSelectedImage(item.image);
    setPredictionResult(item.result);
    setActiveSampleId(null);
  };

  const handleClearHistory = () => {
    setHistory([]);
  };

  return (
    <div className="min-h-screen bg-zinc-950 flex flex-col justify-between text-zinc-100 antialiased">
      
      {/* Navbar */}
      <Navbar
        onOpenHistory={() => setIsHistoryOpen(true)}
        onOpenInfoModal={() => setIsInfoModalOpen(true)}
        historyCount={history.length}
      />

      {/* Main Container */}
      <main className="max-w-4xl mx-auto px-4 sm:px-6 w-full space-y-6 flex-1">
        
        <Hero />

        {/* Sample Indian Inscriptions */}
        <section>
          <SamplePicker
            onSelectSample={handleSelectSample}
            activeSampleId={activeSampleId}
            isAnalyzing={isAnalyzing}
          />
        </section>

        {/* Upload & Inspect */}
        <section>
          <UploadZone
            selectedImage={selectedImage}
            onImageSelect={handleImageUpload}
            onClearImage={handleClearImage}
            onAnalyze={handleAnalyze}
            isAnalyzing={isAnalyzing}
            filterMode={filterMode}
            onChangeFilterMode={setFilterMode}
          />
        </section>

        {/* Prediction Results */}
        {predictionResult && (
          <section id="results-view">
            <PredictionResult
              result={predictionResult}
              onOpenDetails={(scriptData) => setSelectedScriptForModal(scriptData)}
            />
          </section>
        )}

      </main>

      {/* Footer */}
      <Footer onOpenInfoModal={() => setIsInfoModalOpen(true)} />

      {/* Script Epigraphic Profile Modal */}
      <ScriptModal
        scriptData={selectedScriptForModal}
        isOpen={Boolean(selectedScriptForModal)}
        onClose={() => setSelectedScriptForModal(null)}
      />

      {/* Session History Drawer */}
      <HistoryDrawer
        isOpen={isHistoryOpen}
        onClose={() => setIsHistoryOpen(false)}
        history={history}
        onSelectHistoryItem={handleSelectHistoryItem}
        onClearHistory={handleClearHistory}
      />

      {/* Scope & Epigraphy Info Modal */}
      {isInfoModalOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/70 backdrop-blur-xs">
          <div className="relative w-full max-w-lg max-h-[85vh] overflow-y-auto bg-zinc-925 border border-white/10 rounded-2xl p-6 space-y-4">
            <button
              onClick={() => setIsInfoModalOpen(false)}
              className="absolute top-5 right-5 p-1.5 rounded-lg text-zinc-400 hover:text-white hover:bg-zinc-800 transition-colors"
            >
              <X className="w-4 h-4" />
            </button>

            <span className="text-[11px] uppercase tracking-wider font-semibold text-amber-400">
              Epigraphic Scope & Guidelines
            </span>

            <h3 className="font-display text-xl font-bold text-zinc-100">
              DeepScript System Scope
            </h3>

            <div className="p-3 rounded-xl bg-zinc-900 border border-white/5 text-zinc-300 text-xs leading-relaxed flex items-start gap-2.5">
              <ShieldAlert className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
              <div>
                <strong>Strictly Ancient Indian Scripts:</strong> DeepScript is designed exclusively for <strong>Script Identification</strong> (classifying which historical Indian script family appears in an inscription image). It does not perform character-level OCR, phonetic transliteration, or linguistic translation.
              </div>
            </div>

            <div className="space-y-2 pt-1">
              <h4 className="text-xs font-semibold text-zinc-300 uppercase tracking-wider">
                Supported Ancient Indian Scripts
              </h4>
              <div className="grid grid-cols-2 gap-2 text-xs text-zinc-300">
                {Object.keys(ANCIENT_SCRIPTS).map((scriptName) => (
                  <div key={scriptName} className="p-2 rounded-lg bg-zinc-900/60 border border-white/5 flex items-center gap-2">
                    <CheckCircle2 className="w-3.5 h-3.5 text-amber-400" />
                    <span>{scriptName}</span>
                  </div>
                ))}
              </div>
            </div>

            <div className="pt-3 flex justify-end">
              <button
                onClick={() => setIsInfoModalOpen(false)}
                className="btn-clean-primary text-xs px-4 py-1.5"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}

    </div>
  );
}

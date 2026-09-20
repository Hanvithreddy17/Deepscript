import React, { useState, useRef } from 'react';
import { 
  UploadCloud, 
  X, 
  Sparkles, 
  RefreshCw, 
  SlidersHorizontal, 
  Image as ImageIcon,
  Layers,
  ZoomIn,
  FileCheck2
} from 'lucide-react';
import { SamplePicker } from './SamplePicker';

export function UploadZone({
  selectedImage,
  onImageSelect,
  onClearImage,
  onAnalyze,
  isAnalyzing,
  filterMode,
  onChangeFilterMode,
  activeSampleId,
  onSelectSample
}) {
  const [activeTab, setActiveTab] = useState(activeSampleId ? 'samples' : 'upload');
  const [isDragOver, setIsDragOver] = useState(false);
  const fileInputRef = useRef(null);

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setIsDragOver(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragOver(false);
    const files = e.dataTransfer.files;
    if (files && files.length > 0) {
      handleFile(files[0]);
    }
  };

  const handleFileInputChange = (e) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      handleFile(files[0]);
    }
  };

  const handleFile = (file) => {
    if (!file.type.startsWith('image/')) {
      alert('Please upload a valid image file (JPEG, PNG, WebP, SVG)');
      return;
    }
    const reader = new FileReader();
    reader.onload = (event) => {
      onImageSelect({
        file: file,
        url: event.target.result,
        name: file.name,
        size: `${(file.size / 1024).toFixed(1)} KB`,
        source: 'upload'
      });
      setActiveTab('upload');
    };
    reader.readAsDataURL(file);
  };

  const getFilterClass = () => {
    switch (filterMode) {
      case 'high-contrast':
        return 'filter-high-contrast';
      case 'invert':
        return 'filter-invert';
      case 'monochrome':
        return 'filter-monochrome';
      default:
        return '';
    }
  };

  return (
    <div className="w-full clean-card p-5 sm:p-6 space-y-5 bg-white shadow-soft">
      
      {/* Studio Header & Mode Tabs */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-4 border-b border-slate-200/80">
        <div>
          <h2 className="font-display font-bold text-base sm:text-lg text-slate-900 flex items-center gap-2">
            <span>Epigraphic Specimen Studio</span>
          </h2>
          <p className="text-xs text-slate-500 mt-0.5">
            Select an archaeological specimen or upload a photo / stone rubbing
          </p>
        </div>

        {/* Studio Tabs */}
        <div className="inline-flex rounded-xl bg-slate-100 p-1 border border-slate-200/80 self-start sm:self-auto">
          <button
            type="button"
            onClick={() => setActiveTab('samples')}
            className={`px-3 py-1.5 text-xs font-semibold rounded-lg transition-all flex items-center gap-1.5 ${
              activeTab === 'samples'
                ? 'bg-white text-slate-900 shadow-xs'
                : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            <Layers className="w-3.5 h-3.5" />
            <span>Preset Specimens</span>
          </button>
          <button
            type="button"
            onClick={() => setActiveTab('upload')}
            className={`px-3 py-1.5 text-xs font-semibold rounded-lg transition-all flex items-center gap-1.5 ${
              activeTab === 'upload'
                ? 'bg-white text-slate-900 shadow-xs'
                : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            <UploadCloud className="w-3.5 h-3.5" />
            <span>Upload Image</span>
          </button>
        </div>
      </div>

      {/* Tab 1: Presets Picker */}
      {activeTab === 'samples' && (
        <div className="animate-in fade-in duration-150">
          <SamplePicker
            onSelectSample={(sample) => {
              onSelectSample(sample);
            }}
            activeSampleId={activeSampleId}
            isAnalyzing={isAnalyzing}
          />
        </div>
      )}

      {/* Tab 2: Custom Upload Dropzone */}
      {activeTab === 'upload' && !selectedImage && (
        <div
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
          className={`flex flex-col items-center justify-center p-8 sm:p-10 rounded-2xl border-2 border-dashed transition-all cursor-pointer ${
            isDragOver
              ? 'border-amber-500 bg-amber-50/50'
              : 'border-slate-300/90 bg-slate-50/70 hover:bg-slate-100/70 hover:border-slate-400'
          }`}
        >
          <input
            type="file"
            ref={fileInputRef}
            onChange={handleFileInputChange}
            accept="image/png, image/jpeg, image/webp, image/svg+xml"
            className="hidden"
          />

          <div className="w-12 h-12 rounded-xl bg-amber-100 text-amber-800 flex items-center justify-center mb-3 shadow-2xs border border-amber-200">
            <UploadCloud className="w-6 h-6" />
          </div>

          <p className="text-sm font-semibold text-slate-800 text-center mb-1">
            Click to upload or drag & drop stone rubbing / photo
          </p>
          <p className="text-xs text-slate-500 text-center max-w-sm">
            Supports high-resolution PNG, JPG, WebP, or vector SVG. Single glyph or cropped line recommended.
          </p>
        </div>
      )}

      {/* Specimen Live Preview & Image Filters */}
      {selectedImage && (
        <div className="space-y-4 pt-1">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <span className="text-xs font-bold uppercase tracking-wider text-slate-700">
                Active Specimen View
              </span>
              <span className="text-[11px] font-mono text-slate-500 bg-slate-100 px-2 py-0.5 rounded border border-slate-200">
                {selectedImage.name || 'Sample Specimen'}
              </span>
            </div>

            <button
              onClick={onClearImage}
              disabled={isAnalyzing}
              className="text-xs text-slate-500 hover:text-rose-600 inline-flex items-center gap-1 py-1 px-2 rounded-md hover:bg-rose-50 transition-colors border border-transparent hover:border-rose-200"
            >
              <X className="w-3.5 h-3.5" />
              <span>Change Image</span>
            </button>
          </div>

          {/* Image Canvas Frame */}
          <div className="relative w-full h-64 sm:h-72 rounded-2xl overflow-hidden bg-slate-950 border border-slate-200 shadow-inner flex items-center justify-center group">
            <img
              src={selectedImage.url}
              alt="Inscription Preview"
              className={`max-h-full max-w-full object-contain transition-all duration-200 ${getFilterClass()}`}
            />

            {/* Specimen Metadata Pill */}
            <div className="absolute bottom-3 left-3 bg-slate-900/90 backdrop-blur-md px-3 py-1.5 rounded-lg border border-white/10 text-xs text-slate-200 font-medium flex items-center gap-2 shadow-md">
              <span className="w-2 h-2 rounded-full bg-amber-400 animate-pulse" />
              <span className="truncate max-w-[220px]">
                {selectedImage.expectedScript || selectedImage.name || 'Historical Specimen'}
              </span>
              {selectedImage.size && (
                <span className="text-slate-400 font-mono text-[10px] pl-1 border-l border-white/20">
                  {selectedImage.size}
                </span>
              )}
            </div>
          </div>

          {/* Inscription Filters & Analysis Trigger */}
          <div className="flex flex-col sm:flex-row items-stretch sm:items-center justify-between gap-3 pt-2">
            
            {/* Filter Pills */}
            <div className="flex items-center gap-1.5 flex-wrap">
              <span className="text-xs font-semibold text-slate-600 mr-1 flex items-center gap-1">
                <SlidersHorizontal className="w-3.5 h-3.5 text-slate-400" />
                <span>Filter:</span>
              </span>

              <div className="inline-flex rounded-lg bg-slate-100 p-0.5 border border-slate-200">
                <button
                  type="button"
                  onClick={() => onChangeFilterMode('normal')}
                  className={`px-2.5 py-1 text-xs rounded-md transition-all font-medium ${
                    filterMode === 'normal' 
                      ? 'bg-white text-slate-900 shadow-2xs font-semibold' 
                      : 'text-slate-600 hover:text-slate-900'
                  }`}
                >
                  Natural
                </button>
                <button
                  type="button"
                  onClick={() => onChangeFilterMode('high-contrast')}
                  className={`px-2.5 py-1 text-xs rounded-md transition-all font-medium ${
                    filterMode === 'high-contrast' 
                      ? 'bg-white text-slate-900 shadow-2xs font-semibold' 
                      : 'text-slate-600 hover:text-slate-900'
                  }`}
                >
                  High Contrast
                </button>
                <button
                  type="button"
                  onClick={() => onChangeFilterMode('invert')}
                  className={`px-2.5 py-1 text-xs rounded-md transition-all font-medium ${
                    filterMode === 'invert' 
                      ? 'bg-white text-slate-900 shadow-2xs font-semibold' 
                      : 'text-slate-600 hover:text-slate-900'
                  }`}
                >
                  Estampage Invert
                </button>
                <button
                  type="button"
                  onClick={() => onChangeFilterMode('monochrome')}
                  className={`px-2.5 py-1 text-xs rounded-md transition-all font-medium ${
                    filterMode === 'monochrome' 
                      ? 'bg-white text-slate-900 shadow-2xs font-semibold' 
                      : 'text-slate-600 hover:text-slate-900'
                  }`}
                >
                  Ink Rubbing
                </button>
              </div>
            </div>

            {/* Main Action Button */}
            <button
              type="button"
              onClick={onAnalyze}
              disabled={isAnalyzing}
              className="btn-clean-primary w-full sm:w-auto px-6 py-2.5 text-sm"
            >
              {isAnalyzing ? (
                <>
                  <RefreshCw className="w-4 h-4 animate-spin text-white" />
                  <span>Classifying Script...</span>
                </>
              ) : (
                <>
                  <Sparkles className="w-4 h-4 text-white" />
                  <span>Identify Script</span>
                </>
              )}
            </button>

          </div>
        </div>
      )}

    </div>
  );
}

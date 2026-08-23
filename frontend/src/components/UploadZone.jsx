import React, { useState, useRef } from 'react';
import { UploadCloud, X, Sliders, RefreshCw, Sparkles, Image as ImageIcon } from 'lucide-react';

export function UploadZone({
  selectedImage,
  onImageSelect,
  onClearImage,
  onAnalyze,
  isAnalyzing,
  filterMode,
  onChangeFilterMode
}) {
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
      alert('Please upload a valid image file (JPEG, PNG, WebP)');
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
    };
    reader.readAsDataURL(file);
  };

  const getFilterClass = () => {
    switch (filterMode) {
      case 'high-contrast':
        return 'filter-high-contrast';
      case 'invert':
        return 'filter-invert';
      default:
        return '';
    }
  };

  return (
    <div className="w-full clean-card p-5 sm:p-6 relative">
      
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div>
          <h2 className="font-display font-semibold text-sm sm:text-base text-zinc-200">
            Inscription Image
          </h2>
          <p className="text-xs text-zinc-500">
            Upload stone rubbing, copper plate scan, or photo
          </p>
        </div>

        {selectedImage && (
          <button
            onClick={onClearImage}
            disabled={isAnalyzing}
            className="text-xs text-zinc-400 hover:text-zinc-200 inline-flex items-center gap-1 py-1 px-2 rounded-md hover:bg-zinc-800 transition-colors"
          >
            <X className="w-3.5 h-3.5" />
            Clear
          </button>
        )}
      </div>

      {!selectedImage ? (
        /* Minimal Empty Drop Zone */
        <div
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
          className={`flex flex-col items-center justify-center p-8 sm:p-12 rounded-xl border border-dashed transition-all cursor-pointer ${
            isDragOver
              ? 'border-amber-500/60 bg-amber-500/5'
              : 'border-white/10 bg-zinc-950/40 hover:border-white/20 hover:bg-zinc-950/60'
          }`}
        >
          <input
            type="file"
            ref={fileInputRef}
            onChange={handleFileInputChange}
            accept="image/png, image/jpeg, image/webp"
            className="hidden"
          />

          <div className="w-10 h-10 rounded-lg bg-zinc-900 flex items-center justify-center border border-white/5 mb-3 text-zinc-400">
            <UploadCloud className="w-5 h-5" />
          </div>

          <p className="text-xs sm:text-sm font-medium text-zinc-300 mb-1 text-center">
            Click to upload or drag and drop inscription image
          </p>
          <p className="text-[11px] text-zinc-500 text-center">
            PNG, JPG, or WebP up to 15MB
          </p>
        </div>
      ) : (
        /* Image Preview & Clean Controls */
        <div className="space-y-4">
          
          {/* Inscription Preview */}
          <div className="relative w-full h-64 sm:h-72 rounded-xl overflow-hidden bg-zinc-950 border border-white/[0.06] flex items-center justify-center">
            <img
              src={selectedImage.url}
              alt="Inscription Preview"
              className={`max-h-full max-w-full object-contain transition-all duration-200 ${getFilterClass()}`}
            />

            {/* Inscription Tag */}
            <div className="absolute bottom-2.5 left-2.5 bg-zinc-900/90 backdrop-blur-sm px-2.5 py-1 rounded-md border border-white/5 text-[11px] text-zinc-400 font-mono flex items-center gap-1.5">
              <span className="w-1.5 h-1.5 rounded-full bg-amber-400" />
              <span className="truncate max-w-[200px]">{selectedImage.name || 'Sample Specimen'}</span>
            </div>
          </div>

          {/* Clean Controls Bar */}
          <div className="flex flex-wrap items-center justify-between gap-3 pt-1">
            
            {/* Filter Toggle */}
            <div className="flex items-center gap-2 text-xs text-zinc-400">
              <span className="text-zinc-500">Filter:</span>
              <div className="inline-flex rounded-lg bg-zinc-900 p-0.5 border border-white/5">
                <button
                  onClick={() => onChangeFilterMode('normal')}
                  className={`px-2.5 py-1 text-[11px] rounded-md transition-all ${
                    filterMode === 'normal' ? 'bg-zinc-800 text-zinc-200 font-medium' : 'text-zinc-400 hover:text-zinc-300'
                  }`}
                >
                  Normal
                </button>
                <button
                  onClick={() => onChangeFilterMode('high-contrast')}
                  className={`px-2.5 py-1 text-[11px] rounded-md transition-all ${
                    filterMode === 'high-contrast' ? 'bg-zinc-800 text-zinc-200 font-medium' : 'text-zinc-400 hover:text-zinc-300'
                  }`}
                >
                  High Contrast
                </button>
                <button
                  onClick={() => onChangeFilterMode('invert')}
                  className={`px-2.5 py-1 text-[11px] rounded-md transition-all ${
                    filterMode === 'invert' ? 'bg-zinc-800 text-zinc-200 font-medium' : 'text-zinc-400 hover:text-zinc-300'
                  }`}
                >
                  Estampage / Invert
                </button>
              </div>
            </div>

            {/* Action Button */}
            <button
              onClick={onAnalyze}
              disabled={isAnalyzing}
              className="btn-clean-primary w-full sm:w-auto"
            >
              {isAnalyzing ? (
                <>
                  <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                  <span>Identifying Script...</span>
                </>
              ) : (
                <>
                  <Sparkles className="w-3.5 h-3.5" />
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

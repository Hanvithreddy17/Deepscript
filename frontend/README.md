# DeepScript Frontend Application

Client-side user interface application for **DeepScript**, an AI-powered ancient Indian script identification system.

---

## Overview

The DeepScript frontend is built with **React**, **Vite**, and **Tailwind CSS**, featuring an epigraphic dark aesthetic tailored for historians, epigraphists, and researchers.

### Key Capabilities

- **Curated Epigraphic Specimen Showcase:** Built-in interactive specimens for Ashokan Brahmi, Tamil-Brahmi, Kharosthi, Grantha, Gupta Script, and Kadamba inscriptions for 1-click testing.
- **High-Resolution Inscription Uploader:** Drag-and-drop file uploader with epigraphic contrast enhancement modes (*High Contrast*, *Estampage Inversion*).
- **Interactive Script Classification View:**
  - Identified script / character name, phonetic classification, and historical period.
  - Real-time confidence percentage gauge.
  - Top-5 candidate script probability distribution.
  - Execution latency and compute device metrics.
- **Deep Epigraphic & Phonetic Dossiers:** Dedicated 62-class character and script modal dossiers detailing character phonetics, transliterations, visual recognition clues, lineages, and exemplar sites.
- **Session History Drawer:** Re-inspect and compare previous scans with defensive null safety.
- **Live ViT Model Connection Badge:** Automatic `/health` checking with real-time status pill (*ViT Model Live* vs *Simulation Mode*).
- **Seamless Vite Proxy:** Automatic reverse proxy routing `/predict`, `/health`, and `/classes` to `http://localhost:8000`.

---

## Technology Stack

- **Framework:** React 18
- **Build Tool:** Vite 5
- **Icons:** Lucide-React
- **Styling:** Tailwind CSS + Custom Epigraphic Design System tokens
- **Typography:** *Cinzel* (Google Fonts) & *Plus Jakarta Sans*

---

## Getting Started

### 1. Install Dependencies

```bash
cd frontend
npm install
```

### 2. Start Development Server

```bash
npm run dev
```

The application will start at `http://localhost:3000`.

### 3. Build for Production

```bash
npm run build
```

# DeepScript Frontend Application

Client-side user interface application for **DeepScript**, an AI-powered ancient Indian script identification system.

---

## Overview

The DeepScript frontend is built with **React**, **Vite**, and **Tailwind CSS**, featuring an epigraphic dark aesthetic tailored for historians, epigraphists, and researchers.

### Key Capabilities

- **Curated Epigraphic Specimen Showcase:** Built-in interactive specimens for Ashokan Brahmi, Tamil-Brahmi, Kharosthi, Grantha, Gupta Script, and Kadamba inscriptions for 1-click testing.
- **High-Resolution Inscription Uploader:** Drag-and-drop file uploader with epigraphic contrast enhancement modes (*High Contrast*, *Estampage Inversion*).
- **Interactive Script Classification View:**
  - Identified script name and historical period.
  - Confidence percentage gauge.
  - Top-4 candidate script probability breakdown.
  - Latency and feature extraction metrics.
- **Deep Epigraphic Knowledge Profiles:** Dedicated paleographic dossiers detailing script origins, geographic regions, genealogical lineage trees, and notable archaeological sites.
- **Session History Drawer:** Re-inspect and compare previous scans during a session.
- **Dual Live & Simulated API Modes:** Seamlessly toggles between local simulated ViT feature extraction and live FastAPI inference at `POST /predict`.

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

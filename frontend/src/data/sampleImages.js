/**
 * DeepScript — Curated Indian Epigraphic Specimens
 * Clean, high-fidelity procedural SVGs representing ancient Indian stone inscriptions,
 * cavern walls, copper charters, and birch bark.
 */

function svgToDataUrl(svgString) {
  return `data:image/svg+xml;charset=utf-8,${encodeURIComponent(svgString.trim())}`;
}

const ashokanBrahmiSvg = `
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 380" width="100%" height="100%">
  <rect width="600" height="380" fill="#141418"/>
  <rect width="600" height="380" fill="none" stroke="#27272a" stroke-width="1"/>
  
  <!-- Subtle stone background texture lines -->
  <line x1="40" y1="120" x2="560" y2="120" stroke="#1f1f24" stroke-width="1" stroke-dasharray="8 6"/>
  <line x1="40" y1="240" x2="560" y2="240" stroke="#1f1f24" stroke-width="1" stroke-dasharray="8 6"/>

  <!-- Ashokan Brahmi Glyphs (Crisp geometric chisel lines) -->
  <g fill="none" stroke="#f4f4f5" stroke-width="4.5" stroke-linecap="round" stroke-linejoin="round" opacity="0.92">
    <!-- Ka (Cross) -->
    <path d="M 65 65 L 65 115 M 40 90 L 90 90"/>
    <!-- Ma (Circle over base) -->
    <circle cx="140" cy="78" r="14"/>
    <path d="M 140 92 L 140 115 M 125 115 L 155 115"/>
    <!-- Ya (Tuning fork) -->
    <path d="M 200 65 L 200 95 Q 200 115 215 115 Q 230 115 230 95 L 230 65 M 215 95 L 215 115"/>
    <!-- Ra (Vertical line) -->
    <path d="M 275 65 L 275 115"/>
    <!-- Da (Arc with tails) -->
    <path d="M 315 70 Q 345 90 315 110 M 315 70 L 315 65 M 315 110 L 315 115"/>
    <!-- Sa (Hooked stem) -->
    <path d="M 380 80 Q 380 65 400 65 Q 420 65 420 90 L 420 115 M 395 90 L 420 90"/>
    <!-- Ta (Semi-circle) -->
    <path d="M 470 65 Q 495 90 470 115"/>
    <!-- Pa (Hook) -->
    <path d="M 535 65 L 535 115 L 560 115 L 560 80"/>
    
    <!-- Line 2 -->
    <!-- Ga (Inverted V) -->
    <path d="M 50 200 L 75 155 L 100 200"/>
    <!-- Ja (E shape) -->
    <path d="M 160 155 L 135 155 L 135 178 L 160 178 L 135 178 L 135 200 L 160 200"/>
    <!-- Dha (D shape) -->
    <path d="M 205 155 L 205 200 Q 235 178 205 155"/>
    <!-- Na (Inverted T) -->
    <path d="M 280 155 L 280 200 M 265 155 L 295 155"/>
    <!-- Ba (Square) -->
    <rect x="345" y="155" width="40" height="42" fill="none"/>
    <!-- La (Hook) -->
    <path d="M 435 155 L 435 185 Q 435 200 450 200 L 465 200"/>
    <!-- Cha (Bisected circle) -->
    <circle cx="530" cy="178" r="18"/>
    <path d="M 530 155 L 530 200"/>

    <!-- Line 3 -->
    <path d="M 55 260 L 95 260 M 75 260 L 75 305"/>
    <path d="M 130 260 Q 160 280 130 305"/>
    <path d="M 195 260 L 195 305 L 225 305"/>
    <circle cx="280" cy="282" r="16"/>
    <path d="M 335 260 L 360 305 L 385 260"/>
    <path d="M 430 270 L 470 270 M 450 260 L 450 305"/>
    <path d="M 515 265 Q 545 265 545 290 Q 545 305 520 305"/>
  </g>
</svg>
`;

const tamilBrahmiSvg = `
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 380" width="100%" height="100%">
  <rect width="600" height="380" fill="#141418"/>
  <rect width="600" height="380" fill="none" stroke="#27272a" stroke-width="1"/>
  
  <line x1="40" y1="180" x2="560" y2="180" stroke="#1f1f24" stroke-width="1" stroke-dasharray="8 6"/>

  <!-- Tamil-Brahmi Cavern Bed Inscription Glyphs -->
  <g fill="none" stroke="#e4e4e7" stroke-width="4.5" stroke-linecap="round" stroke-linejoin="round" opacity="0.9">
    <!-- Ka with dot (pulli) -->
    <path d="M 60 100 L 60 150 M 40 125 L 80 125"/>
    <circle cx="85" cy="100" r="3" fill="#e4e4e7"/>
    
    <!-- Dravidian 'Zha' (ழ) -->
    <path d="M 130 100 L 130 150 Q 150 150 160 130 Q 170 110 150 110 L 130 110"/>
    
    <!-- Dravidian 'Ra' (ற) -->
    <path d="M 210 100 Q 235 125 210 150 Q 185 125 210 100"/>
    
    <!-- Dravidian 'La' (ள) -->
    <path d="M 270 100 L 270 135 Q 270 150 290 150 Q 310 150 310 130 L 310 100"/>
    
    <!-- Dravidian 'Na' (ன) -->
    <path d="M 355 150 L 355 105 Q 370 105 380 120 Q 390 105 405 105 L 405 150"/>
    
    <!-- Ma -->
    <circle cx="465" cy="115" r="14"/>
    <path d="M 465 129 L 465 150 M 450 150 L 480 150"/>
    
    <!-- Ya -->
    <path d="M 525 100 L 525 130 Q 525 150 540 150 Q 555 150 555 130 L 555 100 M 540 130 L 540 150"/>

    <!-- Line 2: Cave bed donative text -->
    <path d="M 80 230 L 80 275 M 65 250 L 95 250"/>
    <circle cx="150" cy="250" r="18"/>
    <path d="M 205 230 L 235 275 L 265 230"/>
    <path d="M 320 230 Q 345 255 320 275"/>
    <path d="M 380 230 L 380 275 L 410 275"/>
    <path d="M 460 230 L 490 230 M 475 230 L 475 275"/>
  </g>
</svg>
`;

const kharosthiSvg = `
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 380" width="100%" height="100%">
  <rect width="600" height="380" fill="#141418"/>
  <rect width="600" height="380" fill="none" stroke="#27272a" stroke-width="1"/>

  <!-- Kharosthi Cursive Glyphs (Right to Left) -->
  <g fill="none" stroke="#e4e4e7" stroke-width="4.5" stroke-linecap="round" stroke-linejoin="round" opacity="0.9">
    <!-- Right to left flow -->
    <path d="M 540 90 Q 520 100 520 130 L 520 160 M 535 105 L 505 120"/>
    <path d="M 460 85 L 460 155 Q 440 155 435 135 L 435 110"/>
    <path d="M 385 90 Q 370 90 370 115 L 370 160 M 390 110 L 350 130"/>
    <path d="M 305 85 L 285 155 M 315 110 L 275 110"/>
    <path d="M 230 95 Q 210 90 205 125 L 205 160 Q 190 155 185 135"/>
    <path d="M 135 90 L 135 155 M 150 100 L 115 125"/>
    <path d="M 65 85 Q 50 110 50 155"/>

    <!-- Line 2 -->
    <path d="M 535 220 Q 510 230 510 280 M 530 240 L 490 250"/>
    <path d="M 440 215 L 440 285 Q 420 285 410 260"/>
    <path d="M 360 220 Q 340 220 340 250 L 340 285"/>
    <path d="M 275 220 L 255 285 M 285 245 L 245 245"/>
    <path d="M 195 220 Q 170 245 170 285 M 190 235 L 150 250"/>
    <path d="M 95 215 L 95 285 Q 75 270 70 240"/>
  </g>
</svg>
`;

const granthaSvg = `
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 380" width="100%" height="100%">
  <rect width="600" height="380" fill="#141418"/>
  <rect width="600" height="380" fill="none" stroke="#27272a" stroke-width="1"/>

  <!-- Ornate Rounded Grantha Script Glyphs -->
  <g fill="none" stroke="#e4e4e7" stroke-width="4.5" stroke-linecap="round" stroke-linejoin="round" opacity="0.92">
    <!-- Glyph 1: Swirling conjunct -->
    <path d="M 75 80 Q 95 50 120 70 Q 145 90 120 110 Q 95 130 120 150 Q 145 160 160 140"/>
    <!-- Glyph 2: Grantha 'Ka' with loops -->
    <path d="M 195 70 Q 215 50 235 70 L 235 140 Q 235 155 215 155 Q 195 155 195 130 Q 195 105 225 105 L 255 105"/>
    <!-- Glyph 3: Grantha 'Ma' -->
    <path d="M 295 145 L 295 75 Q 320 60 340 85 Q 360 110 335 130 L 295 130 M 295 150 L 360 150"/>
    <!-- Glyph 4: Grantha 'Ta' -->
    <path d="M 395 70 Q 415 55 440 55 Q 465 55 465 85 Q 465 115 440 125 Q 420 135 425 155 Q 435 165 460 160"/>
    <!-- Glyph 5: Conjunct ligature -->
    <path d="M 500 60 L 540 60 M 520 60 L 520 115 Q 520 135 505 145 Q 490 155 510 165 L 540 165"/>

    <!-- Line 2 -->
    <path d="M 80 230 Q 105 205 135 225 Q 160 245 135 270 Q 110 295 140 305"/>
    <path d="M 190 225 L 190 300 Q 220 300 230 270 Q 240 240 215 240 L 190 240"/>
    <path d="M 280 220 Q 305 200 330 220 L 330 290 Q 330 310 355 310"/>
    <path d="M 395 220 L 435 220 M 415 220 L 415 280 Q 415 305 445 305"/>
    <path d="M 490 220 Q 520 220 520 255 Q 520 290 490 290 Q 470 290 475 310 L 525 310"/>
  </g>
</svg>
`;

const guptaScriptSvg = `
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 380" width="100%" height="100%">
  <rect width="600" height="380" fill="#141418"/>
  <rect width="600" height="380" fill="none" stroke="#27272a" stroke-width="1"/>

  <!-- Gupta Script Glyphs with Triangular Wedge Headmarks -->
  <g fill="none" stroke="#e4e4e7" stroke-width="4.5" stroke-linecap="round" stroke-linejoin="round" opacity="0.9">
    <!-- Wedge mark + stem -->
    <path d="M 75 75 L 95 75 L 85 90 Z" fill="#e4e4e7"/>
    <path d="M 85 90 L 85 155 M 55 120 Q 85 115 115 120"/>

    <path d="M 165 75 L 185 75 L 175 90 Z" fill="#e4e4e7"/>
    <path d="M 175 90 L 175 155 M 150 155 Q 175 125 200 155"/>

    <path d="M 255 75 L 275 75 L 265 90 Z" fill="#e4e4e7"/>
    <path d="M 265 90 L 265 155 M 235 125 Q 250 155 265 155 Q 280 155 295 125"/>

    <path d="M 345 75 L 365 75 L 355 90 Z" fill="#e4e4e7"/>
    <path d="M 355 90 L 355 155"/>

    <path d="M 435 75 L 455 75 L 445 90 Z" fill="#e4e4e7"/>
    <path d="M 445 90 L 445 155 M 420 115 Q 445 115 470 140"/>

    <!-- Line 2 -->
    <path d="M 75 220 L 95 220 L 85 235 Z" fill="#e4e4e7"/>
    <path d="M 85 235 L 85 290 M 60 260 L 110 260"/>

    <path d="M 175 220 L 195 220 L 185 235 Z" fill="#e4e4e7"/>
    <path d="M 185 235 Q 215 260 185 290"/>

    <path d="M 275 220 L 295 220 L 285 235 Z" fill="#e4e4e7"/>
    <path d="M 285 235 L 285 290 L 315 290"/>

    <path d="M 375 220 L 395 220 L 385 235 Z" fill="#e4e4e7"/>
    <path d="M 385 235 Q 360 260 385 290 Q 410 260 385 235"/>

    <path d="M 465 220 L 485 220 L 475 235 Z" fill="#e4e4e7"/>
    <path d="M 475 235 L 475 290 M 450 260 Q 475 260 500 290"/>
  </g>
</svg>
`;

const kadambaSvg = `
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 380" width="100%" height="100%">
  <rect width="600" height="380" fill="#141418"/>
  <rect width="600" height="380" fill="none" stroke="#27272a" stroke-width="1"/>

  <!-- Kadamba Box-Headed Glyphs -->
  <g fill="none" stroke="#e4e4e7" stroke-width="4.5" stroke-linecap="round" stroke-linejoin="round" opacity="0.9">
    <!-- Square box top mark -->
    <rect x="70" y="70" width="20" height="16" fill="none" stroke="#e4e4e7" stroke-width="3"/>
    <path d="M 80 86 L 80 155 M 55 120 Q 80 115 110 120"/>

    <rect x="170" y="70" width="20" height="16" fill="none" stroke="#e4e4e7" stroke-width="3"/>
    <path d="M 180 86 L 180 125 Q 180 155 210 155 Q 230 155 230 125 L 230 86"/>

    <rect x="280" y="70" width="20" height="16" fill="none" stroke="#e4e4e7" stroke-width="3"/>
    <path d="M 290 86 L 290 155 M 265 155 Q 290 120 320 155"/>

    <rect x="390" y="70" width="20" height="16" fill="none" stroke="#e4e4e7" stroke-width="3"/>
    <path d="M 400 86 Q 430 110 400 155"/>

    <rect x="490" y="70" width="20" height="16" fill="none" stroke="#e4e4e7" stroke-width="3"/>
    <path d="M 500 86 L 500 155 M 475 115 Q 500 115 525 145"/>

    <!-- Line 2 -->
    <rect x="70" y="210" width="20" height="16" fill="none" stroke="#e4e4e7" stroke-width="3"/>
    <path d="M 80 226 L 80 290 L 115 290"/>

    <rect x="180" y="210" width="20" height="16" fill="none" stroke="#e4e4e7" stroke-width="3"/>
    <path d="M 190 226 Q 220 250 190 290"/>

    <rect x="290" y="210" width="20" height="16" fill="none" stroke="#e4e4e7" stroke-width="3"/>
    <path d="M 300 226 L 300 290 M 275 255 L 325 255"/>

    <rect x="400" y="210" width="20" height="16" fill="none" stroke="#e4e4e7" stroke-width="3"/>
    <path d="M 410 226 Q 380 255 410 290 Q 440 255 410 226"/>
  </g>
</svg>
`;

export const SAMPLE_INSCRIPTIONS = [
  {
    id: 'ashokan-brahmi',
    title: 'Ashokan Rock Edict (Girnar)',
    script: 'Ashokan Brahmi',
    period: 'c. 250 BCE',
    medium: 'Rock Edict',
    description: 'Monumental Mauryan rock inscription in geometric Brahmi.',
    thumbnail: svgToDataUrl(ashokanBrahmiSvg),
    expectedScript: 'Ashokan Brahmi'
  },
  {
    id: 'tamil-brahmi',
    title: 'Mangulam Cavern Inscription',
    script: 'Tamil-Brahmi',
    period: 'c. 2nd Century BCE',
    medium: 'Cave Brow Inscription',
    description: 'Sangam-era cavern inscription recording monastic stone bed endowments.',
    thumbnail: svgToDataUrl(tamilBrahmiSvg),
    expectedScript: 'Tamil-Brahmi'
  },
  {
    id: 'kharosthi',
    title: 'Gandharan Buddhist Schist',
    script: 'Kharosthi',
    period: 'c. 1st Century CE',
    medium: 'Schist Slab',
    description: 'Right-to-left flowing Kharosthi text in Gandhari Prakrit.',
    thumbnail: svgToDataUrl(kharosthiSvg),
    expectedScript: 'Kharosthi'
  },
  {
    id: 'grantha',
    title: 'Pallava Copper Plate Charter',
    script: 'Grantha',
    period: 'c. 7th Century CE',
    medium: 'Copper Plate (Tamra-shasana)',
    description: 'Royal Sanskrit charter in curved, flowing Grantha letterforms.',
    thumbnail: svgToDataUrl(granthaSvg),
    expectedScript: 'Grantha'
  },
  {
    id: 'gupta-script',
    title: 'Prayag Prashasti (Allahabad)',
    script: 'Gupta Script',
    period: 'c. 4th Century CE',
    medium: 'Sandstone Pillar',
    description: 'Classical Sanskrit eulogy with transitional wedge-headed headmarks.',
    thumbnail: svgToDataUrl(guptaScriptSvg),
    expectedScript: 'Gupta Script'
  },
  {
    id: 'kadamba',
    title: 'Halmidi Stone Inscription',
    script: 'Kadamba',
    period: 'c. 450 CE',
    medium: 'Basalt Stele',
    description: 'Oldest known Kannada epigraph with square box-head serifs.',
    thumbnail: svgToDataUrl(kadambaSvg),
    expectedScript: 'Kadamba'
  }
];

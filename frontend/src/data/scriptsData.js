/**
 * DeepScript — Ancient Indian Scripts Database
 * Curated reference records for historical Indian epigraphic scripts.
 */

export const ANCIENT_SCRIPTS = {
  'Ashokan Brahmi': {
    name: 'Ashokan Brahmi',
    alternateNames: ['Early Brahmi', 'Mauryan Brahmi', 'Imperial Brahmi'],
    period: 'c. 3rd Century BCE',
    region: 'Pan-Indian (Northern, Central, and Deccan India)',
    ancestor: 'Proto-Brahmi',
    descendants: ['Gupta Brahmi', 'Tamil-Brahmi', 'Bhattiprolu', 'Kadamba', 'Grantha'],
    mediums: ['Sandstone Rock Edicts', 'Pillars', 'Cavern Walls'],
    keyFeatures: [
      'Standardized geometric and symmetrical letterforms.',
      'Perpendicular vertical stems and crisp circular arcs.',
      'Written left-to-right with perpendicular vowel matras.',
      'Foundational ancestor to all modern Brahmic scripts of India.'
    ],
    famousInscriptions: [
      'Ashokan Major Rock Edict (Girnar, Gujarat)',
      'Delhi-Topra Pillar Inscription',
      'Lumbini Pillar Inscription'
    ],
    historicalContext:
      'The monumental script used across Emperor Ashoka’s empire to inscribe edicts of Dhamma on rock faces and polished pillars. It is the primary graphical ancestor of almost all historical and modern Indian scripts.',
    visualClues: 'Distinct cross shapes (+ for Ka), circles (Ma/Tha), and straight vertical spine lines.'
  },

  'Tamil-Brahmi': {
    name: 'Tamil-Brahmi',
    alternateNames: ['Tamizhi', 'Damili', 'Dravida Brahmi'],
    period: 'c. 3rd Century BCE – 5th Century CE',
    region: 'Southern India (Tamil Nadu, Kerala)',
    ancestor: 'Early Southern Brahmi',
    descendants: ['Vatteluttu', 'Pallava Grantha', 'Modern Tamil'],
    mediums: ['Natural Rock Caverns', 'Jain Stone Beds', 'Inscribed Potsherds'],
    keyFeatures: [
      'Adapted specifically for Old Tamil phonology with characters for ழ (ḻ), ள (ḷ), ற (ṟ), ன (ṉ).',
      'Pioneered early pulli (virama / vowel cancellation) markings.',
      'Simple, clean, unornamented strokes engraved on cavern brows and pottery.',
      'Primary epigraphic evidence for early Sangam-era culture and trade.'
    ],
    famousInscriptions: [
      'Mangulam Cave Inscriptions (Madurai)',
      'Sittanavasal Jain Caverns (Pudukkottai)',
      'Keezhadi Inscribed Potsherds'
    ],
    historicalContext:
      'Used by Sangam-era chieftains, merchants, and ascetics in South India, Tamil-Brahmi modified early Brahmi to represent Dravidian sounds.',
    visualClues: 'Loop-based Dravidian glyphs, short single-line cavern dedications, lack of heavy headmarks.'
  },

  'Kharosthi': {
    name: 'Kharosthi',
    alternateNames: ['Gandhari Script', 'Indo-Bactrian'],
    period: 'c. 4th Century BCE – 3rd Century CE',
    region: 'Northwestern Ancient India (Gandhara Region / Indus Basin)',
    ancestor: 'Achaemenid Aramaic Influenced',
    descendants: ['Northwestern Silk Road cursive variants'],
    mediums: ['Birch Bark Manuscripts', 'Schist Slabs', 'Reliquary Inscriptions', 'Coins'],
    keyFeatures: [
      'Written **Right-to-Left** (unique among ancient Indian scripts).',
      'Cursive, fluid descenders derived from reed pen traditions.',
      'Slanted vowel diacritic strokes placed across upper consonant hooks.',
      'Standard script for Gandhari Prakrit and early Buddhist texts.'
    ],
    famousInscriptions: [
      'Shahbazgarhi Rock Edict of Ashoka',
      'Takht-i-Bahi Stone Inscription',
      'Gandharan Buddhist Birch Bark Scrolls'
    ],
    historicalContext:
      'Flourished in northwestern India under the Mauryas, Indo-Greeks, and Kushans, serving as the script for early Buddhist canon along ancient trade routes.',
    visualClues: 'Flows right-to-left with hooked upper strokes and cursive downward descenders.'
  },

  'Grantha': {
    name: 'Grantha',
    alternateNames: ['Pallava Grantha', 'Middle Grantha'],
    period: 'c. 5th – 19th Century CE',
    region: 'Southern India (Tamil Nadu, Kerala, Andhra)',
    ancestor: 'Southern Brahmi / Pallava Script',
    descendants: ['Malayalam Script', 'Tigalari Script'],
    mediums: ['Granite Temple Inscriptions', 'Copper Plate Charters (Tamra-shasana)', 'Palm Leaves'],
    keyFeatures: [
      'Smooth, circular and rounded letterforms suited for incising on palm leaves without tearing.',
      'Vertical conjunct consonant stacking (ligatures).',
      'Used in South India to transcribe Sanskrit texts alongside regional scripts.',
      'Influenced multiple maritime epigraphic traditions.'
    ],
    famousInscriptions: [
      'Mamallapuram Shore Temple & Cave Inscriptions',
      'Kailasanathar Temple Inscriptions (Kanchipuram)',
      'Thiruvalangadu Copper Plates of Rajendra Chola I'
    ],
    historicalContext:
      'Developed during the Pallava dynasty and refined under Chola rule, Grantha was the premier script for Sanskrit epigraphs and copper plate royal grants in South India.',
    visualClues: 'Circular loops, ornate rounded curves, and stacked vertical consonant conjuncts.'
  },

  'Gupta Script': {
    name: 'Gupta Script',
    alternateNames: ['Gupta Brahmi', 'Late Northern Brahmi'],
    period: 'c. 4th – 6th Century CE',
    region: 'Northern & Central India',
    ancestor: 'Kushan Brahmi',
    descendants: ['Siddhamatrika (Siddham)', 'Sharada', 'Devanagari', 'Bengali-Assamese'],
    mediums: ['Sandstone Pillars', 'Gold & Silver Coins', 'Copper Plates'],
    keyFeatures: [
      'Introduction of triangular solid head-marks (serifs) atop vertical strokes.',
      'Broad, sweeping brush and chisel stroke variations.',
      'Early evolutionary precursor to the continuous headline (shirorekha) of modern Devanagari.',
      'Associated with classical Sanskrit literature during the Gupta era.'
    ],
    famousInscriptions: [
      'Prayag Prashasti (Allahabad Pillar) of Samudragupta',
      'Iron Pillar Inscription of Delhi (King Chandra)',
      'Eran Stone Inscription of Budhagupta'
    ],
    historicalContext:
      'Used during the Golden Age of the Gupta Empire, this script forms the crucial structural link between early geometric Brahmi and northern medieval Nagari scripts.',
    visualClues: 'Triangular wedge/nail headmarks atop letters and broad curved bases.'
  },

  'Kadamba': {
    name: 'Kadamba',
    alternateNames: ['Old Kannada-Telugu', 'Kadamba-Pallava Script'],
    period: 'c. 4th – 7th Century CE',
    region: 'Deccan India (Karnataka, Andhra Pradesh)',
    ancestor: 'Southern Brahmi (Satavahana lineage)',
    descendants: ['Hale Kannada (Old Kannada)', 'Old Telugu'],
    mediums: ['Basalt Stone Pillars', 'Cave Temples', 'Copper Plates'],
    keyFeatures: [
      'Distinctive square hollow box head-marks atop characters.',
      'Evolution towards rounded circular curves in the lower body of letters.',
      'Deep, clear chisel relief on Deccan basalt stones.',
      'Ancestor of the regional Kannada and Telugu writing traditions.'
    ],
    famousInscriptions: [
      'Halmidi Stone Inscription (Oldest known Kannada inscription, c. 450 CE)',
      'Talagunda Pillar Inscription of Shantivarman',
      'Badami Cave Inscriptions'
    ],
    historicalContext:
      'Developed by the chancery of the Kadamba Dynasty of Banavasi, marking the distinct epigraphic foundation for scripts of the Deccan plateau.',
    visualClues: 'Square hollow box head-marks (box-headed characters) with wide lower loops.'
  },

  'Sharada': {
    name: 'Sharada',
    alternateNames: ['Sarada Script', 'Kashmiri Sharada'],
    period: 'c. 8th – 13th Century CE',
    region: 'Northwestern Himalayas (Kashmir, Himachal, Punjab)',
    ancestor: 'Western Gupta / Siddhamatrika',
    descendants: ['Gurmukhi', 'Takri', 'Dogri'],
    mediums: ['Birch Bark (Bhurjapatra) Manuscripts', 'Stone Slabs', 'Temple Prashastis'],
    keyFeatures: [
      'Individual horizontal top bars with pointed, stiff vertical stems.',
      'Sharp right-angled joints adapted for pen strokes on birch bark.',
      'Preserved ancient scientific, mathematical, and philosophical manuscripts.',
      'Used for Sanskrit and early Kashmiri texts.'
    ],
    famousInscriptions: [
      'Bakhshali Mathematical Manuscript',
      'Martand Sun Temple Inscriptions (Kashmir)',
      'Baijnath Temple Prashastis (Himachal Pradesh)'
    ],
    historicalContext:
      'The premier literary and intellectual script of the Northwestern Subcontinent, famous for preserving Kashmiri Sanskrit literature and mathematical works.',
    visualClues: 'Straight individual top bars with rigid sharp angular drops.'
  },

  'Vatteluttu': {
    name: 'Vatteluttu',
    alternateNames: ['Vattezhuthu', 'Round Script'],
    period: 'c. 5th – 14th Century CE',
    region: 'Southern India (Kerala, Southern Tamil Nadu)',
    ancestor: 'Tamil-Brahmi',
    descendants: ['Kolezhuthu', 'Influence on Malayalam'],
    mediums: ['Hero Stones (Viragal)', 'Granite Slabs', 'Copper Plates'],
    keyFeatures: [
      'Extremely cursive, rounded letterforms designed for rapid continuous carving.',
      'Smooth single-stroke flowing circles without sharp corners or boxes.',
      'Used extensively for civic records, land deeds, and hero memorials.',
      'Prevalent under early Pandya and Chera administrations.'
    ],
    famousInscriptions: [
      'Thirunatharkunru Hero Stone Inscription',
      'Quilon Syrian Copper Plates (Kerala)',
      'Jewish Copper Plate of Bhaskara Ravi Varman (Cochin)'
    ],
    historicalContext:
      'A fluid, everyday cursive script in South India that evolved directly from Tamil-Brahmi for administrative, legal, and memorial recordings.',
    visualClues: 'Smooth swirling circles, fluid continuous loops, absence of top boxes or wedge headmarks.'
  }
};

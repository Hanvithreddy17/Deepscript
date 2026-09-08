/**
 * DeepScript — 62-Class Ancient Indian Epigraphic & Phonetic Map
 * Maps raw model output labels to rich epigraphic, phonetic, and historical descriptions.
 */

export const CHARACTER_MAP = {
  // --- Vowels ---
  'a': {
    name: 'Brahmi A (𑀅)',
    transliteration: 'a',
    category: 'Independent Vowel',
    phonetic: 'Short open central unrounded vowel [ɐ / ʌ]',
    scriptFamily: 'Ashokan Brahmi & Lineage',
    period: 'c. 3rd Century BCE onwards',
    region: 'Pan-Indian Epigraphy',
    visualClues: 'Two leftward concave semicircles/hooks connected at a central junction meeting a vertical right spine.',
    historicalContext: 'Foundational initial vowel across Mauryan and Southern rock edicts.'
  },
  'aa': {
    name: 'Brahmi Ā (𑀆)',
    transliteration: 'ā / aa',
    category: 'Independent Long Vowel',
    phonetic: 'Long open central unrounded vowel [aː]',
    scriptFamily: 'Ashokan Brahmi & Lineage',
    period: 'c. 3rd Century BCE onwards',
    region: 'Pan-Indian Epigraphy',
    visualClues: 'Base character "a" with an extended horizontal vowel stroke (matra) attached to the top right stem.',
    historicalContext: 'Primary long vowel glyph engraved across royal and monastic epigraphs.'
  },
  'i': {
    name: 'Brahmi I (𑀇)',
    transliteration: 'i',
    category: 'Independent Short Vowel',
    phonetic: 'Close front unrounded vowel [i]',
    scriptFamily: 'Ashokan Brahmi',
    period: 'c. 3rd Century BCE',
    region: 'Pan-Indian Epigraphy',
    visualClues: 'Three or four distinct dots/circles arranged in a triangular or quad pattern.',
    historicalContext: 'Earliest form of the initial short vowel I, highly distinctive in early Mauryan edicts.'
  },
  'ii': {
    name: 'Brahmi Ī (𑀈)',
    transliteration: 'ī / ii',
    category: 'Independent Long Vowel',
    phonetic: 'Long close front unrounded vowel [iː]',
    scriptFamily: 'Ashokan Brahmi & Gupta Lineage',
    period: 'c. 3rd Century BCE – 5th Century CE',
    region: 'Northern & Deccan India',
    visualClues: 'Two vertical parallel bars or clustered dots representing long vowel elongation.',
    historicalContext: 'Used for initial Sanskrit and Prakrit long "ī" sounds in classical records.'
  },
  'u': {
    name: 'Brahmi U (𑀉)',
    transliteration: 'u',
    category: 'Independent Short Vowel',
    phonetic: 'Close back rounded vowel [u]',
    scriptFamily: 'Ashokan Brahmi & Lineage',
    period: 'c. 3rd Century BCE onwards',
    region: 'Pan-Indian Epigraphy',
    visualClues: 'Right-angled L-shaped vertical stem extending into a horizontal rightward base.',
    historicalContext: 'Simple geometric right-angle glyph found consistently in Prakrit inscriptions.'
  },
  'uu': {
    name: 'Brahmi Ū (𑀊)',
    transliteration: 'ū / uu',
    category: 'Independent Long Vowel',
    phonetic: 'Long close back rounded vowel [uː]',
    scriptFamily: 'Ashokan Brahmi & Lineage',
    period: 'c. 3rd Century BCE onwards',
    region: 'Pan-Indian Epigraphy',
    visualClues: 'L-shape with an additional horizontal middle stroke indicating length.',
    historicalContext: 'Form representing the long high back vowel.'
  },
  'ri': {
    name: 'Brahmi Ṛ (𑀋)',
    transliteration: 'ṛ / ri',
    category: 'Vocalic Liquid',
    phonetic: 'Syllabic retroflex liquid [r̩]',
    scriptFamily: 'Brahmi / Sanskrit Epigraphy',
    period: 'c. 1st – 5th Century CE',
    region: 'Northern & Deccan India',
    visualClues: 'Vertical axis with curved or loop-ended base.',
    historicalContext: 'Introduced for Sanskrit phonetic precision in late Brahmi and Gupta inscriptions.'
  },
  'rii': {
    name: 'Brahmi Ṝ (𑀌)',
    transliteration: 'ṝ / rii',
    category: 'Long Vocalic Liquid',
    phonetic: 'Long syllabic retroflex liquid [r̩ː]',
    scriptFamily: 'Late Brahmi & Classical Epigraphy',
    period: 'c. 4th Century CE onwards',
    region: 'Classical Sanskrit Charters',
    visualClues: 'Elaborated Ṛ with secondary base loop.',
    historicalContext: 'Rare epigraphic vocalic character appearing in Vedic and grammatical texts.'
  },
  'lu': {
    name: 'Brahmi Ḷ / Ḷu',
    transliteration: 'ḷu / lu',
    category: 'Vocalic Liquid',
    phonetic: 'Syllabic lateral [l̩]',
    scriptFamily: 'Brahmi Lineage',
    period: 'c. 2nd – 6th Century CE',
    region: 'Western & Southern Epigraphy',
    visualClues: 'Hooked lateral base glyph.',
    historicalContext: 'Vocalic form preserved in classical epigraphs.'
  },
  'luu': {
    name: 'Brahmi Ḹu',
    transliteration: 'ḹu / luu',
    category: 'Long Vocalic Liquid',
    phonetic: 'Long syllabic lateral [l̩ː]',
    scriptFamily: 'Brahmi Lineage',
    period: 'c. 4th Century CE',
    region: 'Sanskrit Epigraphic Slabs',
    visualClues: 'Double hooked lateral loop.',
    historicalContext: 'Theoretical and grammatical Sanskrit vowel glyph.'
  },
  'e': {
    name: 'Brahmi E (𑀏)',
    transliteration: 'e',
    category: 'Independent Vowel',
    phonetic: 'Close-mid front unrounded vowel [eː / e]',
    scriptFamily: 'Ashokan Brahmi & Lineage',
    period: 'c. 3rd Century BCE onwards',
    region: 'Pan-Indian Epigraphy',
    visualClues: 'Clean triangular geometric shape pointing upward or rightward.',
    historicalContext: 'One of the most recognizable geometric symbols in Ashokan epigraphy (Δ-like glyph).'
  },
  'ai': {
    name: 'Brahmi Ai (𑀐)',
    transliteration: 'ai',
    category: 'Independent Diphthong',
    phonetic: 'Diphthong [aɪ / ɛː]',
    scriptFamily: 'Ashokan Brahmi & Lineage',
    period: 'c. 3rd Century BCE onwards',
    region: 'Pan-Indian Epigraphy',
    visualClues: 'Triangular base glyph with an extra apex diacritic stroke.',
    historicalContext: 'Diphthong initial glyph used in Prakrit and Sanskrit inscriptions.'
  },
  'o': {
    name: 'Brahmi O (𑀑)',
    transliteration: 'o',
    category: 'Independent Vowel',
    phonetic: 'Close-mid back rounded vowel [oː / o]',
    scriptFamily: 'Ashokan Brahmi & Lineage',
    period: 'c. 3rd Century BCE onwards',
    region: 'Pan-Indian Epigraphy',
    visualClues: 'Zig-zag / Z-shaped angular stroke configuration.',
    historicalContext: 'Distinctive stepped geometric symbol in early Indian epigraphy.'
  },
  'au': {
    name: 'Brahmi Au (𑀒)',
    transliteration: 'au',
    category: 'Independent Diphthong',
    phonetic: 'Diphthong [aʊ / ɔː]',
    scriptFamily: 'Ashokan Brahmi & Lineage',
    period: 'c. 3rd Century BCE onwards',
    region: 'Pan-Indian Epigraphy',
    visualClues: 'Stepped glyph with an additional upper horizontal branch.',
    historicalContext: 'Initial diphthong form found in royal panegyrics (prashastis).'
  },
  'am': {
    name: 'Brahmi Anusvara (Am / 𑀁)',
    transliteration: 'aṃ / am',
    category: 'Nasalized Vowel / Anusvara',
    phonetic: 'Nasal resonance / Anusvara [ŋ / ̃]',
    scriptFamily: 'Brahmi Epigraphy',
    period: 'c. 3rd Century BCE onwards',
    region: 'Pan-Indian Epigraphy',
    visualClues: 'Base vowel with distinct superior dot / bindu indicator.',
    historicalContext: 'Marks nasalization across Prakrit and Sanskrit edicts.'
  },
  'ah': {
    name: 'Brahmi Visarga (Ah / 𑀂)',
    transliteration: 'aḥ / ah',
    category: 'Aspirate / Visarga',
    phonetic: 'Voiceless glottal aspirate [h / x]',
    scriptFamily: 'Brahmi Epigraphy',
    period: 'c. 3rd Century BCE onwards',
    region: 'Pan-Indian Epigraphy',
    visualClues: 'Base character followed by two vertical dots (visarga column).',
    historicalContext: 'Marks post-vocalic aspiration in classical epigraphic Sanskrit.'
  },

  // --- Consonants: Velars ---
  'ka': {
    name: 'Brahmi Ka (𑀓)',
    transliteration: 'ka',
    category: 'Velar Stop (Voiceless Unaspirated)',
    phonetic: 'Voiceless velar plosive [k]',
    scriptFamily: 'Ashokan Brahmi & Descendants',
    period: 'c. 3rd Century BCE onwards',
    region: 'Pan-Indian Epigraphy',
    visualClues: 'Classic symmetric perpendicular cross (+) shape.',
    historicalContext: 'The most iconic character in Indian paleography, preserved on pillar and rock edicts across the subcontinent.'
  },
  'kha': {
    name: 'Brahmi Kha (𑀔)',
    transliteration: 'kha',
    category: 'Velar Stop (Voiceless Aspirated)',
    phonetic: 'Voiceless aspirated velar plosive [kʰ]',
    scriptFamily: 'Ashokan Brahmi & Lineage',
    period: 'c. 3rd Century BCE onwards',
    region: 'Pan-Indian Epigraphy',
    visualClues: 'Vertical stem ending in a small bottom circle or hook (crook/hook glyph).',
    historicalContext: 'Distinctive hooked stem symbolizing aspirated velar stop.'
  },
  'ga': {
    name: 'Brahmi Ga (𑀕)',
    transliteration: 'ga',
    category: 'Velar Stop (Voiced Unaspirated)',
    phonetic: 'Voiced velar plosive [ɡ]',
    scriptFamily: 'Ashokan Brahmi & Lineage',
    period: 'c. 3rd Century BCE onwards',
    region: 'Pan-Indian Epigraphy',
    visualClues: 'Inverted V-shape (chevron / Λ) or arch pointing upwards.',
    historicalContext: 'Symmetric geometric arch, foundational to later Nagari and Dravidian forms.'
  },
  'gha': {
    name: 'Brahmi Gha (𑀖)',
    transliteration: 'gha',
    category: 'Velar Stop (Voiced Aspirated)',
    phonetic: 'Voiced aspirated velar plosive [ɡʱ]',
    scriptFamily: 'Ashokan Brahmi & Lineage',
    period: 'c. 3rd Century BCE onwards',
    region: 'Pan-Indian Epigraphy',
    visualClues: 'W-shaped or trident-like multi-prong base with central vertical stem.',
    historicalContext: 'Aspirated voiced velar glyph frequent in royal titles and place names.'
  },

  // --- Consonants: Palatals ---
  'ca': {
    name: 'Brahmi Ca (𑀘)',
    transliteration: 'ca',
    category: 'Palatal Affricate (Voiceless Unaspirated)',
    phonetic: 'Voiceless palatal affricate [t͡ɕ / c]',
    scriptFamily: 'Ashokan Brahmi & Lineage',
    period: 'c. 3rd Century BCE onwards',
    region: 'Pan-Indian Epigraphy',
    visualClues: 'Vertical stem with a leftward loop or semicircle attached.',
    historicalContext: 'Found in early Prakrit royal edicts and dedicatory inscriptions.'
  },
  'cha': {
    name: 'Brahmi Cha (𑀙)',
    transliteration: 'cha',
    category: 'Palatal Affricate (Voiceless Aspirated)',
    phonetic: 'Voiceless aspirated palatal affricate [t͡ɕʰ / cʰ]',
    scriptFamily: 'Ashokan Brahmi & Lineage',
    period: 'c. 3rd Century BCE onwards',
    region: 'Pan-Indian Epigraphy',
    visualClues: 'Bisected circle with a vertical spine (figure-eight / 8-like form).',
    historicalContext: 'Double-lobed circular glyph signifying aspirated palatal sound.'
  },
  'ja': {
    name: 'Brahmi Ja (𑀚)',
    transliteration: 'ja',
    category: 'Palatal Affricate (Voiced Unaspirated)',
    phonetic: 'Voiced palatal affricate [d͡ʑ / ɟ]',
    scriptFamily: 'Ashokan Brahmi & Lineage',
    period: 'c. 3rd Century BCE onwards',
    region: 'Pan-Indian Epigraphy',
    visualClues: 'E-shaped glyph with three horizontal prongs attached to a vertical backbone.',
    historicalContext: 'Key character appearing in royal titles like "Raja" and "Jaya".'
  },
  'jha': {
    name: 'Brahmi Jha (𑀛)',
    transliteration: 'jha',
    category: 'Palatal Affricate (Voiced Aspirated)',
    phonetic: 'Voiced aspirated palatal affricate [d͡ʑʱ / ɟʱ]',
    scriptFamily: 'Ashokan Brahmi & Lineage',
    period: 'c. 3rd Century BCE onwards',
    region: 'Pan-Indian Epigraphy',
    visualClues: 'Vertical stem with horizontal side branches.',
    historicalContext: 'Aspirated voiced palatal character.'
  },
  'nya': {
    name: 'Brahmi Ña / Nya (𑀜)',
    transliteration: 'ña / nya',
    category: 'Palatal Nasal',
    phonetic: 'Palatal nasal [ɲ]',
    scriptFamily: 'Ashokan Brahmi & Lineage',
    period: 'c. 3rd Century BCE onwards',
    region: 'Pan-Indian Epigraphy',
    visualClues: 'Horizontal bar with two downward vertical legs and upper hook.',
    historicalContext: 'Palatal nasal glyph in Brahmic phonology.'
  },

  // --- Consonants: Retroflex ---
  'tta': {
    name: 'Brahmi Ṭa (𑀝)',
    transliteration: 'ṭa / tta',
    category: 'Retroflex Stop (Voiceless Unaspirated)',
    phonetic: 'Voiceless retroflex plosive [ʈ]',
    scriptFamily: 'Ashokan Brahmi & Lineage',
    period: 'c. 3rd Century BCE onwards',
    region: 'Pan-Indian Epigraphy',
    visualClues: 'Semicircular cup or C-shape opening upward or rightward.',
    historicalContext: 'Unaspirated retroflex stop in ancient Indian languages.'
  },
  'ttha': {
    name: 'Brahmi Ṭha (𑀞)',
    transliteration: 'ṭha / ttha',
    category: 'Retroflex Stop (Voiceless Aspirated)',
    phonetic: 'Voiceless aspirated retroflex plosive [ʈʰ]',
    scriptFamily: 'Ashokan Brahmi & Lineage',
    period: 'c. 3rd Century BCE onwards',
    region: 'Pan-Indian Epigraphy',
    visualClues: 'Complete clean circle (O) engraved on stone.',
    historicalContext: 'Clean circular glyph representing aspirated retroflex stop.'
  },
  'dda': {
    name: 'Brahmi Ḍa (𑀟)',
    transliteration: 'ḍa / dda',
    category: 'Retroflex Stop (Voiced Unaspirated)',
    phonetic: 'Voiced retroflex plosive [ɖ]',
    scriptFamily: 'Ashokan Brahmi & Lineage',
    period: 'c. 3rd Century BCE onwards',
    region: 'Pan-Indian Epigraphy',
    visualClues: 'Vertical line bending rightward with bottom hook (E-reversed / S-shape).',
    historicalContext: 'Voiced retroflex plosive.'
  },
  'ddha': {
    name: 'Brahmi Ḍha (𑀠)',
    transliteration: 'ḍha / ddha',
    category: 'Retroflex Stop (Voiced Aspirated)',
    phonetic: 'Voiced aspirated retroflex plosive [ɖʱ]',
    scriptFamily: 'Ashokan Brahmi & Lineage',
    period: 'c. 3rd Century BCE onwards',
    region: 'Pan-Indian Epigraphy',
    visualClues: 'Curved loop with internal hook or spiral tail.',
    historicalContext: 'Aspirated retroflex plosive.'
  },
  'nna': {
    name: 'Brahmi Ṇa (𑀡)',
    transliteration: 'ṇa / nna',
    category: 'Retroflex Nasal',
    phonetic: 'Retroflex nasal [ɳ]',
    scriptFamily: 'Ashokan Brahmi & Lineage',
    period: 'c. 3rd Century BCE onwards',
    region: 'Pan-Indian Epigraphy',
    visualClues: 'Two horizontal bars connected by a central vertical column (I-beam shape).',
    historicalContext: 'Standard retroflex nasal glyph across northern and southern inscriptions.'
  },
  'nnna': {
    name: 'Brahmi Ṉa (Special Dravidian Nasal)',
    transliteration: 'ṉa / nnna',
    category: 'Alveolar Nasal (Tamil-Brahmi)',
    phonetic: 'Alveolar nasal [n / ṉ]',
    scriptFamily: 'Tamil-Brahmi / Southern Brahmi',
    period: 'c. 3rd Century BCE – 4th Century CE',
    region: 'Southern India (Tamil Nadu, Kerala)',
    visualClues: 'Loop-based retroflex/alveolar character characteristic of southern caverns.',
    historicalContext: 'Specialized glyph developed in Tamil-Brahmi to capture Dravidian phonemes.'
  },

  // --- Consonants: Dentals ---
  'ta': {
    name: 'Brahmi Ta (𑀢)',
    transliteration: 'ta',
    category: 'Dental Stop (Voiceless Unaspirated)',
    phonetic: 'Voiceless dental plosive [t̪]',
    scriptFamily: 'Ashokan Brahmi & Lineage',
    period: 'c. 3rd Century BCE onwards',
    region: 'Pan-Indian Epigraphy',
    visualClues: 'Inverted tuning fork or umbrella rib (vertical line branching into two downward legs / ^).',
    historicalContext: 'Universal dental consonant occurring abundantly in Buddhist and civic inscriptions.'
  },
  'tha': {
    name: 'Brahmi Tha (𑀣)',
    transliteration: 'tha',
    category: 'Dental Stop (Voiceless Aspirated)',
    phonetic: 'Voiceless aspirated dental plosive [t̪ʰ]',
    scriptFamily: 'Ashokan Brahmi & Lineage',
    period: 'c. 3rd Century BCE onwards',
    region: 'Pan-Indian Epigraphy',
    visualClues: 'Circle with a central dot (dotted circle ʘ).',
    historicalContext: 'One of the most striking geometric glyphs in ancient Indian epigraphy.'
  },
  'da': {
    name: 'Brahmi Da (𑀤)',
    transliteration: 'da',
    category: 'Dental Stop (Voiced Unaspirated)',
    phonetic: 'Voiced dental plosive [d̪]',
    scriptFamily: 'Ashokan Brahmi & Lineage',
    period: 'c. 3rd Century BCE onwards',
    region: 'Pan-Indian Epigraphy',
    visualClues: 'C-shaped curve opening leftward or rightward with upper and lower vertical tails.',
    historicalContext: 'Common dental plosive found in Sanskrit terms like "Deva", "Dana" (grant/gift).'
  },
  'dha': {
    name: 'Brahmi Dha (𑀥)',
    transliteration: 'dha',
    category: 'Dental Stop (Voiced Aspirated)',
    phonetic: 'Voiced aspirated dental plosive [d̪ʱ]',
    scriptFamily: 'Ashokan Brahmi & Lineage',
    period: 'c. 3rd Century BCE onwards',
    region: 'Pan-Indian Epigraphy',
    visualClues: 'D-shaped capital letter with a flat vertical spine and rightward bow.',
    historicalContext: 'Primary consonant in words like "Dhamma" and "Dharma" across Ashokan edicts.'
  },
  'na': {
    name: 'Brahmi Na (𑀦)',
    transliteration: 'na',
    category: 'Dental Nasal',
    phonetic: 'Dental nasal [n̪]',
    scriptFamily: 'Ashokan Brahmi & Lineage',
    period: 'c. 3rd Century BCE onwards',
    region: 'Pan-Indian Epigraphy',
    visualClues: 'Inverted T-shape (horizontal top bar with downward vertical stem, or vertical stem with horizontal base).',
    historicalContext: 'Key dental nasal glyph.'
  },

  // --- Consonants: Labials ---
  'pa': {
    name: 'Brahmi Pa (𑀧)',
    transliteration: 'pa',
    category: 'Labial Stop (Voiceless Unaspirated)',
    phonetic: 'Voiceless bilabial plosive [p]',
    scriptFamily: 'Ashokan Brahmi & Lineage',
    period: 'c. 3rd Century BCE onwards',
    region: 'Pan-Indian Epigraphy',
    visualClues: 'Open U-hook or J-shape with one longer vertical arm.',
    historicalContext: 'Frequent consonant in kingly titles and Prakrit verbs.'
  },
  'pha': {
    name: 'Brahmi Pha (𑀨)',
    transliteration: 'pha',
    category: 'Labial Stop (Voiceless Aspirated)',
    phonetic: 'Voiceless aspirated bilabial plosive [pʰ]',
    scriptFamily: 'Ashokan Brahmi & Lineage',
    period: 'c. 3rd Century BCE onwards',
    region: 'Pan-Indian Epigraphy',
    visualClues: 'Hook shape with an internal loop or curl at the tip.',
    historicalContext: 'Aspirated bilabial stop.'
  },
  'ba': {
    name: 'Brahmi Ba (𑀩)',
    transliteration: 'ba',
    category: 'Labial Stop (Voiced Unaspirated)',
    phonetic: 'Voiced bilabial plosive [b]',
    scriptFamily: 'Ashokan Brahmi & Lineage',
    period: 'c. 3rd Century BCE onwards',
    region: 'Pan-Indian Epigraphy',
    visualClues: 'Clean geometric square (□) or rectangle.',
    historicalContext: 'Iconic geometric square character in Ashokan Brahmi.'
  },
  'bha': {
    name: 'Brahmi Bha (𑀪)',
    transliteration: 'bha',
    category: 'Labial Stop (Voiced Aspirated)',
    phonetic: 'Voiced aspirated bilabial plosive [bʱ]',
    scriptFamily: 'Ashokan Brahmi & Lineage',
    period: 'c. 3rd Century BCE onwards',
    region: 'Pan-Indian Epigraphy',
    visualClues: 'Vertical line with upper and lower rightward horizontal arms (stepped bar).',
    historicalContext: 'Common in words like "Bhagavan" and "Bhikshu".'
  },
  'ma': {
    name: 'Brahmi Ma (𑀫)',
    transliteration: 'ma',
    category: 'Labial Nasal',
    phonetic: 'Bilabial nasal [m]',
    scriptFamily: 'Ashokan Brahmi & Lineage',
    period: 'c. 3rd Century BCE onwards',
    region: 'Pan-Indian Epigraphy',
    visualClues: 'Circle or loop resting atop a downward U-shaped or angled base (hourglass/fish-like).',
    historicalContext: 'One of the most frequent glyphs in all Indian epigraphy, appearing in names and titles.'
  },

  // --- Semivowels & Liquids ---
  'ya': {
    name: 'Brahmi Ya (𑀬)',
    transliteration: 'ya',
    category: 'Palatal Approximant (Semivowel)',
    phonetic: 'Palatal approximant [j]',
    scriptFamily: 'Ashokan Brahmi & Lineage',
    period: 'c. 3rd Century BCE onwards',
    region: 'Pan-Indian Epigraphy',
    visualClues: 'Three-pronged tuning fork / anchor shape with curved outer prongs and central vertical stem.',
    historicalContext: 'Classic anchor-shaped character in Ashokan and Gupta inscriptions.'
  },
  'ra': {
    name: 'Brahmi Ra (𑀭)',
    transliteration: 'ra',
    category: 'Alveolar Tap / Trill',
    phonetic: 'Alveolar trill / tap [r / ɾ]',
    scriptFamily: 'Ashokan Brahmi & Lineage',
    period: 'c. 3rd Century BCE onwards',
    region: 'Pan-Indian Epigraphy',
    visualClues: 'Single vertical straight line (|) or slightly wavy vertical stem.',
    historicalContext: 'The simplest character in the Brahmi script, written as a single vertical stroke.'
  },
  'rra': {
    name: 'Brahmi Ṟa (Dravidian Alveolar Trill)',
    transliteration: 'ṟa / rra',
    category: 'Alveolar Trill (Tamil-Brahmi)',
    phonetic: 'Alveolar trill [r / ṟ]',
    scriptFamily: 'Tamil-Brahmi / Southern Brahmi',
    period: 'c. 3rd Century BCE onwards',
    region: 'Southern India',
    visualClues: 'Modified vertical stem with horizontal base or loop.',
    historicalContext: 'Distinctive Old Tamil phonological character in southern epigraphs.'
  },
  'la': {
    name: 'Brahmi La (𑀮)',
    transliteration: 'la',
    category: 'Alveolar Lateral Approximant',
    phonetic: 'Alveolar lateral approximant [l]',
    scriptFamily: 'Ashokan Brahmi & Lineage',
    period: 'c. 3rd Century BCE onwards',
    region: 'Pan-Indian Epigraphy',
    visualClues: 'Hook or sickle-shaped curve opening to the right.',
    historicalContext: 'Lateral approximant glyph.'
  },
  'wo': {
    name: 'Brahmi Va / Wo (𑀯)',
    transliteration: 'va / wo',
    category: 'Labiodental / Labial Approximant',
    phonetic: 'Labiodental approximant [ʋ / w]',
    scriptFamily: 'Ashokan Brahmi & Lineage',
    period: 'c. 3rd Century BCE onwards',
    region: 'Pan-Indian Epigraphy',
    visualClues: 'Circle resting on a vertical base or circle with upward vertical stem (lollipop/balloon shape).',
    historicalContext: 'Key character in terms like "Vihara", "Varsha", "Vijaya".'
  },

  // --- Sibilants & Aspirate ---
  'sha': {
    name: 'Brahmi Śa (𑀰)',
    transliteration: 'śa / sha',
    category: 'Palatal Sibilant',
    phonetic: 'Voiceless palato-alveolar sibilant [ɕ / ʃ]',
    scriptFamily: 'Late Brahmi & Sanskrit Epigraphy',
    period: 'c. 1st – 5th Century CE',
    region: 'Northern & Central India',
    visualClues: 'Bow/horseshoe with a central horizontal connecting bar.',
    historicalContext: 'Palatal sibilant in classical Sanskrit inscriptions.'
  },
  'ssa': {
    name: 'Brahmi Ṣa (𑀱)',
    transliteration: 'ṣa / ssa',
    category: 'Retroflex Sibilant',
    phonetic: 'Voiceless retroflex sibilant [ʂ]',
    scriptFamily: 'Ashokan Brahmi & Lineage',
    period: 'c. 3rd Century BCE onwards',
    region: 'Pan-Indian Epigraphy',
    visualClues: 'Pa-like hook with an internal diagonal cross-bar.',
    historicalContext: 'Retroflex sibilant.'
  },
  'saa': {
    name: 'Brahmi Ṣā / Ssa (Long Retroflex Sibilant)',
    transliteration: 'ṣā / saa',
    category: 'Retroflex Sibilant (Vowel Attached)',
    phonetic: 'Voiceless retroflex sibilant with long vowel [ʂaː]',
    scriptFamily: 'Brahmi & Gupta Epigraphy',
    period: 'c. 3rd Century BCE – 6th Century CE',
    region: 'Pan-Indian Epigraphy',
    visualClues: 'Cross-barred hook with extended rightward matra stroke.',
    historicalContext: 'Retroflex sibilant with long ā vowel.'
  },
  'sa': {
    name: 'Brahmi Sa (𑀲)',
    transliteration: 'sa',
    category: 'Dental Sibilant',
    phonetic: 'Voiceless dental sibilant [s̪]',
    scriptFamily: 'Ashokan Brahmi & Lineage',
    period: 'c. 3rd Century BCE onwards',
    region: 'Pan-Indian Epigraphy',
    visualClues: 'Hooked loop on the left with a vertical spine and crossbar on the right.',
    historicalContext: 'The most frequent sibilant in Prakrit inscriptions (e.g., "Samana", "Sata").'
  },
  'ha': {
    name: 'Brahmi Ha (𑀳)',
    transliteration: 'ha',
    category: 'Glottal Fricative',
    phonetic: 'Voiced glottal fricative [ɦ]',
    scriptFamily: 'Ashokan Brahmi & Lineage',
    period: 'c. 3rd Century BCE onwards',
    region: 'Pan-Indian Epigraphy',
    visualClues: 'Vertical stem with a rightward upper hook and a horizontal shoulder bar.',
    historicalContext: 'Aspirate glottal sound.'
  },

  // --- Conjuncts ---
  'ksa': {
    name: 'Brahmi Kṣa (𑀓𑁆𑀱 / Ksha)',
    transliteration: 'kṣa / ksha / ksa',
    category: 'Compound Consonant Conjunct',
    phonetic: 'Velar-retroflex conjunct cluster [kʂɐ]',
    scriptFamily: 'Late Brahmi & Classical Epigraphy',
    period: 'c. 1st – 6th Century CE',
    region: 'Northern, Central & Deccan India',
    visualClues: 'Vertical stacking of "Ka" (+) above "Ṣa" (cross-barred hook).',
    historicalContext: 'Major Sanskrit ligature occurring in regal epithets like "Kshatrapa", "Kshatriya".'
  },
  'tra': {
    name: 'Brahmi Tra (𑀢𑁆𑀭)',
    transliteration: 'tra',
    category: 'Compound Consonant Conjunct',
    phonetic: 'Dental-alveolar conjunct cluster [t̪rɐ]',
    scriptFamily: 'Brahmi & Gupta Epigraphy',
    period: 'c. 3rd Century BCE onwards',
    region: 'Pan-Indian Epigraphy',
    visualClues: 'Ta (inverted tuning fork) combined with an elongated downward Ra stroke.',
    historicalContext: 'Found in words like "Mitra", "Putra" across royal charters.'
  },
  'gyan': {
    name: 'Brahmi Jña / Gyan (𑀚𑁆𑀜)',
    transliteration: 'jña / gyan',
    category: 'Compound Consonant Conjunct',
    phonetic: 'Palatal conjunct cluster [ɟɲɐ / ɡjɐn]',
    scriptFamily: 'Late Brahmi & Classical Epigraphy',
    period: 'c. 2nd – 7th Century CE',
    region: 'Pan-Indian Epigraphy',
    visualClues: 'Composite ligature of Ja and Ña.',
    historicalContext: 'Sanskrit philosophical term ligature (e.g. "Jñana" / wisdom).'
  },

  // --- Epigraphic Numerals ---
  'zero': {
    name: 'Ancient Indian Zero (Bindu / 𑁦)',
    transliteration: 'śūnya / zero (0)',
    category: 'Numeral / Place Value Sign',
    phonetic: 'Śūnya [0]',
    scriptFamily: 'Ancient Indian Mathematics & Epigraphy',
    period: 'c. 3rd Century BCE onwards',
    region: 'Gwalior, Bakhshali, Gujarat',
    visualClues: 'Small circular ring or solid dot (bindu) engraved as a numerical place holder.',
    historicalContext: 'The foundational Indian contribution to world mathematics, recorded in the Gwalior temple inscription and Bakhshali manuscript.'
  },
  'one': {
    name: 'Ancient Indian Numeral 1 (𑁧)',
    transliteration: 'eka / one (1)',
    category: 'Epigraphic Numeral',
    phonetic: 'Eka [1]',
    scriptFamily: 'Brahmi Numeral System',
    period: 'c. 3rd Century BCE onwards',
    region: 'Pan-Indian Epigraphy',
    visualClues: 'Single horizontal stroke or vertical tally bar.',
    historicalContext: 'Standard numeral 1 in early Indian edicts and land grant copper plates.'
  },
  'two': {
    name: 'Ancient Indian Numeral 2 (𑁨)',
    transliteration: 'dvi / two (2)',
    category: 'Epigraphic Numeral',
    phonetic: 'Dvi [2]',
    scriptFamily: 'Brahmi Numeral System',
    period: 'c. 3rd Century BCE onwards',
    region: 'Pan-Indian Epigraphy',
    visualClues: 'Two parallel horizontal bars or curved cursive joint.',
    historicalContext: 'Ancestor of the modern Arabic-Indic numeral 2.'
  },
  'three': {
    name: 'Ancient Indian Numeral 3 (𑁩)',
    transliteration: 'tri / three (3)',
    category: 'Epigraphic Numeral',
    phonetic: 'Tri [3]',
    scriptFamily: 'Brahmi Numeral System',
    period: 'c. 3rd Century BCE onwards',
    region: 'Pan-Indian Epigraphy',
    visualClues: 'Three stacked horizontal strokes or stepped curves.',
    historicalContext: 'Ancestor of modern digit 3.'
  },
  'four': {
    name: 'Ancient Indian Numeral 4 (𑁪)',
    transliteration: 'catur / four (4)',
    category: 'Epigraphic Numeral',
    phonetic: 'Catur [4]',
    scriptFamily: 'Brahmi Numeral System',
    period: 'c. 3rd Century BCE onwards',
    region: 'Pan-Indian Epigraphy',
    visualClues: 'Cross-like or folded loop glyph.',
    historicalContext: 'Epigraphic numeral representing 4.'
  },
  'five': {
    name: 'Ancient Indian Numeral 5 (𑁫)',
    transliteration: 'pañca / five (5)',
    category: 'Epigraphic Numeral',
    phonetic: 'Pañca [5]',
    scriptFamily: 'Brahmi Numeral System',
    period: 'c. 3rd Century BCE onwards',
    region: 'Pan-Indian Epigraphy',
    visualClues: 'Open curve with top horizontal bar.',
    historicalContext: 'Standard epigraphic numeral 5.'
  },
  'six': {
    name: 'Ancient Indian Numeral 6 (𑁬)',
    transliteration: 'ṣaṭ / six (6)',
    category: 'Epigraphic Numeral',
    phonetic: 'Ṣaṭ [6]',
    scriptFamily: 'Brahmi Numeral System',
    period: 'c. 3rd Century BCE onwards',
    region: 'Pan-Indian Epigraphy',
    visualClues: 'Coiled circular loop with upper stem.',
    historicalContext: 'Epigraphic numeral representing 6.'
  },
  'seven': {
    name: 'Ancient Indian Numeral 7 (𑁭)',
    transliteration: 'sapta / seven (7)',
    category: 'Epigraphic Numeral',
    phonetic: 'Sapta [7]',
    scriptFamily: 'Brahmi Numeral System',
    period: 'c. 3rd Century BCE onwards',
    region: 'Pan-Indian Epigraphy',
    visualClues: 'Upper hook with a downward curved stroke.',
    historicalContext: 'Epigraphic numeral representing 7.'
  },
  'eight': {
    name: 'Ancient Indian Numeral 8 (𑁮)',
    transliteration: 'aṣṭa / eight (8)',
    category: 'Epigraphic Numeral',
    phonetic: 'Aṣṭa [8]',
    scriptFamily: 'Brahmi Numeral System',
    period: 'c. 3rd Century BCE onwards',
    region: 'Pan-Indian Epigraphy',
    visualClues: 'Curved arch with two side prongs.',
    historicalContext: 'Epigraphic numeral representing 8.'
  },
  'nine': {
    name: 'Ancient Indian Numeral 9 (𑁯)',
    transliteration: 'nava / nine (9)',
    category: 'Epigraphic Numeral',
    phonetic: 'Nava [9]',
    scriptFamily: 'Brahmi Numeral System',
    period: 'c. 3rd Century BCE onwards',
    region: 'Pan-Indian Epigraphy',
    visualClues: 'Downward spiral or hook with upper loop.',
    historicalContext: 'Ancestor of the modern numeral 9.'
  },
};

/**
 * Retrieves character details with safe fallback.
 */
export function getCharacterDetails(classLabel) {
  if (!classLabel) return null;
  const key = String(classLabel).toLowerCase().trim();
  return CHARACTER_MAP[key] || {
    name: `Class: ${classLabel}`,
    transliteration: classLabel,
    category: 'Ancient Indian Inscription Symbol',
    phonetic: 'Historical epigraphic character',
    scriptFamily: 'Early Brahmi & Descendant Scripts',
    period: 'c. 3rd BCE – 8th CE',
    region: 'Historical Subcontinent',
    visualClues: 'Identified via fine-tuned Vision Transformer metric feature embedding.',
    historicalContext: `Identified character family '${classLabel}' from the 62-class DeepScript epigraphic corpus.`
  };
}

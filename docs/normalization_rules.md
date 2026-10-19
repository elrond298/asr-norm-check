# Text normalisation rules

Two scoring variants are computed for every prediction:

| variant | what it does |
|---|---|
| `raw` | identity — reference and hypothesis are compared exactly as published/produced |
| `norm` | the language-specific pipeline below |

Both models are always scored with the **same** rules for a given language. `raw` is not a
"bad" rule that we fix with `norm`; it is the reference point that shows how much of a score
difference is formatting rather than recognition.

## Rules per language

Every rule below is applied identically to the reference and to the model output. Nothing is
applied only to one side.

### English (`en`)

Pipeline: strip zero-width characters → `EnglishTextNormalizer` (openai-whisper).

- Lower-case everything.
- Expand contractions and titles (`they're` → `they are`, `mr.` → `mister`).
- Map spelled-out numbers to digits (`five dollars` → `$5`, `nineteen ninety` → `1990`), so a
  word-vs-digit rendering of the same number is not counted as an error.
- Drop text in brackets/parentheses (FLEURS uses parentheses for non-spoken glosses).
- Remove punctuation and symbols; collapse whitespace.

### Hindi (`hi`)

Pipeline: NFKC → strip zero-width characters → drop bracketed glosses → lower-case → replace
symbols/punctuation with a space → fold Devanagari digits to ASCII → collapse whitespace.

- **Combining marks are preserved.** This is why openai-whisper's `BasicTextNormalizer` is *not*
  used for Hindi: it maps every Unicode M category character (matras, virama) to a space, which
  turns `मोटरसाइकिल` into a string of bare consonants and would inflate both WER and CER.
- Drop ZWJ (U+200D), ZWNJ (U+200C), ZWSP (U+200B), BOM (U+FEFF). Devanagari text uses these
  inconsistently and they are invisible in both reference and output.
- Drop parenthesised glosses such as `बोडा-बोडा (मोटरसाइकिल टैक्सी)`, which are not spoken.
- Fold Devanagari digits (U+0966–U+096F) to ASCII. NFKC alone does *not* do this; it is an
  explicit rule, because the same number rendered as `५०` or `50` is not a recognition error.
- Strip the danda (`।`) and other punctuation.
- **Not** done: transliteration, spelling variants, or nukta normalisation. Those change the
  token, not the formatting.

### Korean (`ko`)

Pipeline: NFKC → strip zero-width characters → drop bracketed glosses → lower-case → replace
symbols/punctuation with a space → collapse whitespace.

- NFKC folds full-width digits/punctuation to ASCII.
- Drop parenthesised glosses such as `비슈케크(Bishkek)` or `80 km (50 mi)`; the bracketed text
  is a romanisation or conversion that is not spoken.
- Lower-case (only affects Latin fragments inside Korean text).
- Strip punctuation. FLEURS Korean sentences end with `.` or `?` while models often omit it;
  punctuation carries no lexical content for WER.
- **Not** done: spacing normalisation between eojeol, or Hangul↔Hanja conversion. Korean
  spacing is lexical (it changes word boundaries), so rewriting it would hide real errors;
  instead CER is reported on space-free characters, where spacing has no effect.

## Metric definitions

- **WER** = (S + D + I) / N_ref, whitespace tokens, corpus-level (sum of errors / sum of
  reference tokens over the 50 clips of a language).
- **CER** = Levenshtein distance on space-free character strings / reference characters.
  For Hindi and Korean this is the meaningful metric, because a single wrong suffix or a
  spacing decision moves WER by a whole word.
- Per-sample WER/CER are reported too, but the tables use corpus-level ratios (macro vs
  micro: per-sample averaging would let a 4-word clip weigh as much as a 25-word one).

## Known limits

- Reusing openai-whisper's normalisers means the reference implementation of one of the two
  systems under test also defines the scoring rule. It is the de-facto standard in ASR work,
  but it is a choice — `raw` scores are reported alongside so a reader can discount it.
- Only the two variants above are computed. Intermediate variants (e.g. NFKC only, or
  case-folding only) are not, so the tables attribute the *joint* effect of the pipeline, not
  each rule's individual contribution.

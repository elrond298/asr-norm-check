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

### English (`en`)

Pipeline: NFKC → `EnglishTextNormalizer` (openai-whisper).

- Unicode NFKC (folds compatibility forms, e.g. `ﬁ` → `fi`, full-width → ASCII).
- Lower-case everything.
- Expand numbers and ordinals to words (`1990` → `nineteen ninety`, `3rd` → `third`) so a
  digit-vs-word difference is not counted as an error.
- Expand common contractions/abbreviations (`mr.` → `mister`, `they're` → `they are`).
- Remove punctuation and symbols; collapse whitespace.

### Hindi (`hi`)

Pipeline: NFKC → drop zero-width joiners → `BasicTextNormalizer` (openai-whisper).

- Unicode NFKC, which also maps Devanagari digits (U+0966–U+096F, `०१२…`) to ASCII digits.
  FLEURS Hindi references use Devanagari digits in some rows while models emit ASCII digits;
  without this rule the same number is scored as an error.
- Remove ZWJ (U+200D), ZWNJ (U+200C), ZWSP (U+200B), BOM (U+FEFF). Devanagari text uses
  these inconsistently, and they are invisible in both reference and output.
- Lower-case (no effect on Devanagari, but keeps the pipeline uniform), strip punctuation and
  symbols, collapse whitespace.
- **Not** done: transliteration, spelling variants, or nukta normalisation. Those change the
  token, not the formatting.

### Korean (`ko`)

Pipeline: NFKC → `BasicTextNormalizer` (openai-whisper).

- Unicode NFKC (full-width punctuation/digits → ASCII).
- Lower-case (affects Latin fragments inside Korean text, e.g. `Bishkek`).
- Strip punctuation and symbols, collapse whitespace. FLEURS Korean sentences end with `.` or
  `?` while models often omit it; punctuation carries no lexical content for WER.
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

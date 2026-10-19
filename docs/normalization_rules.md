# Text normalisation rules

Two scoring variants are computed for every prediction:

| variant | what it does |
|---|---|
| `raw` | identity — reference and hypothesis are compared exactly as published/produced |
| `norm` | the language-specific pipeline below |

Both models are always scored with the **same** rules for a given language. `raw` is not a
"bad" rule that we fix with `norm`; it is the reference point that shows how much of a score
difference is formatting rather than recognition.

## Bracketed text is kept, in every language

FLEURS references contain parenthetical material, e.g. `People's Organisation (SWAPO)`,
`비슈케크(Bishkek)`, `80 km (50 mi)`, `बोडा-बोडा (मोटरसाइकिल टैक्सी)`. Some of it is read aloud
by the speaker and some is only a written gloss — the audio decides, not the punctuation.

Measured on this subset (corpus WER, 22 of 150 references contain brackets):

| language | model | WER %, brackets kept (used here) | WER %, brackets deleted | difference (deleted − kept, pp) |
|---|---|---:|---:|---:|
| en | Whisper | 4.72% | 6.65% | +1.94 pp |
| en | Qwen3-ASR | 3.65% | 5.38% | +1.73 pp |
| hi | Whisper | 24.09% | 26.16% | +2.07 pp |
| hi | Qwen3-ASR | 11.19% | 13.00% | +1.80 pp |
| ko | Whisper | 16.77% | 14.95% | −1.83 pp |
| ko | Qwen3-ASR | 12.29% | 10.32% | −1.97 pp |

So neither policy is neutral: deleting brackets costs both models ~2 pp on English and Hindi
(consistent with those asides being read aloud — the models transcribe them), while keeping them
costs ~2 pp on Korean (consistent with those asides being written glosses the models do not say).
No listening audit was performed, so that reading of the data is an interpretation; the measured
fact is the four-way table above. This repo keeps the text, because then a model is charged for
what it says, and reports the Korean cost rather than hiding it. openai-whisper's English
normaliser deletes bracketed spans, so for English the brackets are rewritten to commas first,
which keeps the words.

## Rules per language

Every rule below is applied identically to the reference and to the model output. Nothing is
applied only to one side.

### English (`en`)

Pipeline: strip zero-width characters → `EnglishTextNormalizer` (openai-whisper).

- Lower-case everything.
- Expand contractions and titles (`they're` → `they are`, `mr.` → `mister`).
- Map spelled-out numbers to digits (`five dollars` → `$5`, `nineteen ninety` → `1990`), so a
  word-vs-digit rendering of the same number is not counted as an error; also fold British to
  American spellings (`organisation` → `organization`).
- Keep the words inside brackets: `(`, `)`, `[`, `]` are rewritten to commas before the
  normaliser runs, so its bracket-deletion rule never fires.
- Remove punctuation and symbols; collapse whitespace.

### Hindi (`hi`)

Pipeline: NFKC → strip zero-width characters → lower-case → replace symbols/punctuation
(including brackets) with a space → fold Devanagari digits to ASCII → collapse whitespace.

- **Combining marks are preserved.** This is why openai-whisper's `BasicTextNormalizer` is *not*
  used for Hindi: it maps every Unicode M category character (matras, virama) to a space, which
  turns `मोटरसाइकिल` into a string of bare consonants and would inflate both WER and CER.
- Drop ZWJ (U+200D), ZWNJ (U+200C), ZWSP (U+200B), BOM (U+FEFF). Devanagari text uses these
  inconsistently and they are invisible in both reference and output.
- Brackets are punctuation like any other, so their characters become token boundaries while
  the words inside them stay: `बोडा-बोडा (मोटरसाइकिल)` → `बोडा बोडा मोटरसाइकिल`. Compound
  splitting inside the gloss therefore counts as words, exactly as in the rest of the sentence.
- Fold Devanagari digits (U+0966–U+096F) to ASCII. NFKC alone does *not* do this; it is an
  explicit rule, because the same number rendered as `५०` or `50` is not a recognition error.
- Strip the danda (`।`) and other punctuation.
- **Not** done: transliteration, spelling variants, or nukta normalisation. Those change the
  token, not the formatting.

### Korean (`ko`)

Pipeline: NFKC → strip zero-width characters → lower-case → replace symbols/punctuation
(including brackets) with a space → collapse whitespace.

- NFKC folds full-width digits/punctuation to ASCII.
- Brackets become token boundaries, their words stay: `비슈케크(Bishkek)` → `비슈케크 bishkek`.
  When the bracketed text is a conversion the models do not say (`80 km (50 mi)`), those
  extra tokens are charged to the model — the ~2 pp cost of the keep-brackets policy, measured
  above, partly cancelling the gains from punctuation folding. That is why Korean shows the
  smallest raw→norm change of the three languages (−1.88 to −2.52 pp).
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
- Corpus tables sum S/D/I and denominators over the 50 clips (a micro-average); per-clip WER is
  only in `outputs/per_sample_scores.csv` and `outputs/error_cases.csv`, where a 4-word clip
  weighs as much as a 25-word one. `scripts/analyze.py` prints corpus numbers.
- Zero-denominator convention (library edge case, not triggered by these 150 references): an
  empty reference scores 1.0 when the hypothesis is non-empty, otherwise 0.0.

## Known limits

- Reusing openai-whisper's normalisers means the reference implementation of one of the two
  systems under test also defines the scoring rule. It is the de-facto standard in ASR work,
  but it is a choice — `raw` scores are reported alongside so a reader can discount it.
- Only the two variants above are computed. Intermediate variants (e.g. NFKC only, or
  case-folding only) are not, so the tables attribute the *joint* effect of the pipeline, not
  each rule's individual contribution.

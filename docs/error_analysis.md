# Error analysis

All numbers are corpus-level over the 50 clips per language, using the normalised (`norm`) variant
unless stated otherwise. Per-clip numbers and the changed tokens are in
`outputs/error_cases.csv`; normalisation-sensitive clips are ranked in
`outputs/normalization_deltas.csv`. Every case below is traceable by `sample_id` to
`data/manifest.jsonl` (reference + audio) and `outputs/predictions_*.jsonl` (model output).

## 1. Error-type mix

Substitutions / deletions / insertions counted by jiwer on the normalised strings:

| model | lang | ref tokens | S | D | I | WER | CER |
|---|---|---:|---:|---:|---:|---:|---:|
| whisper | en | 1123 | 39 | 6 | 8 | 4.72% | 2.30% |
| whisper | hi | 1233 | 262 | 9 | 26 | 24.09% | 8.76% |
| whisper | ko | 781 | 68 | 55 | 8 | 16.77% | 4.84% |
| qwen3-asr | en | 1123 | 25 | 6 | 10 | 3.65% | 1.53% |
| qwen3-asr | hi | 1233 | 111 | 15 | 12 | 11.19% | 4.54% |
| qwen3-asr | ko | 781 | 50 | 38 | 8 | 12.29% | 4.20% |

Three things stand out:

1. **Substitutions dominate everywhere** (52–88% of errors). Neither model is losing words; they
   are choosing the wrong word or the wrong spelling of the right word.
2. **Hindi is a spelling problem, not a listening problem.** Whisper's 262 Hindi substitutions
   against 9 deletions show it transcribes roughly the right number of words but renders vowels
   and word boundaries differently (`स्वीकार` → `स्विकार`, `लोकप्रिय` → `लोग प्रिय`).
3. **Korean errors are the only place deletions matter** (40–42% of errors). That is the eojeol
   spacing/compound-boundary issue: `하원 의원` vs `하원의원` merges two reference tokens into one
   hypothesis token, which the WER alignment books as a deletion plus a substitution.

## 2. What normalisation changes, and what it does not

Corpus WER / CER, `raw` → `norm`:

| model | lang | WER raw | WER norm | Δ | CER raw | CER norm | Δ |
|---|---|---:|---:|---:|---:|---:|---:|
| whisper | en | 12.04% | 4.72% | −7.32 | 4.21% | 2.30% | −1.91 |
| whisper | hi | 32.38% | 24.09% | −8.29 | 10.98% | 8.76% | −2.21 |
| whisper | ko | 19.29% | 16.77% | −2.52 | 7.13% | 4.84% | −2.29 |
| qwen3-asr | en | 10.86% | 3.65% | −7.21 | 3.39% | 1.53% | −1.86 |
| qwen3-asr | hi | 18.84% | 11.19% | −7.65 | 6.38% | 4.54% | −1.84 |
| qwen3-asr | ko | 14.17% | 12.29% | −1.89 | 6.32% | 4.20% | −2.12 |

The Qwen3-ASR − Whisper gap in WER moves from **−7.05 pp (raw) to −6.56 pp (norm)** across all 150
clips. So the observed gap is *not* mostly formatting: normalisation removes 2–8 WER points from
each model (7–8 on English and Hindi, ~2 on Korean) and leaves the ranking, and roughly the margin,
intact.

The per-language gap does change character, though:

| language | gap raw (pp) | gap norm (pp) | comment |
|---|---:|---:|---|
| en | −1.18 | −1.07 | near parity; differences are a handful of proper nouns |
| hi | −13.54 | −12.90 | the gap is real recognition, not formatting |
| ko | −5.12 | −4.48 | about 0.6 pp of the gap was punctuation/spacing formatting |

Read the deltas as *scoring rule sensitivity*, not as model improvement: the audio and the
hypotheses are identical in both columns.

## 3. Cases

17 clips, chosen because each one isolates a mechanism. "Whisper"/"Qwen" show the raw output;
the error type is the dominant one after normalisation.

### English

**1. `fleurs_en_1885` — number formatting survives normalisation (artifact)**
Reference: *Between 10:00-11:00 pm MDT, a fire was started…*
Whisper: *between 10 to 11 p.m. MDT a fire was started…* — Qwen: same shape.
Type: substitution (`0 → to`, `0 → p`). The English normaliser turns the reference `10:00` into
tokens `10 0`, so both models' perfectly reasonable *10 to 11 p.m.* is charged two errors. This is
the clearest example of a normaliser that does not only help — the reference's clock notation, not
the model output, is the outlier.

**2. `fleurs_en_1922` — homophone substitution vs. deletion + insertion**
Reference: *There are of course Christian theological explanations…*
Whisper: *There are coarse Christian…* → substitution (`of course` → `coarse`), a lexical error
with no acoustic excuse beyond the homophone.
Qwen: *There are Christian… it may be a well, be a pre-Christian ritual* → deletion of `of course`
plus insertion of `be a`, i.e. it restarted the clause. Two different failure modes on one clip.

**3. `fleurs_en_1906` — insertion without a listening error**
Reference: *This is the closing of one chapter…*
Both models: *This is going to be the closing of one chapter…*
Type: insertion (4 tokens). The pipeline is not mis-hearing; it is completing the pattern started
in the previous sentence (*This is not going to be goodbye*). This is the WER penalty that LM-style
decoding buys for fluency.

**4. `fleurs_en_1760` — one proper noun, twice**
Reference: *Mosasaurus … other mosasaurs.*
Whisper: *Mosoceros … other Mosoceros.* — Qwen: exact.
Type: substitution ×2. 13% WER on this clip from a single rare dinosaur name, in both of its
occurrences. Rare-term coverage is where the two models differ most in English.

**5. `fleurs_en_1822` — transliterating a non-English proper noun**
Reference: *…Daru l-Badya…*
Whisper: *Daru Ibadia* — Qwen: *Daru Ibadiya*.
Type: substitution. Neither model can render the Arabic article split as `l-`; both glue it into
the name. Not fixable by normalisation — it is a genuine lexical gap.

**6. `fleurs_en_1830` — compound spelling plus a normalisation artifact**
Reference: *South West Africa People's Organisation (SWAPO)…*
Whisper: *Southwest Africa People's Organization, SWAPO…* — Qwen: same but keeps `(SWAPO)`.
Type: substitution (`south west` → `southwest`). Both models keep the spoken gloss `SWAPO`, which
the reference nests in parentheses — the argument for not deleting bracketed text.
Note what normalisation already absorbs on this clip: `Organisation`/`Organization` and
`People's`/`people is` are folded identically on both sides (whisper's spelling normaliser maps
British to American spellings), so only the compound spelling is charged. Everything the
normaliser folds, it folds for the reference too.

### Hindi

**7. `fleurs_hi_1774` — orthographic variants, not recognition failures**
Manually compared on this clip: Whisper 44% WER, Qwen 6%.
Whisper's errors are `बाहरी → बहारी`, `द्वीपों → द्वीपो` (vowel length / final matra dropped),
`स्वीकार → स्विकार`, `किए → किये`; Qwen's remaining errors are `जाएँगे → जाएंगे` (chandrabindu) and
`हालाँकि → हालांकि` (candrabindu).
Type: substitution throughout. Both outputs are readable Hindi; a native reader would call several
of them spelling conventions rather than misrecognitions. Devanagari needs a spelling-variant
normaliser (nukta/chandrabindu folding) to separate the two — this pipeline deliberately does not
do that (see `docs/normalization_rules.md`), so those errors are counted.

**8. `fleurs_hi_1748` — digits, and a spacing merge**
Reference: *11:35 बजे अग्नि बचाव दल ने अंततः आग बुझा दी थी.*
Whisper: *11.45 … अगुनी बचावदल ने अंततह आग बजादी थी* — Qwen: *11:45 … अग्नि बचाव दल ने अंततः आग बुझा दी थी।*
Type: substitution. Both models misread the number (35 → 45). Qwen is otherwise exact and still
shows 8% WER; Whisper additionally merges `बचाव दल` into `बचावदल`, which costs one substitution
and one deletion even though a Hindi reader sees the same two words.

**9. `fleurs_hi_1832` — compound segmentation decides the score**
Reference: *…फ़ुटबॉल, बास्केटबॉल, वॉलीबॉल, वाटर पोलो, तलवारबाज़ी, रग्बी, साइकिलिंग, आइस हॉकी…*
Whisper: *…फूट बॉल, बास्केट बॉल, वॉली बॉल, वाटर पॉलो, तलवार बाजी, रब्बी, … आईस होकी…* (91% WER)
Qwen: keeps the compounds but spells them its own way (*फुटबॉल* without nukta, *आइसहॉकी* closed up).
Type: substitution, but the dominant mechanism is word segmentation. Whisper splits every compound
into two tokens where the reference has one, so one segmentation policy turns into six errors;
Qwen closes *आइस हॉकी* and *मोटर रेसिंग* where the reference keeps them open, and pays for that.
On Hindi, WER is heavily a segmentation convention; CER is the fairer number here.

**10. `fleurs_hi_1728` — deleting a spoken gloss**
Reference: *…आधारभूत (क्षारीय) रसायन…*
Whisper: *…आधार गूत रसायन…* — Qwen: *…आधारभूत रसायन…* (drops `क्षारीय`).
Type: substitution (Whisper) vs deletion (Qwen). Qwen is cleaner overall (14% vs 36% WER) but here
it drops the parenthetical word entirely, while Whisper garbles it. Shows why keeping bracketed
text in the reference is the right default for Hindi: the aside is spoken.

**11. `fleurs_hi_1915` — loanwords and nukta**
Reference: *…बोडा-बोडा (मोटरसाइकिल टैक्सी) ले सकते हैं…कांगोलीज़ फ़्रैंक…*
Whisper: *…बोड़ा बोड़ा मोटर साइकल…* — Qwen: *…मोटरसाइकल… कागोलिस फ्रैंक…*
Type: substitution. `मोटरसाइकिल` (with the final ि) is reduced to `मोटरसाइकल` by both models, and
the nukta in `फ़्रैंक` is dropped by Qwen but kept by Whisper. Loanword orthography is the single
most persistent Hindi error family after compounds.

### Korean

**12. `fleurs_ko_1744` — Latin gloss inside the reference**
Reference: *…로스비 수(Rossby number)가 작을수록…*
Both models: *로스비수/로스비스가* — neither says *Rossby number*.
Type: deletion of the Latin gloss (4 reference tokens). Both models are right about the audio; the
reference contains a written gloss the speaker does not read. This is the Korean side of the
bracket question: keeping brackets helps English and Hindi but costs Korean about 1.9 pp WER.

**13. `fleurs_ko_1962` — gloss plus eojeol spacing**
Reference: *…화학 물질에서 수소(pH에서 H) 이온의 양으로…*
Whisper: *…화학물질에서 수소이온의 양으로…* — Qwen: *…화학 물질에서 수소 이온의 양으로…*
Type: Whisper deletes the gloss and closes both compounds (2 substitutions + 4 deletions, 50% WER);
Qwen also drops the gloss but keeps the spacing exactly (2 deletions, 17% WER). Pure spacing policy
costs Whisper a third of this clip.

**14. `fleurs_ko_1893` — units and a dropped conversion**
Reference: *…80 km (50 mi)에 이르는…*
Both models: *…80km에 이르는…*
Type: substitution `80 → 80km에` + deletion `km`, `50`, `mi`, `에`. Identical output from both
models, 35% WER each: they say the right thing (*80 km*) and are penalised for a bracketed unit
conversion nobody speaks plus one spacing decision.

**15. `fleurs_ko_1929` — reading a digit as a numeral**
Reference: *…2세트에서 먼저 어드밴티지를…*
Whisper: *2세트 … 어드벤티즈를* — Qwen: *이 세트 … 어드밴티지를*.
Type: substitution. Qwen reads `2` as the Sino-Korean numeral *이* instead of *2*, and that one
decision costs two tokens (the digit plus the spacing that follows). Whisper keeps the digit but
alters `어드밴티지` → `어드벤티즈`.

**16. `fleurs_ko_1776` — spelled numerals are not folded (scoring gap)**
Reference: *…인쇄물이 150에서 200장 만들어졌다.*
Whisper: *…백오십에서 이백장 만들어졌다.* — Qwen: *150에서 200장…*
Type: substitution. Whisper said the numbers correctly in words; the reference uses digits. Unlike
English, this pipeline has no Korean numeral-word fold, so 2 of ~12 tokens are charged. A
Korean-aware normaliser would repair this; the gap is a limitation of the rule set, not of Whisper.

**17. `fleurs_ko_1708` — compound noun + particle**
Reference: *…에드 데이비 하원 의원에 의해… 노먼 램 하원 의원은…*
Both models: *…하원의원에/하원의원은…*
Type: token-level errors from one spacing decision, repeated twice — each merge is charged
against the reference tokens on either side of the boundary.
`하원 의원` (assembly member, two eojeol) vs `하원의원` (one) is a legitimate variation in Korean
orthography, and WER cannot tell it apart from missing a word.

## 4. Categories we deliberately did not add

- **Named entities / numbers / code-switching** were not turned into frequency tables. With 50
  clips per language the per-category counts are 0–8 (proper nouns) and 1–6 (digits), so a category
  rate would be noise dressed as a percentage. The case table covers the mechanisms instead.
- **Code-switching**: the subset contains Latin fragments only as names, units and loanwords
  (`Rossby number`, `SWAPO`, `F-1`); FLEURS is read Wikipedia prose and has no true intra-sentential
  Spanish/English-style switching. Nothing here supports conclusions about code-switched speech.
- **Reference errors** were not audited by re-listening to audio. Cases 12/14 show the references
  themselves carry unspoken glosses, which is enough to say the WER numbers have a
  reference-quality floor — not enough to correct individual references.

## 5. Limitations of these conclusions

- 50 clips per language: a 1–2 pp difference (all of English) is inside the sampling noise; the
  Hindi and Korean gaps are far larger than the noise floor and are the only differences worth
  ranking on.
- Read speech, single speaker per clip, studio recording: no noise, no overlap, no disfluency.
  Insertion errors seen here come from language-model fluency, not from bad acoustics.
- One decode per clip, no seed/hardware variance estimate.
- WER on Hindi and Korean is dominated by orthographic policy (segmentation, spacing, nukta). CER
  is reported alongside for exactly that reason, and the two metrics disagree in size — but they
  never disagree in ranking between these two models on this subset.

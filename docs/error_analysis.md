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

![Error composition](img/fig_error_mix.png)

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
| qwen3-asr | ko | 14.17% | 12.29% | −1.88 | 6.32% | 4.20% | −2.12 |

![WER before and after normalisation](img/fig_normalisation.png)

The Qwen3-ASR − Whisper gap in WER moves from **−7.05 pp (raw) to −6.57 pp (norm)** across all 150
clips (from unrounded rates; the rounded table in the README subtracts to 6.56). So the observed gap is *not* mostly formatting: normalisation removes 1.9–8.3 WER points
from each model (7–8 on English and Hindi, ~2 on Korean) and leaves the ranking, and roughly the
margin, intact.

The per-language gap does change character, though:

| language | gap raw (pp) | gap norm (pp) | comment |
|---|---:|---:|---|
| en | −1.18 | −1.07 | near parity; differences are a handful of proper nouns |
| hi | −13.54 | −12.90 | the gap is real recognition, not formatting |
| ko | −5.12 | −4.48 | about 0.6 pp of the gap was punctuation/spacing formatting |

Read the deltas as *scoring rule sensitivity*, not as model improvement: the audio and the
hypotheses are identical in both columns.

## 3. Cases at a glance

| # | sample_id | lang | mechanism | clip WER (norm) |
|---|---|---|---|---|
| 1 | `fleurs_en_1885` | en | clock notation → reference tokenised as `10 0` | 17.6% both |
| 2 | `fleurs_en_1922` | en | homophone substitution vs. clause restart | 9.1 / 18.2% |
| 3 | `fleurs_en_1906` | en | insertion from fluent completion | 14.3% both |
| 4 | `fleurs_en_1760` | en | rare proper noun, twice | 13.3 / 0.0% |
| 5 | `fleurs_en_1822` | en | transliterating `l-Badya` | 16.0 / 8.0% |
| 6 | `fleurs_en_1830` | en | compound spelling; contraction absorbed | 11.1% both |
| 7 | `fleurs_hi_1774` | hi | orthographic variants (vowel length, chandrabindu) | 44.4 / 5.6% |
| 8 | `fleurs_hi_1748` | hi | one digit + one spacing merge | 58.3 / 8.3% |
| 9 | `fleurs_hi_1832` | hi | compound segmentation, both directions | 90.9 / 59.1% |
| 10 | `fleurs_hi_1728` | hi | spoken aside dropped by the better model | 35.7 / 14.3% |
| 11 | `fleurs_hi_1915` | hi | loanwords, nukta, shared lexical error | 36.7 / 20.0% |
| 12 | `fleurs_ko_1744` | ko | Latin gloss `(Rossby number)` | 53.8% both |
| 13 | `fleurs_ko_1962` | ko | gloss + eojeol spacing | 50.0 / 16.7% |
| 14 | `fleurs_ko_1893` | ko | unit conversion in brackets | 35.3% both |
| 15 | `fleurs_ko_1929` | ko | digit ↔ numeral, compound closing | 40.0 / 26.7% |
| 16 | `fleurs_ko_1776` | ko | spoken numerals not folded | 58.3 / 25.0% |
| 17 | `fleurs_ko_1708` | ko | the clip where Qwen3-ASR is worse | 33.3 / 38.1% |

WER column is Whisper / Qwen3-ASR. Every number is in `outputs/error_cases.csv`.

## 4. Cases, one by one

17 clips, chosen because each one isolates a mechanism. Counts are S/D/I from jiwer on the
normalised strings; the word-level colouring in the viewer uses difflib and may pair tokens
differently on a single clip. "WER" is that clip's own per-sample WER.

### English

**1. `fleurs_en_1885` — clock notation turns a correct reading into three errors**
Reference: *Between 10:00-11:00 pm MDT, a fire was started by the inmates in the yard.*
Both models: *between 10 to 11 p.m. MDT a fire was started by the inmates in the yard*
After normalisation the reference reads `between 10 0 11 0 pm mdt …` while the output reads
`between 10 to 11 p m mdt …`: **3 S** (`0→to`, `0→p`, `pm→m`), 17.6% WER each.
The models heard *10 to 11*; the reference's `10:00` is tokenised into `10` + `0`. Any rule set
that expands clock times would remove this; this one does not, so the score is a formatting
artifact, not a listening error.

**2. `fleurs_en_1922` — one clip, two different failure modes**
Reference: *There are of course Christian theological explanations for this tradition…*
Whisper: *There are coarse Christian…* → **1 S + 1 D** (`of→coarse`, `course` dropped), 9.1%.
Qwen3-ASR: *There are Christian… it may be a well, be a pre-Christian ritual* → **2 D + 2 I**
(`of course` dropped, `be a` inserted), 18.2%.
Whisper mis-hears a homophone; Qwen restarts the clause. Same audio, same score shape, different
bug.

**3. `fleurs_en_1906` — insertion without a listening error**
Reference: *This is the closing of one chapter…*
Both models: *This is going to be the closing of one chapter…* → **3 I**, 14.3% each.
The pattern from the previous sentence (*This is not going to be goodbye*) is completed by both
models. This is the fluency penalty: decoders that model language add words that were not said.

**4. `fleurs_en_1760` — one proper noun, twice**
Reference: *Mosasaurus … other mosasaurs.*
Whisper: *Mosoceros … other Mosoceros* → **2 S**, 13.3%. Qwen3-ASR reproduces the name exactly,
0%.
Rare-word coverage is where the two models differ most in English; both occurrences of the same
name are charged.

**5. `fleurs_en_1822` — transliterating a non-English proper noun**
Reference: *…Daru l-Badya … traders who established trading bases there.*
Whisper: *Daru Ibadia … traders who were establishing trading bases there* → **2 S + 1 D + 1 I**
(`l→ibadia`, `established→were`, `badya` dropped, `establishing` inserted), 16.0%.
Qwen3-ASR: *Daru Ibadiya, … traders who established trading bases there* → **1 S + 1 D**
(`l→ibadiya`, `badya` dropped), 8.0%.
Neither model can render the Arabic article split as `l-` plus the name; both glue it into one
token. Normalisation cannot repair this — it is a lexical gap, not formatting.

**6. `fleurs_en_1830` — compound spelling, and what the normaliser already absorbs**
Reference: *…South West Africa People's Organisation (SWAPO)…*
Whisper: *…Southwest Africa People's Organization, SWAPO…*; Qwen3-ASR keeps the parentheses:
*…Southwest Africa People's Organization (SWAPO)…*. Both → **1 S + 1 D** (`south→southwest`,
`west` dropped), 11.1% each — the only difference between them is the brackets their output has
and the normaliser removes from both.
Two things normalisation already absorbs identically on both sides: `Organisation` → `organization`
and the contraction `People's` → `people is`. Only the compound spelling is charged. The spoken
gloss `SWAPO` matches the reference because brackets are kept (see
`docs/normalization_rules.md`).

### Hindi

**7. `fleurs_hi_1774` — orthographic variants dominate, not recognition**
Whisper **15 S + 1 I**, 44.4% WER; Qwen3-ASR **2 S**, 5.6%.
Whisper's substitutions include `बाहरी→बहारी` (vowel length), `द्वीपों→द्वीपो` (final matra),
`स्वीकार→स्विकार`, `किए→किये`, and heavier damage on words it heard less well
(`हालाँकि→हाला`, `ब्रिटिश→प्रितिश`, `संयुक्त→सैयोक्त`, `निर्धारित→नरधारत`).
Qwen3-ASR's only two errors are anusvara/chandrabindu variants: `जाएँगे→जाएंगे`,
`हालाँकि→हालांकि`.
Both outputs are readable Hindi. Separating spelling convention from misrecognition needs a
Devanagari variant normaliser (nukta, chandrabindu), which this pipeline deliberately does not
apply — so all of it is charged.

**8. `fleurs_hi_1748` — a digit, and a spacing merge**
Reference: *11:35 बजे अग्नि बचाव दल ने अंततः आग बुझा दी थी.*
Whisper: *11.45 बजे अगुनी बचावदल ने अंततह आग बजादी थी* → **5 S + 2 D**, 58.3%.
Qwen3-ASR: *11:45 बजे अग्नि बचाव दल ने अंततः आग बुझा दी थी।* → **1 S**, 8.3%.
Both misread the number (35 → 45) — one error each. Whisper additionally merges `बचाव दल` into
`बचावदल`, which the alignment books as a substitution plus a deletion even though a reader sees
the same two words.

**9. `fleurs_hi_1832` — compound segmentation decides the score**
Reference: *…फ़ुटबॉल, बास्केटबॉल, वॉलीबॉल, वाटर पोलो, तलवारबाज़ी, रग्बी, साइकिलिंग, आइस हॉकी, रोलर हॉकी और एफ़ 1 मोटर रेसिंग…*
Whisper **14 S + 1 D + 5 I**, 90.9%: it splits compounds (`लोकप्रिय→लोग प्रिय`,
`फ़ुटबॉल→फूट बॉल`, `बास्केटबॉल→बास्केट बॉल`, `आइस हॉकी→आईस होकी`).
Qwen3-ASR **8 S + 5 D**, 59.1%: it closes compounds (`वाटर पोलो→वॉटरपॉलो`,
`आइस हॉकी→आइसहॉकी`, `मोटर रेसिंग→मोटररेसिंग`), and each merge costs one substitution plus one
deletion.
Neither is a listening failure; the two models disagree about where Hindi word boundaries are, and
WER charges both. CER is the fairer number on this clip.

**10. `fleurs_hi_1728` — deleting a spoken aside**
Reference: *…आधारभूत (क्षारीय) रसायन…*
Whisper: *…आधार गूत रसायन…* → **5 S**, 35.7%.
Qwen3-ASR: *…आधारभूत रसायन…* → **1 S + 1 D** (`अम्लीय→अमलिया`, `क्षारीय` dropped), 14.3%.
With brackets kept, the aside is part of the reference; Qwen's cleaner sentence still pays for
omitting it. This is the case that shows the keep-brackets policy is not free for Hindi either.

**11. `fleurs_hi_1915` — loanwords, nukta, and a shared lexical error**
Reference: *…बोडा-बोडा (मोटरसाइकिल टैक्सी)… छोटी सैर… 500 कांगोलीज़ फ़्रैंक…*
Whisper **9 S + 1 D + 1 I**, 36.7%: `बोडा→बोड़ा`, `मोटरसाइकिल→मोटर साइकल`, `सैर→सहर`,
`कांगोलीज़→कागोलीस`, `फ़्रैंक→फ्रेंक`.
Qwen3-ASR **5 S + 1 D**, 20.0%: `मोटरसाइकिल→मोटरसाइकल`, `सैर→शहर`, `कांगोलीज़→कागोलिस`,
`फ़्रैंक→फ्रैंक`.
Both drop the nukta in `फ़्रैंक`; whisper also changes the vowel. `सैर→शहर`/`सहर` is a genuine
lexical error shared by both. The spoken gloss `मोटरसाइकिल` is transcribed by both models (as
`मोटर साइकल` / `मोटरसाइकल`), so keeping brackets did not punish them here — unlike the Korean
cases below.

### Korean

**12. `fleurs_ko_1744` — a Latin gloss the models do not say**
Reference: *…로스비 수(Rossby number)가 작을수록…*
Both models say *로스비수/로스비스가*, i.e. they merge `로스비 수` and never say *Rossby number*:
**2 S + 5 D** each, 53.8%.
The two deletions that matter are `rossby` and `number` — tokens the reference contains because
of the bracket; neither model says them.

**13. `fleurs_ko_1962` — gloss plus eojeol spacing**
Reference: *…화학 물질에서 수소(pH에서 H) 이온의 양으로…*
Whisper: *…화학물질에서 수소이온의 양으로…* → **2 S + 4 D**, 50.0% (closes both compounds and
drops the gloss).
Qwen3-ASR: *…화학 물질에서 수소 이온의 양으로…* → **0 S + 2 D**, 16.7% (keeps the spacing exactly,
loses only `pH에서 H`).
Identical audio; the 33-point gap is entirely spacing policy plus the gloss.

**14. `fleurs_ko_1893` — units and a dropped conversion**
Reference: *…80 km (50 mi)에 이르는…*
Both models: *…80km에 이르는…* → **2 S + 4 D** each, 35.3% (`순다르반은→순다르바는`,
`80→80km에`, then `km`, `50`, `mi`, `에` deleted).
They say the right thing and are charged for a unit conversion in brackets plus one spacing
decision. Identical output from both models, identical score.

**15. `fleurs_ko_1929` — reading a digit as a numeral**
Reference: *…2세트에서 먼저 어드밴티지를…*
Whisper **3 S + 3 D**, 40.0%: `어드밴티지를→어드벤티즈를`, and the compounds
`6 대 6→6대6`, `타이 브레이크→타이브레이크` are closed, each merge charged twice.
Qwen3-ASR **2 S + 1 D + 1 I**, 26.7%: it reads `2` as the Sino-Korean numeral `이`
(`2세트에서→이`), inserting `세트에서` as a separate token.
One model imposes digit→word, the other word→digit; both pay.

**16. `fleurs_ko_1776` — spelled numerals are not folded (scoring gap)**
Reference: *…인쇄물이 150에서 200장 만들어졌다.*
Whisper: *…백오십에서 이백장 만들어졌다* → **5 S + 2 D**, 58.3%, of which two substitutions
(`150→백오십`, `200→이백`) are the number words, plus `밤 동안→밤동안` and `던랩→덜렙`.
Qwen3-ASR: *…150에서 200장…* → **2 S + 1 D**, 25.0% (the name `던랩→델랩` and the particle
`라고` swallowed into `브로드사이드라고`).
Whisper said the numbers correctly, in words. English gets a digit-fold rule; Korean does not, so
this is a limitation of the rule set that happens to favour the digit-emitting model.

**17. `fleurs_ko_1708` — where Qwen3-ASR is *worse* on a clip**
Reference: *…에드 데이비 하원 의원에 의해… 노먼 램 하원 의원은… 밭을 것으로 예산된다.*
Whisper **5 S + 2 D**, 33.3%; Qwen3-ASR **6 S + 2 D**, 38.1%.
Both rename the person to `후는` and merge `하원 의원→하원의원` twice (each merge = 1 S + 1 D).
Qwen3-ASR additionally writes `에드→애드`, which is the one extra substitution.
Both also "correct" the reference's obvious typos (`밭을→받을`, `예산된다→예상된다`) and are charged
for it — a reminder that reference text, not model output, sets the ceiling.

## 5. Categories we deliberately did not add

- **Named entities / numbers / code-switching** were not turned into frequency tables. With 50
  clips per language the per-category counts are 0–8 (proper nouns) and 1–6 (digits), so a category
  rate would be noise dressed as a percentage. The case table covers the mechanisms instead.
- **Code-switching**: the subset contains Latin fragments only as names, units and loanwords
  (`Rossby number`, `SWAPO`, `F-1`); FLEURS is read Wikipedia prose and has no true intra-sentential
  Spanish/English-style switching. Nothing here supports conclusions about code-switched speech.
- **Reference errors** were not audited by re-listening to audio. Cases 12/14 show references that
  carry bracketed glosses neither model transcribes; whether the speaker says them is unknown
  without listening. That is enough to say the WER numbers have a reference-quality floor — not
  enough to correct individual references.

## 6. Limitations of these conclusions

- **No uncertainty analysis was run** — no bootstrap, no repeated decodes, no significance test.
  The numbers are 150 point estimates. The English gap (about 1 pp) is small enough that this
  subset cannot support a ranking claim; the Hindi gap (≈13 pp) is far larger than any plausible
  sampling effect, but "far larger than plausible" is still an argument, not a computed interval.
- Read speech, one speaker per clip, studio-style recordings: no noise, no overlap, no
  disfluency. The insertion cases (3, and the merges in 15) are consistent with LM-style decoding
  completing familiar patterns, but that is an interpretation — no listening audit was performed,
  and the transcripts alone cannot separate "model added a word" from "speaker said it quickly".
- One decode per clip, no seed/hardware variance estimate. Decoding is greedy/beam at temperature
  0, so a re-run is expected to be identical; that was spot-checked (10 clips per model), not
  proven for all 150.
- WER on Hindi and Korean is dominated by orthographic policy (segmentation, spacing, nukta). CER
  is reported alongside for exactly that reason, and the two metrics disagree in size — but they
  never disagree in ranking between these two models on this subset.

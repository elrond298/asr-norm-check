# asr-bench — Whisper vs Qwen3-ASR on a frozen FLEURS subset

A small, reproducible multilingual ASR benchmark and error-analysis tool: 150 fixed clips of
FLEURS (English / Hindi / Korean, 50 each), two models run through their official interfaces,
one scoring rule, a raw-vs-normalised comparison, and a Gradio viewer for the results.

## Scope (frozen)

| | |
|---|---|
| Languages | English (`en_us`), Hindi (`hi_in`), Korean (`ko_kr`) |
| Clips | 150 total — 50 per language, fixed list in `data/manifest.jsonl` |
| Data | FLEURS `test` split, dataset revision `70bb2e84…` (pinned), CC-BY-4.0 |
| Models | `openai/whisper-large-v3` (openai-whisper) and `Qwen/Qwen3-ASR-1.7B-hf` (Transformers) |
| Metrics | corpus WER (whitespace tokens) and CER (space-free characters), per language |
| Scoring variants | `raw` (as produced) and `norm` (per-language normalisation) |
| Language mode | forced per clip for both models, so the comparison is like-for-like |

**Not in scope** (deliberately): no model training or fine-tuning, no audio resampling or
denoising experiments, no streaming, no translation, no extra languages, no test-set sampling
to "balance" anything, no LLM-based semantic scoring, no hyper-parameter search.

## Results

WER / CER in %, corpus-level over the 50 clips of each language (150 total). Full tables with S/D/I counts and per-clip
scores are in `outputs/`; error mechanisms are analysed case by case in
[`docs/error_analysis.md`](docs/error_analysis.md).

**After normalisation (`norm` variant, the headline numbers):**

| language | Whisper WER | Qwen3-ASR WER | Whisper CER | Qwen3-ASR CER |
|---|---:|---:|---:|---:|
| en (50) | 4.72% | **3.65%** | 2.30% | **1.53%** |
| hi (50) | 24.09% | **11.19%** | 8.76% | **4.54%** |
| ko (50) | 16.77% | **12.29%** | 4.84% | **4.20%** |
| all (150) | 15.33% | **8.77%** | 5.32% | **3.21%** |

**Effect of normalisation (same audio, same hypotheses, different scoring rule):**

| language | Whisper WER raw → norm | Qwen3-ASR WER raw → norm | Whisper CER raw → norm | Qwen3-ASR CER raw → norm |
|---|---|---|---|---|
| en | 12.04% → 4.72% | 10.86% → 3.65% | 4.21% → 2.30% | 3.39% → 1.53% |
| hi | 32.38% → 24.09% | 18.84% → 11.19% | 10.98% → 8.76% | 6.38% → 4.54% |
| ko | 19.29% → 16.77% | 14.17% → 12.29% | 7.13% → 4.84% | 6.32% → 4.20% |
| all | 21.89% → 15.33% | 14.84% → 8.77% | 7.41% → 5.32% | 5.11% → 3.21% |

**What survives normalisation:** the Qwen3-ASR − Whisper WER gap is −7.05 pp raw and −6.56 pp
normalised. Both models lose 2–8 WER points to formatting, so the gap is mostly recognition
quality rather than formatting, and the ranking does not depend on the scoring variant. The
per-language gaps (norm, pp): en −1.07, hi −12.90, ko −4.48 — only the Hindi and Korean gaps are
larger than what 50 clips can resolve.

**Two findings worth reading the details for:**
1. The stock English normaliser deletes text in parentheses, and in FLEURS those asides are
   *spoken* — that alone overcharges both models by ~1.9 pp WER (see
   `docs/normalization_rules.md`); this repo keeps them.
2. On Hindi and Korean, WER is dominated by segmentation/spacing policy (compounds, eojeol,
   nukta), not by listening. CER is reported for that reason; both metrics agree on the ranking.


## Demo

`uv run python app/gradio_app.py` → http://127.0.0.1:7860 (no model loading; reads `outputs/` only).

| Metrics tab | Samples tab |
|---|---|
| ![metrics tab](docs/img/app_metrics.png) | ![samples tab](docs/img/app_samples.png) |

The Metrics tab shows the per-language table for the selected scoring variant plus the raw-vs-norm
comparison; the Samples tab plays the clip, shows both models' scores for that `sample_id`, and
highlights insertions/deletions/substitutions at word level.

## Repository layout

```
asrbench/            normalisation + scoring library (imported by scripts and app)
scripts/             fetch_fleurs.py, run_whisper.py, run_qwen.py, score.py, analyze.py
app/gradio_app.py    Gradio viewer over the precomputed outputs
data/manifest.jsonl  frozen sample list (id, language, reference, audio path, sha256)
data/manifest.meta.json  selection rule, dataset revision, counts, manifest hash
outputs/             predictions_*.jsonl, metrics_*.csv, per_sample_scores.csv, error_cases.csv
docs/                normalization_rules.md, error_analysis.md
```

## Reproduce

Requires Python 3.12, [uv](https://docs.astral.sh/uv/), ffmpeg, and a CUDA GPU (~8 GB is
enough for both models). Wall time on an RTX 4060 Laptop 8 GB: audio fetch ~2 min,
Whisper ~38 min (0.8× realtime), Qwen3-ASR ~8 min (3.8× realtime), scoring and analysis
seconds. The Gradio app needs no GPU.

```bash
uv sync                                   # creates .venv from uv.lock
uv run python scripts/fetch_fleurs.py     # -> data/audio/**, data/manifest.jsonl
uv run python scripts/run_whisper.py      # -> outputs/predictions_whisper.jsonl
uv run python scripts/run_qwen.py         # -> outputs/predictions_qwen.jsonl
uv run python scripts/score.py            # -> outputs/metrics_*.csv, per_sample_scores.csv
uv run python scripts/analyze.py          # -> outputs/error_cases.csv, normalization_deltas.csv
uv run python app/gradio_app.py           # http://127.0.0.1:7860
```

No GPU? The committed `outputs/` directory contains every result file, so after `uv sync` the
last command alone is enough to browse the tables and samples without re-running inference.

The sample list is pinned twice over: the dataset revision in `scripts/fetch_fleurs.py`, and a
sha256 per audio file in `data/manifest.jsonl`. To confirm your download is identical to the one
scored here:

```bash
uv run python scripts/fetch_fleurs.py --verify
```

The Gradio app reads only precomputed files — it never loads a model.

Reproducibility was checked directly: re-running the first 10 clips with the same commands gave
byte-identical `model_output` for both models (10/10 Whisper, 10/10 Qwen3-ASR), and
`outputs/alignment_check.md` verifies that both prediction files cover exactly the manifest ids.

## Model versions and inference settings

| | Whisper | Qwen3-ASR |
|---|---|---|
| Weights | `large-v3` (openai-whisper, azure CDN) | `Qwen/Qwen3-ASR-1.7B-hf` (HF) |
| Code | `openai-whisper` (see `uv.lock`) | `transformers` (see `uv.lock`) |
| Decoding | beam search, beam 5, `temperature=0.0`, `condition_on_previous_text=False` | greedy (`do_sample=False`), `max_new_tokens=256` |
| Precision | fp16 | bfloat16 |
| Language | forced (`en`/`hi`/`ko`) | forced (`en`/`hi`/`ko`) |

Exact strings are recorded per prediction file in `model_version` and `inference_config`.

## What each file contains

- `outputs/predictions_{whisper,qwen}.jsonl` — one row per clip:
  `sample_id, language, reference, model_output, model, model_version, inference_config,
  audio_path, duration_s, runtime_s`.
- `outputs/metrics_baseline.csv` — `raw` variant, per model × language (+ `all`), WER/CER and
  the underlying S/D/I counts.
- `outputs/metrics_normalized.csv` — same for the `norm` variant.
- `outputs/per_sample_scores.csv` — per model × clip × variant: WER, CER, S/D/I, lengths.
- `outputs/error_cases.csv` — per clip: raw/norm scores, dominant error type, changed tokens,
  reference and hypothesis.
- `outputs/normalization_deltas.csv` — clips whose score moves most under normalisation.
- `outputs/alignment_check.md` — proof that both models cover the same sample ids, same field
  schema, and that references match the frozen manifest.

## Scoring

- WER = (substitutions + deletions + insertions) / reference tokens, corpus-level.
- CER = Levenshtein distance on space-free characters / reference characters.
- Normalisation: NFKC, case folding, punctuation removal, number expansion (English), Devanagari
  digit folding and zero-width-joiner removal (Hindi) — written up rule by rule in
  `docs/normalization_rules.md`.

## Known limits

- **Small sample.** 50 clips per language cannot rank two ASR systems in general; differences of
  a few tenths of a point are noise at this size. The numbers are per-language point estimates.
- **Read speech only.** FLEURS is read-aloud Wikipedia text: no spontaneous speech, no
  overlapping speakers, clean studio-ish audio, so error types here are not representative of
  real deployments.
- **Code-switching.** The subset contains only incidental Latin fragments inside Hindi/Korean
  clips (names, English loanwords). Nothing here supports conclusions about code-switched speech.
- **Normalisation is a choice, not a truth.** The `norm` rules come from openai-whisper's
  normalisers; a different rule set moves both models' scores. Both variants are reported.
- **Single run, no repeats.** Each clip was decoded once; no variance estimate across seeds or
  hardware is reported (greedy/beam decoding at temperature 0 is deterministic up to
  floating-point reduction order).
- **Reference-quality unknown.** Reference texts were not manually re-listened. Indirect evidence
  that audio and reference belong together: both models reproduce the references closely (English
  WER 3.7–4.7%) and every clip is taken straight from FLEURS' own `test.tsv` row. FLEURS
  references carry occasional orthography choices that neither model is wrong about, and some
  contain bracketed glosses that are never spoken (see the Korean cases in
  `docs/error_analysis.md`).
- **Not done:** the optional language-weight sensitivity analysis (stage 9) and any
  statistical significance testing.

## Data licence

FLEURS is distributed under CC-BY-4.0 by Google. Audio files are not committed to this
repository (`data/audio/` is git-ignored); they are re-downloaded from the pinned revision and
verified against the sha256 hashes in `data/manifest.jsonl`.

#!/usr/bin/env python
"""Gradio viewer for the precomputed results (no model inference at runtime).

  uv run python app/gradio_app.py            # http://127.0.0.1:7860

Reads: data/manifest.jsonl, outputs/predictions_*.jsonl, outputs/metrics_*.csv,
       outputs/per_sample_scores.csv, data/audio/**
"""

from __future__ import annotations

import csv
import difflib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gradio as gr
import pandas as pd

from asrbench.common import OUT_DIR, ROOT, load_jsonl
from asrbench.normalize import normalize

MODELS = ["whisper", "qwen3-asr"]
LANGUAGES = ["en", "hi", "ko"]
DATA = ROOT / "data"

MANIFEST = {r["sample_id"]: r for r in load_jsonl(DATA / "manifest.jsonl")}
PREDICTION_FILES = {"whisper": "predictions_whisper.jsonl", "qwen3-asr": "predictions_qwen.jsonl"}
PREDS = {m: {r["sample_id"]: r for r in load_jsonl(OUT_DIR / name)}
         for m, name in PREDICTION_FILES.items()}
with (OUT_DIR / "per_sample_scores.csv").open(encoding="utf-8") as fh:
    PER_SAMPLE = {(r["model"], r["sample_id"], r["variant"]): r for r in csv.DictReader(fh)}
METRICS = pd.concat([
    pd.read_csv(OUT_DIR / "metrics_baseline.csv").assign(variant="raw"),
    pd.read_csv(OUT_DIR / "metrics_normalized.csv").assign(variant="norm"),
], ignore_index=True)

LABELS = {f"{sid}  ·  {row['reference'][:48]}": sid for sid, row in MANIFEST.items()}

_STYLES = {
    "del": "background:#ffd7d5;color:#8b1a10;text-decoration:line-through",
    "ins": "background:#d6f5d6;color:#14532d",
    "sub": "background:#ffe9b8;color:#7c4a03",
}


def diff_html(ref: str, hyp: str) -> str:
    """Word-level reference/hypothesis view with insertions, deletions and substitutions marked."""
    a, b = ref.split(), hyp.split()
    ref_parts, hyp_parts = [], []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b).get_opcodes():
        ref_chunk, hyp_chunk = " ".join(a[i1:i2]), " ".join(b[j1:j2])
        if tag == "equal":
            ref_parts.append(ref_chunk)
            hyp_parts.append(hyp_chunk)
        elif tag == "replace":
            # pair what can be paired; a surplus token on either side is a deletion/insertion
            paired = min(i2 - i1, j2 - j1)
            paired_ref, paired_hyp = " ".join(a[i1:i1 + paired]), " ".join(b[j1:j1 + paired])
            surplus_ref, surplus_hyp = " ".join(a[i1 + paired:i2]), " ".join(b[j1 + paired:j2])
            chunk = []
            if paired_ref:
                chunk.append(f'<span style="{_STYLES["sub"]}">{paired_ref}</span>')
            if surplus_ref:
                chunk.append(f'<span style="{_STYLES["del"]}">{surplus_ref}</span>')
            ref_parts.append(" ".join(chunk))
            chunk = []
            if paired_hyp:
                chunk.append(f'<span style="{_STYLES["sub"]}">{paired_hyp}</span>')
            if surplus_hyp:
                chunk.append(f'<span style="{_STYLES["ins"]}">{surplus_hyp}</span>')
            hyp_parts.append(" ".join(chunk))
        elif tag == "delete":
            ref_parts.append(f'<span style="{_STYLES["del"]}">{ref_chunk}</span>')
        elif tag == "insert":
            hyp_parts.append(f'<span style="{_STYLES["ins"]}">{hyp_chunk}</span>')

    def block(title: str, parts: list[str]) -> str:
        # join with spaces: adjacent chunks must not glue together ("and" + highlighted word)
        return f"<div style='margin:2px 0'><b>{title}</b> {' '.join(parts) or '<i>(empty)</i>'}</div>"

    return block("reference&nbsp;", ref_parts) + block("hypothesis", hyp_parts)


def metrics_table(variant: str) -> pd.DataFrame:
    df = METRICS[METRICS["variant"] == variant]
    rows = []
    for language in LANGUAGES + ["all"]:
        row = {"language": language}
        rates = {}
        for model in MODELS:
            hit = df[(df["model"] == model) & (df["language"] == language)].iloc[0]
            rates[model] = float(hit["wer"])
            row[f"{model} WER %"] = round(rates[model] * 100, 2)
            row[f"{model} CER %"] = round(float(hit["cer"]) * 100, 2)
        # difference of the rates, not of the rounded percentages
        row["ΔWER (qwen−whisper, pp)"] = round((rates["qwen3-asr"] - rates["whisper"]) * 100, 2)
        rows.append(row)
    return pd.DataFrame(rows)


def normalization_table() -> pd.DataFrame:
    rows = []
    for model in MODELS:
        for language in LANGUAGES + ["all"]:
            cells = {v: METRICS[(METRICS["model"] == model) & (METRICS["language"] == language)
                                & (METRICS["variant"] == v)].iloc[0] for v in ("raw", "norm")}
            rows.append({
                "model": model, "language": language,
                "WER raw %": round(float(cells["raw"]["wer"]) * 100, 2),
                "WER norm %": round(float(cells["norm"]["wer"]) * 100, 2),
                "Δ WER pp": round((float(cells["norm"]["wer"]) - float(cells["raw"]["wer"])) * 100, 2),
                "CER raw %": round(float(cells["raw"]["cer"]) * 100, 2),
                "CER norm %": round(float(cells["norm"]["cer"]) * 100, 2),
                "Δ CER pp": round((float(cells["norm"]["cer"]) - float(cells["raw"]["cer"])) * 100, 2),
            })
    return pd.DataFrame(rows)


def view(label: str, variant: str):
    sid = LABELS.get(label, label)
    row = MANIFEST[sid]
    language = row["language"]
    ref = normalize(row["reference"], language, variant)
    audio_path = ROOT / row["audio_path"]
    audio = str(audio_path) if audio_path.exists() else None

    blocks = [f"<div style='color:#555'>{sid} · {language} · {row['duration_s']}s · "
              f"{row['gender']} · variant=<b>{variant}</b></div>"]
    if audio is None:
        blocks.append("<div style='color:#b45309'>audio file missing — run "
                      "<code>scripts/fetch_fleurs.py</code> to enable playback</div>")
    scores = []
    for model in MODELS:
        pred = PREDS[model].get(sid)
        score = PER_SAMPLE.get((model, sid, variant))
        if pred is None or score is None:
            blocks.append(f"<h4>{model}</h4><i>no prediction</i>")
            continue
        hyp = normalize(pred["model_output"], language, variant)
        blocks.append(f"<h4 style='margin:10px 0 2px'>{model}</h4>{diff_html(ref, hyp)}")
        scores.append({
            "model": model, "WER %": round(float(score["wer"]) * 100, 1),
            "CER %": round(float(score["cer"]) * 100, 1),
            "sub": int(score["substitutions"]), "del": int(score["deletions"]),
            "ins": int(score["insertions"]),
        })
    return audio, "".join(blocks), pd.DataFrame(scores)


def build() -> gr.Blocks:
    choices = sorted(LABELS)
    with gr.Blocks(title="asr-norm-check · FLEURS en/hi/ko") as demo:
        gr.Markdown(
            "# Whisper vs Qwen3-ASR on FLEURS\n"
            "50 clips each for English, Hindi, Korean. **WER** = word error rate, edits ÷ reference "
            "words; **CER** = the same on space-free characters (the meaningful metric for "
            "Hindi/Korean). *raw* scores strings as produced; *norm* applies the language-specific "
            "normalisation in `docs/normalization_rules.md`.\n\n"
            "Every number here comes from the committed files in `outputs/` — this app loads "
            "precomputed predictions and never runs a model."
        )
        variant = gr.Radio(["norm", "raw"], value="norm", label="Scoring variant (WER/CER columns below)")
        variant_note = gr.Markdown("Showing **norm** scoring.")

        with gr.Tab("Metrics"):
            gr.Markdown("### Per-language results")
            comparison = gr.Dataframe(value=metrics_table("norm"),
                                      label="Per-language results · scores in %, Δ in percentage points",
                                      interactive=False, wrap=True)
            gr.Markdown("### Effect of text normalisation (both variants, independent of the switch above)")
            gr.Dataframe(value=normalization_table(),
                         label="raw vs norm · WER/CER in %, Δ in percentage points",
                         interactive=False, wrap=True)
            gr.Markdown(
                "A negative Δ means the model was penalised for formatting that normalisation "
                "ignores. It does not mean the model got better: the same audio, the same "
                "hypotheses, a different scoring rule — see `docs/normalization_rules.md`."
            )
            def show_variant(selected_variant: str):
                return metrics_table(selected_variant), f"Showing **{selected_variant}** scoring."

            variant.change(show_variant, inputs=variant, outputs=[comparison, variant_note])

        with gr.Tab("Samples"):
            language = gr.Radio(LANGUAGES, value="en", label="Language")
            en_choices = [c for c in choices if MANIFEST[LABELS[c]]["language"] == "en"]
            picker = gr.Dropdown(choices=en_choices, value=en_choices[0] if en_choices else None,
                                 label="Sample", filterable=True,
                                 info="type a sample id (e.g. fleurs_hi_1774) or part of the text to filter")
            detail = gr.HTML()
            with gr.Row():
                audio = gr.Audio(label="Audio", type="filepath", interactive=False)
                scores = gr.Dataframe(label="Scores for this clip · S/D/I = substitution/deletion/insertion",
                                      interactive=False, wrap=True)
            gr.Markdown(
                "Colours: "
                "<span style='background:#ffe9b8;color:#7c4a03;padding:0 4px'>substitution</span> "
                "<span style='background:#ffd7d5;color:#8b1a10;padding:0 4px'>deletion</span> "
                "<span style='background:#d6f5d6;color:#14532d;padding:0 4px'>insertion</span>",)
            gr.Markdown(
                "*Highlighting is a word-level diff (difflib). The S/D/I columns are jiwer's "
                "counts; on individual clips the two can pair tokens differently, so trust the "
                "counts for numbers and the colours for where to look.*", elem_classes="muted")

            def on_language(lang: str, selected_variant: str):
                opts = [c for c in choices if MANIFEST[LABELS[c]]["language"] == lang]
                first = opts[0] if opts else None
                update = gr.update(choices=opts, value=first)
                if first is None:
                    return (update, None, "", pd.DataFrame())
                return (update, *view(first, selected_variant))

            outputs = [audio, detail, scores]
            language.change(on_language, inputs=[language, variant],
                            outputs=[picker, *outputs])
            picker.change(view, inputs=[picker, variant], outputs=outputs)
            variant.change(view, inputs=[picker, variant], outputs=outputs)
            demo.load(view, inputs=[picker, variant], outputs=outputs)
    return demo


if __name__ == "__main__":
    build().launch(server_name="0.0.0.0", server_port=7860, allowed_paths=[str(DATA)])

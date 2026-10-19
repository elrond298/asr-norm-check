#!/usr/bin/env python
"""Turn per-sample scores into error-analysis tables.

  uv run python scripts/analyze.py

Inputs : outputs/per_sample_scores.csv, outputs/predictions_*.jsonl
Outputs: outputs/error_cases.csv          one row per model x sample, raw+norm scores,
                                          dominant error type, changed tokens
         outputs/normalization_deltas.csv samples whose score changes most under
                                          normalisation (format-only differences)
Prints : error-type mix per model/language, raw-vs-norm deltas, the model gap
         before and after normalisation.
"""

from __future__ import annotations

import csv
import difflib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from asrbench.common import OUT_DIR, load_jsonl
from asrbench.normalize import normalize

MODELS = {"whisper": "predictions_whisper.jsonl", "qwen3-asr": "predictions_qwen.jsonl"}
LANGUAGES = ["en", "hi", "ko"]
CASE_FIELDS = [
    "model", "language", "sample_id", "wer_raw", "wer_norm", "cer_raw", "cer_norm",
    "substitutions", "deletions", "insertions", "dominant_error",
    "substituted", "deleted", "inserted", "reference", "hypothesis", "normalised_reference",
]


def load_metrics() -> dict[tuple[str, str, str], dict]:
    """Corpus-level rates as written by score.py, keyed by (model, language, variant)."""
    rows = {}
    for variant, name in (("raw", "metrics_baseline.csv"), ("norm", "metrics_normalized.csv")):
        with (OUT_DIR / name).open(encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                rows[(row["model"], row["language"], variant)] = row
    return rows


def load_scores() -> dict[tuple[str, str, str], dict]:
    with (OUT_DIR / "per_sample_scores.csv").open(encoding="utf-8") as fh:
        return {(r["model"], r["sample_id"], r["variant"]): r for r in csv.DictReader(fh)}


def tokens_changed(ref: str, hyp: str) -> tuple[list[str], list[str], list[str]]:
    """Word-level substitutions / deletions / insertions, ref and hyp already normalised.

    The alignment here is difflib's (what the UI highlights); the S/D/I counts in the CSV
    come from jiwer, so the two can disagree on individual clips.
    """
    a, b = ref.split(), hyp.split()
    subs, dels, ins = [], [], []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b).get_opcodes():
        if tag == "replace":
            # pair what can be paired; surplus ref tokens are deletions, surplus hyp insertions
            paired = min(i2 - i1, j2 - j1)
            subs += [f"{x} -> {y}" for x, y in zip(a[i1:i1 + paired], b[j1:j1 + paired])]
            dels += a[i1 + paired:i2]
            ins += b[j1 + paired:j2]
        elif tag == "delete":
            dels += a[i1:i2]
        elif tag == "insert":
            ins += b[j1:j2]
    return subs, dels, ins


def dominant(row: dict) -> str:
    counts = {"substitution": int(row["substitutions"]), "deletion": int(row["deletions"]),
              "insertion": int(row["insertions"])}
    top = max(counts, key=counts.get)
    return "none" if counts[top] == 0 else top


def main() -> None:
    scores = load_scores()

    cases = []
    for model, fname in MODELS.items():
        for pred in load_jsonl(OUT_DIR / fname):
            sid, lang = pred["sample_id"], pred["language"]
            ref_n = normalize(pred["reference"], lang, "norm")
            hyp_n = normalize(pred["model_output"], lang, "norm")
            subs, dels, ins = tokens_changed(ref_n, hyp_n)
            raw = scores[(model, sid, "raw")]
            norm = scores[(model, sid, "norm")]
            cases.append({
                "model": model, "language": lang, "sample_id": sid,
                "wer_raw": round(float(raw["wer"]), 4), "wer_norm": round(float(norm["wer"]), 4),
                "cer_raw": round(float(raw["cer"]), 4), "cer_norm": round(float(norm["cer"]), 4),
                "substitutions": norm["substitutions"], "deletions": norm["deletions"],
                "insertions": norm["insertions"],
                "dominant_error": dominant(norm),
                "substituted": " | ".join(subs[:6]), "deleted": " ".join(dels[:8]),
                "inserted": " ".join(ins[:8]),
                "reference": pred["reference"], "hypothesis": pred["model_output"],
                "normalised_reference": ref_n,
            })

    with (OUT_DIR / "error_cases.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CASE_FIELDS)
        writer.writeheader()
        writer.writerows(cases)

    deltas = sorted(cases, key=lambda c: abs(c["wer_norm"] - c["wer_raw"]) + abs(c["cer_norm"] - c["cer_raw"]),
                    reverse=True)
    delta_fields = ["model", "language", "sample_id", "wer_raw", "wer_norm", "cer_raw", "cer_norm",
                    "dominant_error", "reference", "hypothesis"]
    with (OUT_DIR / "normalization_deltas.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=delta_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(deltas)

    print("== error-type mix (normalised scoring) ==")
    print(f"{'model':10s} {'lang':5s} {'sub':>5s} {'del':>5s} {'ins':>5s}  share of errors")
    for model in MODELS:
        for lang in LANGUAGES:
            rows = [c for c in cases if c["model"] == model and c["language"] == lang]
            s = sum(int(c["substitutions"]) for c in rows)
            d = sum(int(c["deletions"]) for c in rows)
            i = sum(int(c["insertions"]) for c in rows)
            total = max(s + d + i, 1)
            print(f"{model:10s} {lang:5s} {s:5d} {d:5d} {i:5d}  S {s / total:.0%} D {d / total:.0%} I {i / total:.0%}")

    print("\n== raw -> normalised WER / CER (corpus level, as in metrics_*.csv) ==")
    metrics = load_metrics()
    for model in MODELS:
        for lang in LANGUAGES:
            raw, norm = metrics[(model, lang, "raw")], metrics[(model, lang, "norm")]
            print(f"{model:10s} {lang:5s} "
                  f"WER {float(raw['wer']) * 100:6.2f}% -> {float(norm['wer']) * 100:6.2f}%  "
                  f"CER {float(raw['cer']) * 100:6.2f}% -> {float(norm['cer']) * 100:6.2f}%")

    print("\n== model gap (qwen3-asr minus whisper, corpus WER) ==")
    for lang in LANGUAGES + ["all"]:
        parts = []
        for variant in ("raw", "norm"):
            gap = (float(metrics[("qwen3-asr", lang, variant)]["wer"])
                   - float(metrics[("whisper", lang, variant)]["wer"]))
            parts.append(f"{variant}: {gap * 100:+.2f}pp")
        print(f"{lang:5s} " + "   ".join(parts))

    print("\n== 8 largest normalisation shifts ==")
    for c in deltas[:8]:
        print(f"{c['model']:10s} {c['language']:3s} {c['sample_id']:22s} "
              f"WER {c['wer_raw'] * 100:6.1f}% -> {c['wer_norm'] * 100:6.1f}%")


if __name__ == "__main__":
    main()

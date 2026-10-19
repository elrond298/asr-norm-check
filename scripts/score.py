#!/usr/bin/env python
"""Score both models with one rule set: raw and normalised, per sample and per language.

  uv run python scripts/score.py

Inputs : data/manifest.jsonl, outputs/predictions_{whisper,qwen}.jsonl
Outputs: outputs/per_sample_scores.csv
         outputs/metrics_baseline.csv     (variant = raw)
         outputs/metrics_normalized.csv   (variant = norm)
         outputs/alignment_check.md
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from asrbench.common import MANIFEST, OUT_DIR, load_jsonl
from asrbench.normalize import VARIANTS, normalize
from asrbench.scoring import per_sample

MODELS = {"whisper": "predictions_whisper.jsonl", "qwen3-asr": "predictions_qwen.jsonl"}
PER_SAMPLE = [
    "model", "language", "sample_id", "variant", "wer", "cer",
    "hits", "substitutions", "deletions", "insertions",
    "ref_tokens", "hyp_tokens", "ref_chars", "char_errors",
]


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in fields})


def check_alignment(preds: dict[str, list[dict]], manifest: list[dict]) -> str:
    ids = {name: [r["sample_id"] for r in rows] for name, rows in preds.items()}
    ref_ids = [r["sample_id"] for r in manifest]
    refs = {r["sample_id"]: r["reference"] for r in manifest}
    lines = ["# Prediction alignment check", ""]
    ok = True
    for name, got in ids.items():
        same_order = got == ref_ids
        same_set = set(got) == set(ref_ids)
        dupes = len(got) - len(set(got))
        missing = sorted(set(ref_ids) - set(got))[:5]
        lines.append(
            f"- **{name}**: {len(got)} rows, same set as manifest: {same_set}, "
            f"same order: {same_order}, duplicates: {dupes}, missing (first 5): {missing or 'none'}"
        )
        ok &= same_set and dupes == 0
        # reference text must match the manifest for every row
        drift = [r["sample_id"] for r in preds[name] if refs.get(r["sample_id"]) != r["reference"]]
        lines.append(f"- **{name}** reference drift vs manifest: {len(drift)} rows")
        ok &= not drift
    fields = {name: tuple(sorted(rows[0])) for name, rows in preds.items()}
    lines.append(f"- field sets identical across models: {len(set(fields.values())) == 1}")
    ok &= len(set(fields.values())) == 1
    lines += ["", f"**Result: {'PASS' if ok else 'FAIL'}**", ""]
    return "\n".join(lines)


def score_model(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    per, corpus = [], []
    for variant in VARIANTS:
        by_lang: dict[str, list[dict]] = {}
        for row in rows:
            ref = normalize(row["reference"], row["language"], variant)
            hyp = normalize(row["model_output"], row["language"], variant)
            detail = per_sample(ref, hyp)
            per.append({
                "model": rows[0]["model"], "language": row["language"],
                "sample_id": row["sample_id"], "variant": variant, **detail,
            })
            by_lang.setdefault(row["language"], []).append(detail)
        for language, details in by_lang.items():
            corpus.append(_aggregate(rows[0]["model"], language, variant, details))
        corpus.append(_aggregate(rows[0]["model"], "all", variant, [d for ds in by_lang.values() for d in ds]))
    return per, corpus


def _aggregate(model: str, language: str, variant: str, details: list[dict]) -> dict:
    total = {k: sum(d[k] for d in details) for k in
             ("substitutions", "deletions", "insertions", "ref_tokens", "ref_chars", "char_errors")}
    errors = total["substitutions"] + total["deletions"] + total["insertions"]
    return {
        "model": model, "language": language, "variant": variant, "n": len(details),
        **total,
        "wer": round(errors / total["ref_tokens"], 6) if total["ref_tokens"] else 0.0,
        "cer": round(total["char_errors"] / total["ref_chars"], 6) if total["ref_chars"] else 0.0,
    }


def main() -> None:
    manifest = load_jsonl(MANIFEST)
    preds = {name: load_jsonl(OUT_DIR / fname) for name, fname in MODELS.items()}

    alignment = check_alignment(preds, manifest)
    (OUT_DIR / "alignment_check.md").write_text(alignment, encoding="utf-8")
    print(alignment)

    per_all, metrics = [], []
    for rows in preds.values():
        per, corpus = score_model(rows)
        per_all += per
        metrics += corpus

    write_csv(OUT_DIR / "per_sample_scores.csv", per_all, PER_SAMPLE)
    fields = ["model", "language", "variant", "n", "wer", "cer", "substitutions", "deletions",
              "insertions", "ref_tokens", "ref_chars", "char_errors"]
    for variant, name in (("raw", "metrics_baseline.csv"), ("norm", "metrics_normalized.csv")):
        write_csv(OUT_DIR / name, [m for m in metrics if m["variant"] == variant], fields)

    print(f"{'model':10s} {'lang':5s} {'variant':8s} {'n':>3s} {'WER':>7s} {'CER':>7s}")
    for m in metrics:
        print(f"{m['model']:10s} {m['language']:5s} {m['variant']:8s} {m['n']:3d} "
              f"{m['wer'] * 100:6.2f}% {m['cer'] * 100:6.2f}%")


if __name__ == "__main__":
    main()

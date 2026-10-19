"""Scoring: corpus-level and per-sample WER/CER, computed identically for every model."""

from __future__ import annotations

import json
from pathlib import Path

import jiwer

from asrbench.normalize import normalize, without_spaces

# WER tokenisation is whitespace for all three languages (FLEURS references are
# space-separated, including Korean). CER is computed on space-free characters,
# which is what makes it meaningful for Devanagari and Hangul.


def load_manifest(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_predictions(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _squeeze(text: str) -> str:
    """Collapse every whitespace run to one space (jiwer only splits on ' ', not tabs/newlines)."""
    return " ".join(text.split())


def _counts(ref: str, hyp: str) -> dict:
    ref_tokens, hyp_tokens = ref.split(), hyp.split()
    if not hyp_tokens:
        return {"hits": 0, "substitutions": 0, "deletions": len(ref_tokens), "insertions": 0}
    if not ref_tokens:
        return {"hits": 0, "substitutions": 0, "deletions": 0, "insertions": len(hyp_tokens)}
    out = jiwer.process_words(ref, hyp)
    return {
        "hits": out.hits,
        "substitutions": out.substitutions,
        "deletions": out.deletions,
        "insertions": out.insertions,
    }


def per_sample(ref: str, hyp: str) -> dict:
    """WER/CER for one pair of already-normalised strings.

    Zero-denominator convention: an empty reference scores WER/CER 1.0 when the hypothesis is
    non-empty and 0.0 when both sides are empty. Every reference in the frozen manifest is
    non-empty, so this only affects ad-hoc use of the library.
    """
    ref, hyp = _squeeze(ref), _squeeze(hyp)
    counts = _counts(ref, hyp)
    n_ref = len(ref.split())
    errors = counts["substitutions"] + counts["deletions"] + counts["insertions"]
    ref_chars, hyp_chars = without_spaces(ref), without_spaces(hyp)
    char_errors = 0
    if ref_chars or hyp_chars:
        chars = jiwer.process_characters(ref_chars, hyp_chars)
        char_errors = chars.substitutions + chars.deletions + chars.insertions
    return {
        **counts,
        "ref_tokens": n_ref,
        "hyp_tokens": len(hyp.split()),
        "ref_chars": len(ref_chars),
        "char_errors": char_errors,
        "wer": errors / n_ref if n_ref else (1.0 if hyp_chars else 0.0),
        "cer": char_errors / len(ref_chars) if ref_chars else (1.0 if hyp_chars else 0.0),
    }


def score_set(rows: list[dict]) -> dict:
    """rows: [{reference, hypothesis}] -> corpus totals and rates."""
    totals = {"n": len(rows), "substitutions": 0, "deletions": 0, "insertions": 0,
              "ref_tokens": 0, "ref_chars": 0, "char_errors": 0}
    for row in rows:
        d = per_sample(row["reference"], row["hypothesis"])
        for key in ("substitutions", "deletions", "insertions", "ref_tokens", "ref_chars", "char_errors"):
            totals[key] += d[key]
    errors = totals["substitutions"] + totals["deletions"] + totals["insertions"]
    totals["wer"] = errors / totals["ref_tokens"] if totals["ref_tokens"] else 0.0
    totals["cer"] = totals["char_errors"] / totals["ref_chars"] if totals["ref_chars"] else 0.0
    return totals


def score_variant(records: list[dict], language: str, variant: str) -> list[dict]:
    """Return the records with reference/hypothesis replaced by their normalised form."""
    return [
        {
            **r,
            "language": language,
            "reference": normalize(r["reference"], language, variant),
            "hypothesis": normalize(r["hypothesis"], language, variant),
        }
        for r in records
    ]

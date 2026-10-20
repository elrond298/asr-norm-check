#!/usr/bin/env python
"""Per-rule ablation: which normalisation steps actually move WER.

  uv run python scripts/ablate.py        # -> outputs/ablation.csv + printed table

Each row re-scores the committed predictions with *one* rule applied on top of the raw strings,
so the drop is attributable to that rule rather than to the whole pipeline. No model runs here;
denominators are recomputed per variant because normalisation can change token counts.
"""

from __future__ import annotations

import csv
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from whisper.normalizers.english import (  # noqa: E402
    EnglishNumberNormalizer, EnglishSpellingNormalizer, remove_symbols_and_diacritics,
)

from asrbench.common import OUT_DIR, load_jsonl  # noqa: E402
from asrbench.normalize import normalize  # noqa: E402
from asrbench.scoring import per_sample  # noqa: E402

PREDICTIONS = {"whisper": "predictions_whisper.jsonl", "qwen3-asr": "predictions_qwen.jsonl"}
LANGS = ("en", "hi", "ko")
ZW = dict.fromkeys(map(ord, "\u200b\u200c\u200d\ufeff"))
DEVA = str.maketrans("०१२३४५६७८९", "0123456789")

_number = EnglishNumberNormalizer()
_spelling = EnglishSpellingNormalizer()

_CONTRACTIONS = {
    r"\bwon't\b": "will not", r"\bcan't\b": "can not", r"\blet's\b": "let us",
    r"\bgonna\b": "going to", r"\bmr\b": "mister ", r"\bdr\b": "doctor ",
    r"\bprof\b": "professor ", r"n't\b": " not", r"'re\b": " are", r"'s\b": " is",
    r"'ll\b": " will", r"'ve\b": " have", r"'m\b": " am",
}


def _squeeze(text: str) -> str:
    return " ".join(text.split())


def _drop_punct_keep_case(text: str) -> str:
    """Symbols/punctuation -> space; letters, digits and combining marks survive."""
    out = "".join(" " if unicodedata.category(ch)[0] in "SP" else ch
                  for ch in unicodedata.normalize("NFKC", text))
    return _squeeze(out)


def _en_punct_only(text: str) -> str:
    return _squeeze(remove_symbols_and_diacritics(text.lower(), keep=".%$¢€£"))


def _en_numbers_only(text: str) -> str:
    return _squeeze(_number(text.lower()))


def _en_spelling_only(text: str) -> str:
    return _squeeze(_spelling(text.lower()))


def _en_contractions_only(text: str) -> str:
    import re

    out = text.lower()
    for pattern, replacement in _CONTRACTIONS.items():
        out = re.sub(pattern, replacement, out)
    return _squeeze(out)


def _lower(text: str) -> str:
    return _squeeze(text.lower())


def _zw(text: str) -> str:
    return _squeeze(text.translate(ZW))


def _deva_digits(text: str) -> str:
    return _squeeze(text.translate(DEVA))


def _nfkc(text: str) -> str:
    return _squeeze(unicodedata.normalize("NFKC", text))


#: rule name -> function; "" is the raw baseline.
STEPS: dict[str, dict[str, object]] = {
    "en": {
        "raw": None,
        "lowercase": _lower,
        "punctuation + symbols": _en_punct_only,
        "contractions / titles": _en_contractions_only,
        "number + currency wording": _en_numbers_only,
        "British -> American spelling": _en_spelling_only,
        "full norm pipeline": lambda t: normalize(t, "en"),
    },
    "hi": {
        "raw": None,
        "zero-width characters": _zw,
        "lowercase": _lower,
        "punctuation / symbols": _drop_punct_keep_case,
        "Devanagari digits -> ASCII": _deva_digits,
        "full norm pipeline": lambda t: normalize(t, "hi"),
    },
    "ko": {
        "raw": None,
        "zero-width characters": _zw,
        "lowercase": _lower,
        "NFKC (full-width -> ASCII)": _nfkc,
        "punctuation / symbols": _drop_punct_keep_case,
        "full norm pipeline": lambda t: normalize(t, "ko"),
    },
}


def corpus_wer(rows: list[dict], step, language: str) -> float:
    errors = tokens = 0
    for row in rows:
        ref, hyp = row["reference"], row["model_output"]
        if step is not None:
            ref, hyp = step(ref), step(hyp)
        detail = per_sample(ref, hyp)
        errors += detail["substitutions"] + detail["deletions"] + detail["insertions"]
        tokens += detail["ref_tokens"]
    return 100 * errors / tokens if tokens else 0.0


def main() -> None:
    out_rows = []
    for language in LANGS:
        for model, fname in PREDICTIONS.items():
            rows = [r for r in load_jsonl(OUT_DIR / fname) if r["language"] == language]
            base = corpus_wer(rows, None, language)
            for step_name, step in STEPS[language].items():
                wer = corpus_wer(rows, step, language)
                out_rows.append({
                    "language": language, "model": model, "rule": step_name,
                    "wer": round(wer, 4), "delta_vs_raw": round(wer - base, 4),
                })

    path = OUT_DIR / "ablation.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["language", "model", "rule", "wer", "delta_vs_raw"])
        writer.writeheader()
        writer.writerows(out_rows)

    for language in LANGS:
        print(f"\n{language}  (corpus WER %, raw = "
              f"{next(r['wer'] for r in out_rows if r['language'] == language and r['model'] == 'whisper' and r['rule'] == 'raw'):.2f}"
              " for whisper)")
        print(f"{'rule':32s} {'whisper':>9s} {'Δ':>7s} {'qwen':>9s} {'Δ':>7s}")
        for name in STEPS[language]:
            cells = [r for r in out_rows if r["language"] == language and r["rule"] == name]
            w = next(r for r in cells if r["model"] == "whisper")
            q = next(r for r in cells if r["model"] == "qwen3-asr")
            print(f"{name:32s} {w['wer']:8.2f} {w['delta_vs_raw']:+7.2f} "
                  f"{q['wer']:8.2f} {q['delta_vs_raw']:+7.2f}")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()

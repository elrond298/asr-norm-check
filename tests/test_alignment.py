#!/usr/bin/env python
"""Self-check for the alignment gate that score.py runs before writing metrics.

  uv run python tests/test_alignment.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import score  # noqa: E402

MANIFEST = [
    {"sample_id": "a", "language": "en", "reference": "one two", "audio_path": "x.wav"},
    {"sample_id": "b", "language": "hi", "reference": "तीन चार", "audio_path": "y.wav"},
]


def row(sid: str, language: str, reference: str, text: str, model: str = "m") -> dict:
    return {"sample_id": sid, "language": language, "reference": reference,
            "model_output": text, "model": model}


GOOD = {
    "whisper": [row("a", "en", "one two", "one two"), row("b", "hi", "तीन चार", "तीन चार")],
    "qwen": [row("a", "en", "one two", "one to"), row("b", "hi", "तीन चार", "तीन चार")],
}


def check(name: str, preds: dict, expect_ok: bool) -> None:
    _, ok = score.check_alignment(preds, MANIFEST)
    assert ok is expect_ok, f"{name}: ok={ok}, expected {expect_ok}"


def main() -> None:
    check("aligned", GOOD, True)

    missing = {k: [r for r in v if r["sample_id"] == "a"] for k, v in GOOD.items()}
    check("dropped clip", missing, False)

    language = {k: [dict(r) for r in v] for k, v in GOOD.items()}
    language["whisper"][1]["language"] = "en"          # silently selects the wrong normaliser
    check("language tamper", language, False)

    reference = {k: [dict(r) for r in v] for k, v in GOOD.items()}
    reference["qwen"][0]["reference"] = "one two three"  # changes the WER denominator
    check("reference drift", reference, False)

    empty = {k: [dict(r) for r in v] for k, v in GOOD.items()}
    empty["whisper"][0]["model_output"] = "   "
    check("empty output", empty, False)

    dupes = {k: [dict(r) for r in v] for k, v in GOOD.items()}
    dupes["qwen"][1]["sample_id"] = "a"
    check("duplicate id", dupes, False)

    extra = {k: [dict(r) for r in v] for k, v in GOOD.items()}
    extra["qwen"].append(row("zzz", "en", "ghost", "ghost"))
    check("unknown id", extra, False)

    schema = {k: [dict(r) for r in v] for k, v in GOOD.items()}
    schema["qwen"][1]["extra_field"] = 1
    check("schema drift on a later row", schema, False)

    print("all alignment checks passed")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Whisper (openai/whisper, official reference implementation) predictions.

  uv run python scripts/run_whisper.py                # all 150 clips
  uv run python scripts/run_whisper.py --limit 10     # timing probe

Model weights come from openaipublic.azureedge.net (whisper's own loader).
Audio is resampled to 16 kHz mono by whisper.load_audio().
"""

from __future__ import annotations

import argparse
import importlib.metadata as md
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import whisper

from asrbench.common import MANIFEST, OUT_DIR, load_jsonl, write_predictions

MODEL_NAME = "openai/whisper-large-v3"
CHECKPOINT = "large-v3"
#: whisper downloads from a content-addressed URL; its sha256 is the immutable weight identity.
CHECKPOINT_SHA256 = whisper._MODELS[CHECKPOINT].rsplit("/", 2)[-2]
CONFIG = {
    "checkpoint": CHECKPOINT,
    "checkpoint_url": whisper._MODELS[CHECKPOINT],
    "device": "cuda",
    "sample_rate": 16000,
    "beam_size": 5,
    "temperature": 0.0,
    "condition_on_previous_text": False,
    "language": "forced per sample (en/hi/ko)",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="score only the first N clips (timing probes)")
    # a probe run must never overwrite the canonical predictions the results were scored from
    ap.add_argument("--out", type=Path, default=None,
                    help="default: outputs/predictions_whisper.jsonl, or outputs/probe_whisper.jsonl with --limit")
    args = ap.parse_args()

    manifest = load_jsonl(MANIFEST)
    if args.limit:
        manifest = manifest[: args.limit]
    out = args.out or OUT_DIR / ("probe_whisper.jsonl" if args.limit else "predictions_whisper.jsonl")

    model = whisper.load_model(CHECKPOINT, device=CONFIG["device"])
    version = f"openai-whisper {md.version('openai-whisper')} / {MODEL_NAME} @ {CHECKPOINT_SHA256[:12]}"

    def transcribe(row: dict) -> str:
        audio = whisper.load_audio(str(Path(__file__).resolve().parents[1] / row["audio_path"]))
        result = model.transcribe(
            audio,
            language=row["language"],
            beam_size=CONFIG["beam_size"],
            temperature=CONFIG["temperature"],
            condition_on_previous_text=CONFIG["condition_on_previous_text"],
            fp16=True,
            verbose=None,
        )
        return result["text"].strip()

    write_predictions(out, manifest, transcribe, "whisper", version, CONFIG)


if __name__ == "__main__":
    main()

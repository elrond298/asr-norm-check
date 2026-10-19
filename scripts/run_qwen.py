#!/usr/bin/env python
"""Qwen3-ASR (Qwen/Qwen3-ASR-1.7B-hf) predictions via the Transformers-native path.

  uv run python scripts/run_qwen.py
  uv run python scripts/run_qwen.py --limit 10

Language is forced per sample (en/hi/ko) so the comparison against Whisper is
like-for-like; Qwen is also able to auto-detect, which is not what we measure here.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import soundfile as sf
import torch
import transformers
from transformers import AutoModelForMultimodalLM, AutoProcessor

from asrbench.common import MANIFEST, OUT_DIR, load_jsonl, write_predictions

MODEL_ID = "Qwen/Qwen3-ASR-1.7B-hf"
#: pinned HF commit, so a later upload cannot silently change the weights or processor config
REVISION = "bcd2b5b7f32b480ab5790554cfa8347f246a14f3"
CONFIG = {
    "model_id": MODEL_ID,
    "revision": REVISION,
    "device": "cuda",
    "dtype": "bfloat16",
    "do_sample": False,
    "max_new_tokens": 256,
    "language": "forced per sample (en/hi/ko)",
    "audio": "soundfile, 16 kHz mono (FLEURS native rate, no resampling)",
}


def _audio_array(path: Path):
    """FLEURS clips are 16 kHz mono; fail loudly rather than silently resample."""
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    if sr != 16000 or data.shape[1] != 1:
        raise ValueError(f"{path}: expected 16 kHz mono, got {sr} Hz / {data.shape[1]} ch")
    return data[:, 0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    # a probe run must never overwrite the canonical predictions the results were scored from
    ap.add_argument("--out", type=Path, default=None,
                    help="default: outputs/predictions_qwen.jsonl, or outputs/probe_qwen.jsonl with --limit")
    args = ap.parse_args()

    manifest = load_jsonl(MANIFEST)
    if args.limit:
        manifest = manifest[: args.limit]
    out = args.out or OUT_DIR / ("probe_qwen.jsonl" if args.limit else "predictions_qwen.jsonl")

    processor = AutoProcessor.from_pretrained(MODEL_ID, revision=REVISION)
    model = AutoModelForMultimodalLM.from_pretrained(MODEL_ID, revision=REVISION, dtype=torch.bfloat16)
    model = model.to(CONFIG["device"]).eval()
    version = f"transformers {transformers.__version__} / {MODEL_ID} @ {REVISION[:12]}"

    def transcribe(row: dict) -> str:
        inputs = processor.apply_transcription_request(
            audio=_audio_array(Path(__file__).resolve().parents[1] / row["audio_path"]),
            language=row["language"],
        ).to(model.device, model.dtype)
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                do_sample=CONFIG["do_sample"],
                max_new_tokens=CONFIG["max_new_tokens"],
            )
        new_ids = generated[:, inputs["input_ids"].shape[1]:]
        return processor.decode(new_ids, return_format="transcription_only")[0].strip()

    write_predictions(out, manifest, transcribe, "qwen3-asr", version, CONFIG)


if __name__ == "__main__":
    main()

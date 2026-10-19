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

import torch
import transformers
from transformers import AutoModelForMultimodalLM, AutoProcessor

from asrbench.common import MANIFEST, OUT_DIR, load_jsonl, write_predictions

MODEL_ID = "Qwen/Qwen3-ASR-1.7B-hf"
CONFIG = {
    "model_id": MODEL_ID,
    "device": "cuda",
    "dtype": "bfloat16",
    "do_sample": False,
    "max_new_tokens": 256,
    "language": "forced per sample (en/hi/ko)",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", type=Path, default=OUT_DIR / "predictions_qwen.jsonl")
    args = ap.parse_args()

    manifest = load_jsonl(MANIFEST)
    if args.limit:
        manifest = manifest[: args.limit]

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForMultimodalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map=CONFIG["device"]
    ).eval()
    version = f"transformers {transformers.__version__} / {MODEL_ID}"

    def transcribe(row: dict) -> str:
        inputs = processor.apply_transcription_request(
            audio=str(Path(__file__).resolve().parents[1] / row["audio_path"]),
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

    write_predictions(args.out, manifest, transcribe, "qwen3-asr", version, CONFIG)


if __name__ == "__main__":
    main()

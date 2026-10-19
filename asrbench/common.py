"""Shared plumbing for the prediction stages."""

from __future__ import annotations

import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "manifest.jsonl"
OUT_DIR = ROOT / "outputs"
PREDICTION_FIELDS = [
    "sample_id", "language", "reference", "model_output",
    "model", "model_version", "inference_config",
    "audio_path", "duration_s", "runtime_s",
]


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def resolve_audio(row: dict) -> str:
    return str(ROOT / row["audio_path"])


def write_predictions(path: Path, manifest: list[dict], transcribe, model: str,
                      model_version: str, config: dict) -> None:
    """transcribe(row) -> str. Writes one JSONL row per manifest entry, in manifest order."""
    rows = []
    started = time.perf_counter()
    for i, row in enumerate(manifest, 1):
        t0 = time.perf_counter()
        text = transcribe(row)
        runtime = round(time.perf_counter() - t0, 3)
        rows.append({
            "sample_id": row["sample_id"],
            "language": row["language"],
            "reference": row["reference"],
            "model_output": text,
            "model": model,
            "model_version": model_version,
            "inference_config": config,
            "audio_path": row["audio_path"],
            "duration_s": row["duration_s"],
            "runtime_s": runtime,
        })
        print(f"[{i}/{len(manifest)}] {row['sample_id']} {runtime:.2f}s :: {text[:80]!r}", flush=True)
    write_jsonl(path, rows)
    total = time.perf_counter() - started
    audio = sum(r["duration_s"] for r in manifest)
    print(f"wrote {path} ({len(rows)} rows) in {total:.1f}s, {audio / total:.1f}x realtime", flush=True)

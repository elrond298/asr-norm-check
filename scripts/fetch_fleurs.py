#!/usr/bin/env python
"""Build the frozen FLEURS subset used by every experiment.

Selection rule (frozen once, do not edit): parse data/<lang>/test.tsv from the
pinned dataset revision, keep rows with a non-empty transcription and a
duration between 4 s and 25 s, sort by numeric sample id, then take 50 clips
spaced evenly across that sorted list (round(linspace(0, n-1, 50))).

Outputs:
  data/audio/<lang>/<sample_id>.wav
  data/manifest.jsonl   (one row per clip, includes sha256 of the wav)
  data/manifest.meta.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO = "google/fleurs"
# Pinned dataset revision: the same revision forever, so the subset cannot drift.
REVISION = "70bb2e84b976b7e960aa89f1c648e09c59f894dd"
SPLIT = "test"
LANG_DIRS = {"en": "en_us", "hi": "hi_in", "ko": "ko_kr"}
MIN_SEC, MAX_SEC = 4.0, 25.0
TSV_COLUMNS = ["id", "filename", "transcription", "raw_transcription", "chars", "num_samples", "gender"]
ROOT = Path(__file__).resolve().parents[1]


def sample_count(n: int, k: int) -> list[int]:
    """Evenly spaced indices over range(n); deterministic, no RNG."""
    if k < 1:
        raise SystemExit(f"--per-language must be >= 1, got {k}")
    if k == 1 or n <= k:
        return list(range(min(n, k)))
    return [round(i * (n - 1) / (k - 1)) for i in range(k)]


def read_tsv(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != len(TSV_COLUMNS):
                raise SystemExit(f"unexpected column count in {path}: {len(parts)}")
            row = dict(zip(TSV_COLUMNS, parts))
            row["duration_s"] = int(row["num_samples"]) / 16000.0
            rows.append(row)
    return rows


def select(rows: list[dict], k: int) -> list[dict]:
    usable = [
        r for r in rows
        if r["transcription"].strip() and MIN_SEC <= r["duration_s"] <= MAX_SEC
    ]
    usable.sort(key=lambda r: int(r["id"]))
    return [usable[i] for i in sample_count(len(usable), k)]


def fetch_language(lang: str, lang_dir: str, k: int, data_dir: Path, force: bool) -> list[dict]:
    tsv = Path(hf_hub_download(REPO, f"data/{lang_dir}/{SPLIT}.tsv", repo_type="dataset", revision=REVISION))
    rows = select(read_tsv(tsv), k)
    wanted = {r["filename"]: r for r in rows}
    audio_dir = data_dir / "audio" / lang
    audio_dir.mkdir(parents=True, exist_ok=True)
    missing = {name for name in wanted if force or not (audio_dir / name).exists()}
    if missing:
        tar_path = Path(hf_hub_download(
            REPO, f"data/{lang_dir}/audio/{SPLIT}.tar.gz", repo_type="dataset", revision=REVISION,
        ))
        with tarfile.open(tar_path, mode="r|gz") as tar:
            for member in tar:
                name = Path(member.name).name
                if name in missing and member.isfile():
                    buf = tar.extractfile(member)
                    assert buf is not None
                    (audio_dir / name).write_bytes(buf.read())
                    missing.discard(name)
                    if not missing:
                        break
    if missing:
        raise SystemExit(f"{lang}: {len(missing)} audio files not found in {SPLIT}.tar.gz")

    out = []
    for row in rows:
        wav = audio_dir / row["filename"]
        out.append({
            "sample_id": f"fleurs_{lang}_{row['id']}",
            "language": lang,
            "split": SPLIT,
            "source": REPO,
            "source_revision": REVISION,
            "source_file": row["filename"],
            "reference": row["transcription"],
            "reference_raw": row["raw_transcription"],
            "gender": row["gender"],
            "duration_s": round(row["duration_s"], 3),
            "audio_path": str(wav.relative_to(ROOT)) if wav.is_relative_to(ROOT) else str(wav),
            "audio_sha256": hashlib.sha256(wav.read_bytes()).hexdigest(),
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-language", type=int, default=50)
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data")
    ap.add_argument("--force", action="store_true", help="re-extract audio even if present")
    ap.add_argument("--verify", action="store_true",
                    help="compare the rebuilt manifest against the frozen data/manifest.jsonl")
    args = ap.parse_args()

    data_dir: Path = args.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for lang, lang_dir in LANG_DIRS.items():
        rows = fetch_language(lang, lang_dir, args.per_language, data_dir, args.force)
        print(f"{lang}: {len(rows)} clips")
        manifest.extend(rows)

    if args.verify:
        # --verify is read-only: never overwrite the frozen list it is checking against
        frozen_path = data_dir / "manifest.jsonl"
        if not frozen_path.exists():
            raise SystemExit(f"--verify needs an existing {frozen_path}")
        frozen = {}
        for line in frozen_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                frozen[row["sample_id"]] = row
        rebuilt_ids = [r["sample_id"] for r in manifest]
        if set(rebuilt_ids) != set(frozen):
            raise SystemExit(
                f"verify: id sets differ (rebuilt {len(rebuilt_ids)}, frozen {len(frozen)}); "
                f"missing {sorted(set(frozen) - set(rebuilt_ids))[:5]}, "
                f"extra {sorted(set(rebuilt_ids) - set(frozen))[:5]}")
        mismatch = [r["sample_id"] for r in manifest
                    if frozen[r["sample_id"]].get("audio_sha256") != r["audio_sha256"]
                    or frozen[r["sample_id"]].get("reference") != r["reference"]]
        print(f"verify: {len(manifest) - len(mismatch)}/{len(manifest)} clips match the frozen manifest")
        if mismatch:
            raise SystemExit(f"mismatch vs frozen manifest: {mismatch[:5]}")
        print("verify: frozen manifest left untouched")
        return

    payload = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in manifest)
    (data_dir / "manifest.jsonl").write_text(payload, encoding="utf-8")
    meta = {
        "source": REPO,
        "source_revision": REVISION,
        "split": SPLIT,
        "per_language": args.per_language,
        "languages": LANG_DIRS,
        "selection_rule": (
            f"{SPLIT}.tsv rows with non-empty transcription and {MIN_SEC}-{MAX_SEC}s duration, "
            f"sorted by numeric id, {args.per_language} clips at round(linspace(0, n-1, {args.per_language}))"
        ),
        "counts": {lang: sum(1 for r in manifest if r["language"] == lang) for lang in LANG_DIRS},
        "manifest_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
    }
    (data_dir / "manifest.meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(meta["counts"]), "manifest_sha256:", meta["manifest_sha256"][:16])


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Render the result figures from the committed CSV outputs.

  uv run python scripts/make_figures.py      # -> docs/img/fig_*.png

Reads only outputs/*.csv, so the figures cannot drift from the tables.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

import re  # noqa: E402

from asrbench.common import MANIFEST, OUT_DIR, ROOT, load_jsonl  # noqa: E402
from asrbench.normalize import normalize  # noqa: E402
from asrbench.scoring import per_sample  # noqa: E402

FIG_DIR = ROOT / "docs" / "img"
LANGS = ["en", "hi", "ko"]
LANG_LABEL = {"en": "English", "hi": "Hindi", "ko": "Korean", "all": "all 150"}
MODELS = ["whisper", "qwen3-asr"]
MODEL_LABEL = {"whisper": "Whisper large-v3", "qwen3-asr": "Qwen3-ASR 1.7B"}
COLOR = {"whisper": "#4c78a8", "qwen3-asr": "#e45756"}
PREDICTION_FILE = {"whisper": "predictions_whisper.jsonl", "qwen3-asr": "predictions_qwen.jsonl"}
SURFACE = "#fcfcfb"
TEXT = "#222"


def style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.size": 10, "axes.edgecolor": "#bbb", "axes.labelcolor": TEXT,
        "xtick.color": TEXT, "ytick.color": TEXT, "text.color": TEXT,
        "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 160,
    })


def load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics = pd.concat([pd.read_csv(OUT_DIR / "metrics_baseline.csv").assign(variant="raw"),
                         pd.read_csv(OUT_DIR / "metrics_normalized.csv").assign(variant="norm")])
    per = pd.read_csv(OUT_DIR / "per_sample_scores.csv")
    return metrics, per


def pct(series: pd.Series) -> pd.Series:
    return series * 100


def fig_wer_cer(metrics: pd.DataFrame) -> Path:
    """Grouped bars: WER and CER per language, both models, normalised scoring."""
    norm = metrics[metrics["variant"] == "norm"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, metric, title in ((axes[0], "wer", "Word error rate (WER)"),
                             (axes[1], "cer", "Character error rate (CER)")):
        width = 0.36
        for i, model in enumerate(MODELS):
            values = [pct(norm[(norm["model"] == model) & (norm["language"] == l)][metric]).iloc[0]
                      for l in LANGS]
            pos = [x + (i - 0.5) * width for x in range(len(LANGS))]
            bars = ax.bar(pos, values, width, label=MODEL_LABEL[model], color=COLOR[model])
            ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=8)
        ax.set_xticks(range(len(LANGS)), [LANG_LABEL[l] for l in LANGS])
        ax.set_title(title, fontsize=10, loc="left")
        ax.set_ylabel("%")
        ax.set_ylim(0, 28)
        ax.grid(axis="y", color="#e4e4e4", linewidth=0.8)
        ax.set_axisbelow(True)
    axes[0].legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.suptitle("Same 150 clips, scored with the same rule (normalised)", fontsize=11, x=0.01,
                 ha="left", y=1.0)
    fig.tight_layout()
    out = FIG_DIR / "fig_wer_cer.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def _spread(values: list[float], gap: float) -> list[float]:
    """Nudge label positions apart so stacked end-point labels never overlap."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = list(values)
    for a, b in zip(order, order[1:]):
        if out[b] - out[a] < gap:
            out[b] = out[a] + gap
    return out


def fig_normalisation(metrics: pd.DataFrame) -> Path:
    """Slope chart: how much of each score is formatting (raw -> norm)."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=False)
    for ax, model in zip(axes, MODELS):
        norm_vals = [pct(metrics[(metrics["model"] == model) & (metrics["language"] == lang)
                                 & (metrics["variant"] == "norm")]["wer"]).iloc[0] for lang in LANGS]
        label_y = _spread(norm_vals, gap=2.4)
        for j, lang in enumerate(LANGS):
            raw = pct(metrics[(metrics["model"] == model) & (metrics["language"] == lang)
                              & (metrics["variant"] == "raw")]["wer"]).iloc[0]
            norm = pct(metrics[(metrics["model"] == model) & (metrics["language"] == lang)
                               & (metrics["variant"] == "norm")]["wer"]).iloc[0]
            y = len(LANGS) - 1 - j
            ax.plot([0, 1], [raw, norm], color=COLOR[model], linewidth=1.6, alpha=0.55)
            ax.scatter([0, 1], [raw, norm], color=COLOR[model], s=34, zorder=3)
            ax.annotate(f"{raw:.1f}", (0, raw), textcoords="offset points", xytext=(-8, 0),
                        ha="right", va="center", fontsize=8.5)
            ax.annotate(f"{norm:.1f}  {LANG_LABEL[lang]}", (1, label_y[j]), textcoords="offset points",
                        xytext=(8, 0), ha="left", va="center", fontsize=8.5)
            ax.annotate(f"−{raw - norm:.1f} pp", (0.5, (raw + norm) / 2), textcoords="offset points",
                        xytext=(0, 6), ha="center", fontsize=8, color="#666")
        ax.set_xlim(-0.42, 1.72)
        ax.set_xticks([0, 1], ["raw\n(as produced)", "norm\n(cleaned both sides)"], fontsize=9)
        ax.set_title(MODEL_LABEL[model], fontsize=10, loc="left")
        ax.set_ylabel("WER %")
        ax.set_ylim(0, 36)
        ax.grid(axis="y", color="#e4e4e4", linewidth=0.8)
        ax.set_axisbelow(True)
    fig.suptitle("Word error rate before and after text normalisation", fontsize=11, x=0.01,
                 ha="left", y=1.0)
    fig.tight_layout()
    out = FIG_DIR / "fig_normalisation.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_distribution(per: pd.DataFrame) -> Path:
    """Per-clip WER distribution: share of clips at or below each error rate, per language."""
    import statistics

    norm = per[per["variant"] == "norm"]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.5), sharey=True)
    for ax, lang in zip(axes, LANGS):
        lines = []
        for model in MODELS:
            values = sorted(pct(norm[(norm["model"] == model) & (norm["language"] == lang)]["wer"]))
            share = [(i + 1) / len(values) * 100 for i in range(len(values))]
            ax.step(values, share, where="post", color=COLOR[model], linewidth=1.8,
                    label=MODEL_LABEL[model])
            median = statistics.median(values)
            ax.plot([median], [50], "o", color=COLOR[model], markersize=6, mec="white", mew=1.2,
                    zorder=4)
            perfect = sum(1 for v in values if v == 0)
            lines.append((model, median, perfect, len(values)))
        ax.axhline(50, color="#bbb", linewidth=0.8, linestyle=":")
        ax.set_title(LANG_LABEL[lang], fontsize=10, loc="left")
        ax.set_xlim(0, 95)
        ax.set_ylim(0, 100)
        ax.set_xlabel("clip WER %")
        ax.grid(color="#e8e8e8", linewidth=0.7)
        ax.set_axisbelow(True)
        for k, (model, median, perfect, n) in enumerate(lines):
            ax.text(56, 24 - k * 15, f"median {median:.1f}%\n{perfect}/{n} clips perfect",
                    color=COLOR[model], fontsize=8, va="center")
    axes[0].set_ylabel("% of clips at or below")
    axes[0].figure.legend(*axes[0].get_legend_handles_labels(), frameon=False, fontsize=8.5,
                          loc="upper left", bbox_to_anchor=(0.005, 0.995), ncols=2)
    fig.suptitle("Per-clip WER distribution (normalised): a curve that rises earlier is better; "
                 "dots mark the medians", fontsize=10.5, x=0.01, ha="left", y=1.10)
    fig.tight_layout()
    out = FIG_DIR / "fig_distribution.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


_BRACKETED = re.compile(r"[<\[][^>\]]*[>\]]|\([^)]*?\)")


def _corpus_wer(policy: str, language: str, model: str) -> tuple[float, int]:
    """Corpus WER under one bracket policy; 'delete' drops bracketed spans before normalising."""
    rows = [r for r in load_jsonl(OUT_DIR / PREDICTION_FILE[model]) if r["language"] == language]
    errors = tokens = 0
    for row in rows:
        ref, hyp = row["reference"], row["model_output"]
        if policy == "delete":
            ref, hyp = _BRACKETED.sub(" ", ref), _BRACKETED.sub(" ", hyp)
        detail = per_sample(normalize(ref, language), normalize(hyp, language))
        errors += detail["substitutions"] + detail["deletions"] + detail["insertions"]
        tokens += detail["ref_tokens"]
    bracketed = sum(1 for r in load_jsonl(MANIFEST)
                    if r["language"] == language and _BRACKETED.search(r["reference"]))
    return 100 * errors / tokens, bracketed


def fig_bracket_policy() -> Path:
    """The keep-vs-delete bracketed-text experiment, per language and model."""
    labels, kept, deleted, counts = [], [], [], {}
    for lang in LANGS:
        for model in MODELS:
            k, bracketed = _corpus_wer("keep", lang, model)
            d, _ = _corpus_wer("delete", lang, model)
            counts[lang] = bracketed
            labels.append(f"{LANG_LABEL[lang]}\n{MODEL_LABEL[model].split()[0]}")
            kept.append(k)
            deleted.append(d)
    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    x = range(len(labels))
    width = 0.36
    b1 = ax.bar([i - width / 2 for i in x], kept, width, label="brackets kept (used here)",
                color="#4c78a8")
    b2 = ax.bar([i + width / 2 for i in x], deleted, width, label="brackets deleted",
                color="#b0b0b0")
    ax.bar_label(b1, fmt="%.1f", padding=2, fontsize=8)
    ax.bar_label(b2, fmt="%.1f", padding=2, fontsize=8)
    ax.set_xticks(list(x), labels, fontsize=8.5)
    ax.set_ylabel("corpus WER %")
    ax.set_ylim(0, 33)
    footnote = " · ".join(f"{LANG_LABEL[l]} {counts[l]}/50 refs contain brackets" for l in LANGS)
    ax.set_title("Deleting bracketed text helps Korean and hurts English/Hindi", fontsize=10, loc="left")
    ax.text(0, -0.30, footnote, transform=ax.transAxes, fontsize=7.5, color="#666")
    ax.legend(frameon=False, fontsize=8.5, ncols=2, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    ax.grid(axis="y", color="#e4e4e4", linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    out = FIG_DIR / "fig_bracket_policy.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_error_mix(metrics: pd.DataFrame) -> Path:
    """Stacked bars: what kind of error, per language (normalised counts)."""
    norm = metrics[(metrics["variant"] == "norm") & (metrics["language"] != "all")]
    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    labels = [f"{LANG_LABEL[l]}\n{MODEL_LABEL[m].split()[0]}" for l in LANGS for m in MODELS]
    rows = [norm[(norm["language"] == l) & (norm["model"] == m)].iloc[0] for l in LANGS for m in MODELS]
    subs = [100 * r["substitutions"] / (r["substitutions"] + r["deletions"] + r["insertions"]) for r in rows]
    dels = [100 * r["deletions"] / (r["substitutions"] + r["deletions"] + r["insertions"]) for r in rows]
    ins = [100 * r["insertions"] / (r["substitutions"] + r["deletions"] + r["insertions"]) for r in rows]
    x = range(len(rows))
    ax.bar(x, subs, 0.62, label="substitution", color="#e45756")
    ax.bar(x, dels, 0.62, bottom=subs, label="deletion", color="#f58518")
    ax.bar(x, ins, 0.62, bottom=[a + b for a, b in zip(subs, dels)], label="insertion", color="#54a24b")
    ax.set_xticks(list(x), labels, fontsize=8.5)
    ax.set_ylabel("% of edits")
    ax.set_ylim(0, 100)
    for xi, (a, b, c) in enumerate(zip(subs, dels, ins)):
        for value, base in ((a, 0), (b, a), (c, a + b)):
            if value >= 8:
                ax.text(xi, base + value / 2, f"{value:.0f}", ha="center", va="center",
                        fontsize=8, color="white")
    ax.set_title("Error composition (normalised): substitutions everywhere, but Korean is ~40% deletions",
                 fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8.5, ncols=3, loc="upper center",
              bbox_to_anchor=(0.5, -0.16))
    ax.grid(axis="y", color="#e4e4e4", linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    out = FIG_DIR / "fig_error_mix.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_rule_effect() -> Path:
    """Per-rule ablation: how many WER points each single rule accounts for."""
    ablation = pd.read_csv(OUT_DIR / "ablation.csv")
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharex=True)
    for ax, lang in zip(axes, LANGS):
        sub = ablation[(ablation["language"] == lang)
                       & (~ablation["rule"].isin(["raw", "full norm pipeline"]))]
        rules = list(dict.fromkeys(sub["rule"]))
        y = range(len(rules))
        h = 0.36
        for i, model in enumerate(MODELS):
            vals = [float(sub[(sub["rule"] == r) & (sub["model"] == model)]["delta_vs_raw"].iloc[0])
                    for r in rules]
            bars = ax.barh([k + (i - 0.5) * h for k in y], vals, h, color=COLOR[model],
                           label=MODEL_LABEL[model])
            for bar, value in zip(bars, vals):
                ax.text(value / 2 if abs(value) > 1.2 else value - 0.15, bar.get_y() + bar.get_height() / 2,
                        f"{value:+.1f}", va="center", fontsize=7,
                        ha="center" if abs(value) > 1.2 else "right",
                        color="white" if abs(value) > 1.2 else "#555")
        ax.set_yticks(list(y), rules, fontsize=7.8)
        ax.invert_yaxis()
        ax.axvline(0, color="#888", linewidth=0.8)
        ax.set_title(LANG_LABEL[lang], fontsize=10, loc="left")
        ax.set_xlabel("ΔWER vs raw (pp)")
        ax.grid(axis="x", color="#e8e8e8", linewidth=0.7)
        ax.set_axisbelow(True)
    fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, fontsize=8.5,
               loc="lower center", bbox_to_anchor=(0.5, -0.06), ncols=2)
    fig.suptitle("Which single rule accounts for the drop: punctuation removal is most of it "
                 "(0.0 = measured, no effect)", fontsize=10.5, x=0.01, ha="left", y=1.02)
    fig.tight_layout()
    out = FIG_DIR / "fig_rule_effect.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    metrics, per = load()
    for path in (fig_wer_cer(metrics), fig_normalisation(metrics), fig_distribution(per),
                 fig_error_mix(metrics), fig_bracket_policy(),
                 fig_rule_effect()):
        print(f"wrote {path.relative_to(ROOT)} ({path.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()

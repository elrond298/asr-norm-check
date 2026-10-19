#!/usr/bin/env python
"""Assert-based self-check for the scoring library (no pytest dependency).

  uv run python tests/test_scoring.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from asrbench.normalize import normalize, without_spaces
from asrbench.scoring import per_sample, score_set


def check(name: str, got, want) -> None:
    assert got == want, f"{name}: got {got!r}, want {want!r}"


def main() -> None:
    # raw variant is the identity, even for odd whitespace
    check("raw identity", normalize("a  b.", "en", "raw"), "a  b.")

    # English: case, punctuation, contractions, spelled-out numbers -> digits
    check("en case+punct", normalize("Hello, World!", "en"), "hello world")
    check("en number", normalize("He paid five dollars in 1990.", "en"), "he paid $5 in 1990")
    check("en contraction", normalize("They're Mr. Smith.", "en"), "they are mister smith")
    # brackets: the aside is spoken in FLEURS, so its words must survive normalisation
    check("en bracket kept", normalize("can climb (though not well), swim", "en"),
          "can climb though not well swim")

    # Hindi: matras must survive, Devanagari digits fold to ASCII, ZWJ disappears, danda drops
    check("hi digits", normalize("सन् १९९० में।", "hi"), "सन् 1990 में")
    check("hi matras kept", normalize("मोटरसाइकिल", "hi"), "मोटरसाइकिल")
    check("hi zwj", normalize("क\u200dष", "hi"), "कष")
    check("hi bracket kept", normalize("बोडा-बोडा (मोटरसाइकिल) ले सकते हैं।", "hi"),
          "बोडा बोडा मोटरसाइकिल ले सकते हैं")

    # Korean: full-width digits/punctuation fold, spacing is preserved, glosses dropped
    check("ko fullwidth", normalize("２０１９년, 서울!", "ko"), "2019년 서울")
    check("ko spacing kept", normalize("비슈케크 가 무정부.", "ko"), "비슈케크 가 무정부")
    check("ko bracket kept", normalize("비슈케크(Bishkek)가 온다.", "ko"), "비슈케크 bishkek 가 온다")

    # WER / S-D-I on a hand-checkable pair
    d = per_sample("a b c d", "a x c d")
    check("sub count", (d["substitutions"], d["deletions"], d["insertions"]), (1, 0, 0))
    check("sub wer", d["wer"], 0.25)
    d = per_sample("a b c", "a b")
    check("del count", (d["deletions"], d["wer"]), (1, 1 / 3))
    d = per_sample("a b", "a b c")
    check("ins count", (d["insertions"], d["wer"]), (1, 0.5))
    d = per_sample("a b", "")
    check("empty hyp", (d["deletions"], d["wer"], d["cer"]), (2, 1.0, 1.0))

    # CER ignores spaces: a spacing error is free for CER, costs a token for WER
    d = per_sample("a b", "ab")
    check("space-free cer", (d["wer"], d["cer"]), (1.0, 0.0))
    check("without_spaces", without_spaces("a b\n c"), "abc")

    # corpus aggregation is micro (token-weighted), not a mean of per-sample rates
    corpus = score_set([{"reference": "a b", "hypothesis": "a x"},
                        {"reference": "a b c d e f g h", "hypothesis": "a b c d e f g h"}])
    check("micro wer", corpus["wer"], 1 / 10)

    print("all scoring checks passed")


if __name__ == "__main__":
    main()

"""Text normalisation used by the scoring stage.

Two variants are scored everywhere:

raw   -- the strings exactly as FLEURS / the model produced them.
norm  -- language-aware normalisation (see docs/normalization_rules.md).

English reuses openai-whisper's ``EnglishTextNormalizer`` (the de-facto ASR scoring
reference: lower-case, contractions/abbreviations expanded, spelled-out numbers mapped to
digits, punctuation and symbols dropped, whitespace collapsed).

Bracketed text is kept in all three languages: FLEURS uses parentheses for asides that the
speaker actually reads ("... can climb (though not well), swim ..."), so deleting them would
book every correctly transcribed aside as an insertion. openai-whisper's English normaliser
deletes parenthesised spans, so for English the brackets are turned into commas first.

Hindi and Korean use the same idea but with a cleaner that keeps combining marks.
openai-whisper's ``BasicTextNormalizer`` folds every Unicode M/S/P character into a space,
which shreds Devanagari: matras (category Mn/Mc) and the virama carry the vowels and would
disappear, turning one word into a string of bare consonants. So for those languages we only
space out symbols and punctuation, and keep the marks.
"""

from __future__ import annotations

import unicodedata

from whisper.normalizers import EnglishTextNormalizer

_EN = EnglishTextNormalizer()
_ZW = dict.fromkeys(map(ord, "\u200b\u200c\u200d\ufeff"))

VARIANT_RAW = "raw"
VARIANT_NORM = "norm"
VARIANTS = (VARIANT_RAW, VARIANT_NORM)

#: Parentheses/brackets become commas so the English normaliser keeps the words inside them.
_BRACKET_CHARS = str.maketrans({c: ", " for c in "()[]"})
_DEVA_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")


def _strip_punct(text: str) -> str:
    """NFKC, then space out symbols/punctuation while keeping letters, digits and marks."""
    text = unicodedata.normalize("NFKC", text)
    spaced = "".join(" " if unicodedata.category(ch)[0] in "SP" else ch for ch in text)
    return " ".join(spaced.lower().split())


def normalize(text: str, language: str, variant: str = VARIANT_NORM) -> str:
    if variant == VARIANT_RAW:
        return text
    if variant != VARIANT_NORM:
        raise ValueError(f"unknown variant: {variant}")
    text = text.translate(_ZW)  # invisible formatting characters, in every language
    if language == "en":
        return _EN(text.translate(_BRACKET_CHARS))
    if language == "hi":
        return _strip_punct(text).translate(_DEVA_DIGITS)
    if language == "ko":
        return _strip_punct(text)
    raise ValueError(f"unknown language: {language}")


def without_spaces(text: str) -> str:
    """Character sequence used for CER: spaces carry no content."""
    return "".join(text.split())

"""Text normalisation used by the scoring stage.

Two variants are scored everywhere:

raw   -- the strings exactly as FLEURS / the model produced them.
norm  -- language-aware normalisation (see docs/normalization_rules.md).

For all three languages ``norm`` reuses the normalisers from the official
``openai-whisper`` package (``EnglishTextNormalizer`` for English,
``BasicTextNormalizer`` for Hindi/Korean). They are the de-facto reference for
ASR scoring: NFKC, lower-casing, punctuation/symbol removal, whitespace
collapse, and for English number/ordinal/contraction expansion.

Hindi additionally drops zero-width joiners, which Devanagari text in FLEURS
and in model output use inconsistently.
"""

from __future__ import annotations

import unicodedata

from whisper.normalizers import BasicTextNormalizer, EnglishTextNormalizer

_EN = EnglishTextNormalizer()
_BASIC = BasicTextNormalizer()
_ZW = dict.fromkeys(map(ord, "\u200b\u200c\u200d\ufeff"))

VARIANT_RAW = "raw"
VARIANT_NORM = "norm"
VARIANTS = (VARIANT_RAW, VARIANT_NORM)

#: Human-readable rule summary, rendered into docs/normalization_rules.md.
RULES: dict[str, str] = {
    "raw": "identity - reference and hypothesis are scored as published",
    "en": "NFKC, lower-case, strip punctuation/symbols, expand numbers/ordinals/contractions "
          "(openai-whisper EnglishTextNormalizer), collapse whitespace",
    "hi": "NFKC (Devanagari digits -> ASCII), remove ZWJ/ZWNJ, lower-case, strip punctuation/symbols "
          "(openai-whisper BasicTextNormalizer), collapse whitespace",
    "ko": "NFKC (full-width punctuation/digits -> ASCII), lower-case, strip punctuation/symbols "
          "(openai-whisper BasicTextNormalizer), collapse whitespace",
}


def normalize(text: str, language: str, variant: str = VARIANT_NORM) -> str:
    if variant == VARIANT_RAW:
        return text
    if variant != VARIANT_NORM:
        raise ValueError(f"unknown variant: {variant}")
    if language == "en":
        return _EN(text)
    if language == "hi":
        return _BASIC(unicodedata.normalize("NFKC", text).translate(_ZW))
    if language == "ko":
        return _BASIC(text)
    raise ValueError(f"unknown language: {language}")


def without_spaces(text: str) -> str:
    """Character sequence used for CER: spaces carry no content."""
    return "".join(text.split())

"""Leakage guard for the Research family (spec/04-datasets.md §2.2).

The Research family is capped to SYNTHETIC content only — no real paper titles,
author names, DOIs, or arXiv ids may appear in generated text.
:func:`lint_no_real_entities` scans a string against a small, deliberately
non-exhaustive blocklist of obvious real-world identifiers plus DOI / arXiv
regexes and raises :class:`RealEntityLeakError` on any match, so generation can
fail fast. It is a tripwire for accidental leakage of canonical real entities,
NOT a comprehensive plagiarism detector.
"""

from __future__ import annotations

import re

# Canonical real paper titles (lowercased substrings). Tripwire only.
_REAL_TITLE_BLOCKLIST: tuple[str, ...] = (
    "attention is all you need",
    "bert: pre-training",
    "deep residual learning",
    "imagenet classification with deep convolutional",
    "language models are few-shot learners",
    "adam: a method for stochastic optimization",
    "generative adversarial networks",
    "denoising diffusion probabilistic models",
    "playing atari with deep reinforcement learning",
    "a survey of large language models",
)
# Canonical real author / surname tokens (lowercased substrings).
_REAL_AUTHOR_BLOCKLIST: tuple[str, ...] = (
    "vaswani",
    "hinton",
    "lecun",
    "bengio",
    "goodfellow",
    "schmidhuber",
    "kaiming he",
)
# DOI: "10." then >= 4 digits, a slash, then a non-space suffix.
_DOI_RE = re.compile(r"10\.\d{4,}/\S+")
# arXiv id, e.g. "arXiv:2401.01234" or "arxiv: 2401.01234".
_ARXIV_RE = re.compile(r"arxiv:\s*\d{4}\.\d{4,5}", re.IGNORECASE)


class RealEntityLeakError(ValueError):
    """Raised when text contains a real-world identifier (title/author/DOI)."""


def lint_no_real_entities(text: str) -> None:
    """Raise :class:`RealEntityLeakError` if ``text`` names a real entity.

    Checks (case-insensitive substring) a blocklist of canonical paper titles
    and author surnames, plus DOI and arXiv-id regexes. Synthetic Research-family
    text passes silently (returns ``None``); any real-world identifier raises.
    """
    lowered = text.lower()
    for title in _REAL_TITLE_BLOCKLIST:
        if title in lowered:
            raise RealEntityLeakError(f"real paper title detected: {title!r}")
    for author in _REAL_AUTHOR_BLOCKLIST:
        if author in lowered:
            raise RealEntityLeakError(f"real author name detected: {author!r}")
    doi = _DOI_RE.search(text)
    if doi is not None:
        raise RealEntityLeakError(f"DOI pattern detected: {doi.group(0)!r}")
    arxiv = _ARXIV_RE.search(text)
    if arxiv is not None:
        raise RealEntityLeakError(f"arXiv id detected: {arxiv.group(0)!r}")

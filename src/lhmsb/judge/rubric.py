"""Versioned fixed rubric for the LLM judge.

The rubric text and its version live in a markdown file (default
``configs/judge_rubric.md``).  The version is encoded in a machine-readable HTML
comment at the top of that file::

    <!-- rubric-version: 1.0.0 -->

Encoding the version in the rubric file (rather than in source) means every
recorded ``JudgeScore`` is traceable to an exact rubric revision, and changing
the criteria forces a version bump in one place.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_VERSION_RE = re.compile(r"<!--\s*rubric-version:\s*([^\s>]+)\s*-->", re.IGNORECASE)


@dataclass(frozen=True)
class Rubric:
    """A fixed, versioned scoring rubric.

    Attributes:
        version: The rubric version string (e.g. ``"1.0.0"``), parsed from the
            ``rubric-version`` metadata marker.  Recorded on every JudgeScore.
        criteria: The full rubric text (the markdown body) given to the judge.
        source_path: Path the rubric was loaded from (for audit provenance).
    """

    version: str
    criteria: str
    source_path: str


def parse_rubric_version(text: str) -> str:
    """Extract the ``rubric-version`` marker from rubric markdown.

    Raises:
        ValueError: if no ``<!-- rubric-version: ... -->`` marker is present.
    """
    match = _VERSION_RE.search(text)
    if match is None:
        raise ValueError(
            "rubric is missing a '<!-- rubric-version: X.Y.Z -->' marker; "
            "every rubric MUST declare a version for auditability"
        )
    return match.group(1)


def load_rubric(path: str) -> Rubric:
    """Load a :class:`Rubric` from a markdown file.

    Args:
        path: Path to the rubric markdown file (e.g. ``configs/judge_rubric.md``).

    Returns:
        A frozen :class:`Rubric` with the parsed version and full criteria text.

    Raises:
        FileNotFoundError: if ``path`` does not exist.
        ValueError: if the file lacks a ``rubric-version`` marker.
    """
    text = Path(path).read_text(encoding="utf-8")
    version = parse_rubric_version(text)
    return Rubric(version=version, criteria=text, source_path=path)

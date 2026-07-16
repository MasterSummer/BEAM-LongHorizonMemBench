"""Leak-safe public continuation records for real policy evaluation."""

from __future__ import annotations

import ast
import hashlib
import io
import json
import random
import re
import tokenize
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, cast

from lhmsb.longhorizon.schema import ContinuationOpportunity


class SurfaceLeakError(ValueError):
    """Raised when evaluator-only information crosses the public boundary."""


@dataclass(frozen=True)
class SurfaceLeak:
    """One forbidden token found at a recursive public-payload path."""

    path: str
    matched: str
    category: str


@dataclass(frozen=True)
class SurfaceValidationReport:
    """Successful public-surface validation summary."""

    scanned_strings: int
    scanned_keys: int


@dataclass(frozen=True)
class SurfaceLeakPolicy:
    """Dataset-specific information-firewall vocabulary."""

    forbidden_state_ids: tuple[str, ...] = ()
    forbidden_action_ids: tuple[str, ...] = ()
    answer_revealing_phrases: tuple[str, ...] = ()
    forbidden_field_names: tuple[str, ...] = (
        "valid_action_ids",
        "satisfies_state_ids",
        "violates_state_ids",
        "global_utility",
        "local_utility",
        "source_event_ids",
        "recoverability_by_state",
        "dependency_ids",
        "focal_state_ids",
        "required_state_ids",
        "intervention_target_ids",
        "option_to_action",
    )
    validity_labels: tuple[str, ...] = (
        "revoked",
        "invalidated",
        "validity label",
        "stale-state",
    )


@dataclass(frozen=True)
class PublicActionOption:
    """Opaque action option visible to a policy model."""

    option_id: str
    files: tuple[tuple[str, str], ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "option_id": self.option_id,
            "files": [list(pair) for pair in self.files],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> PublicActionOption:
        raw_files = data.get("files", ())
        if not isinstance(raw_files, Sequence) or isinstance(raw_files, (str, bytes)):
            raise TypeError("files must be a sequence of path/content pairs")
        files: list[tuple[str, str]] = []
        for raw_pair in raw_files:
            if (
                not isinstance(raw_pair, Sequence)
                or isinstance(raw_pair, (str, bytes))
                or len(raw_pair) != 2
            ):
                raise TypeError("files must contain path/content pairs")
            files.append((str(raw_pair[0]), str(raw_pair[1])))
        return cls(option_id=str(data["option_id"]), files=tuple(files))


@dataclass(frozen=True)
class PublicContinuation:
    """Gold-free continuation request and its opaque action options."""

    opportunity_id: str
    checkpoint_session: int
    request: str
    options: tuple[PublicActionOption, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "opportunity_id": self.opportunity_id,
            "checkpoint_session": self.checkpoint_session,
            "request": self.request,
            "options": [option.to_dict() for option in self.options],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> PublicContinuation:
        raw_options = data.get("options", ())
        if not isinstance(raw_options, Sequence) or isinstance(raw_options, (str, bytes)):
            raise TypeError("options must be a sequence")
        options: list[PublicActionOption] = []
        for raw_option in raw_options:
            if not isinstance(raw_option, Mapping):
                raise TypeError("options must contain objects")
            options.append(PublicActionOption.from_dict(raw_option))
        checkpoint = data["checkpoint_session"]
        if isinstance(checkpoint, bool) or not isinstance(checkpoint, (int, str)):
            raise TypeError("checkpoint_session must be an integer")
        return cls(
            opportunity_id=str(data["opportunity_id"]),
            checkpoint_session=int(checkpoint),
            request=str(data["request"]),
            options=tuple(options),
        )


@dataclass(frozen=True)
class EvaluatorContinuation:
    """Private mapping from opaque public options to latent actions."""

    opportunity_id: str
    option_to_action: tuple[tuple[str, str], ...]

    def action_for_option(self, option_id: str) -> str:
        try:
            return dict(self.option_to_action)[option_id]
        except KeyError as exc:
            raise KeyError(f"unknown public option: {option_id}") from exc

    def to_dict(self) -> dict[str, object]:
        return {
            "opportunity_id": self.opportunity_id,
            "option_to_action": [list(pair) for pair in self.option_to_action],
        }


def canonical_public_json(value: object) -> str:
    """Serialize a public dataclass/payload with stable JSON rules."""
    payload = _public_jsonable(value)
    return json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"))


def _public_jsonable(value: object) -> object:
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _public_jsonable(to_dict())
    if is_dataclass(value) and not isinstance(value, type):
        return _public_jsonable(asdict(cast(Any, value)))
    if isinstance(value, Mapping):
        return {str(key): _public_jsonable(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_public_jsonable(child) for child in value]
    return value


def public_surface_hash(value: object) -> str:
    """Hash the exact canonical public payload."""
    return hashlib.sha256(canonical_public_json(value).encode("utf-8")).hexdigest()


def _docstring_lines(source: str) -> set[int]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    lines: set[int] = set()
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if not isinstance(body, list) or not body:
            continue
        first = body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            end = first.end_lineno or first.lineno
            lines.update(range(first.lineno, end + 1))
    return lines


def strip_python_evaluator_hints(source: str) -> str:
    """Remove Python comments and statement docstrings while preserving code."""
    docstring_lines = _docstring_lines(source)
    tokens: list[tokenize.TokenInfo] = []
    try:
        stream = tokenize.generate_tokens(io.StringIO(source).readline)
        for token in stream:
            if token.type == tokenize.COMMENT:
                continue
            if token.type == tokenize.STRING and token.start[0] in docstring_lines:
                continue
            tokens.append(token)
    except (IndentationError, tokenize.TokenError):
        return source
    rendered = tokenize.untokenize(tokens)
    return rendered.rstrip() + "\n"


def _matches_identifier(text: str, identifier: str) -> bool:
    pattern = rf"(?<![A-Za-z0-9_]){re.escape(identifier)}(?![A-Za-z0-9_])"
    return re.search(pattern, text) is not None


def validate_public_payload(
    payload: object,
    policy: SurfaceLeakPolicy,
) -> SurfaceValidationReport:
    """Recursively reject evaluator IDs, fields, labels, and answer hints."""
    leaks: list[SurfaceLeak] = []
    scanned_strings = 0
    scanned_keys = 0
    forbidden_keys = {item.casefold() for item in policy.forbidden_field_names}

    def inspect_text(text: str, path: str) -> None:
        nonlocal scanned_strings
        scanned_strings += 1
        for identifier in (*policy.forbidden_state_ids, *policy.forbidden_action_ids):
            if identifier and _matches_identifier(text, identifier):
                leaks.append(SurfaceLeak(path, identifier, "gold_identifier"))
        lowered = text.casefold()
        for phrase in (*policy.answer_revealing_phrases, *policy.validity_labels):
            if phrase and phrase.casefold() in lowered:
                leaks.append(SurfaceLeak(path, phrase, "forbidden_phrase"))

    def walk(value: object, path: str) -> None:
        nonlocal scanned_keys
        if isinstance(value, Mapping):
            for raw_key, child in value.items():
                key = str(raw_key)
                scanned_keys += 1
                if key.casefold() in forbidden_keys:
                    leaks.append(SurfaceLeak(f"{path}.{key}", key, "evaluator_field"))
                inspect_text(key, f"{path}.<key>")
                walk(child, f"{path}.{key}")
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                walk(child, f"{path}[{index}]")
        elif isinstance(value, str):
            inspect_text(value, path)
        elif is_dataclass(value) and not isinstance(value, type):
            walk(asdict(cast(Any, value)), path)

    walk(payload, "$")
    if leaks:
        first = leaks[0]
        raise SurfaceLeakError(
            f"public surface leak at {first.path}: {first.category} {first.matched!r}"
        )
    return SurfaceValidationReport(scanned_strings=scanned_strings, scanned_keys=scanned_keys)


def render_public_continuation(
    *,
    episode_id: str,
    semantic_seed: int,
    opportunity: ContinuationOpportunity,
) -> tuple[PublicContinuation, EvaluatorContinuation]:
    """Neutralize and deterministically permute one latent action catalog."""
    ordered = list(opportunity.action_catalog)
    seed_payload = f"{episode_id}|{opportunity.opportunity_id}|{semantic_seed}".encode()
    seed = int.from_bytes(hashlib.sha256(seed_payload).digest()[:8], "big")
    random.Random(seed).shuffle(ordered)
    options: list[PublicActionOption] = []
    mapping: list[tuple[str, str]] = []
    for index, action in enumerate(ordered, start=1):
        option_id = f"option-{index:02d}"
        files = tuple(
            (
                path,
                strip_python_evaluator_hints(content) if path.endswith(".py") else content,
            )
            for path, content in action.files
        )
        options.append(PublicActionOption(option_id=option_id, files=files))
        mapping.append((option_id, action.action_id))
    public = PublicContinuation(
        opportunity_id=opportunity.opportunity_id,
        checkpoint_session=opportunity.checkpoint_session,
        request=opportunity.request,
        options=tuple(options),
    )
    policy = SurfaceLeakPolicy(
        forbidden_state_ids=opportunity.focal_state_ids,
        forbidden_action_ids=tuple(action.action_id for action in opportunity.action_catalog),
        answer_revealing_phrases=("correct action", "globally correct", "forbidden"),
    )
    validate_public_payload(public, policy)
    return public, EvaluatorContinuation(
        opportunity_id=opportunity.opportunity_id,
        option_to_action=tuple(mapping),
    )


__all__ = [
    "EvaluatorContinuation",
    "PublicActionOption",
    "PublicContinuation",
    "SurfaceLeak",
    "SurfaceLeakError",
    "SurfaceLeakPolicy",
    "SurfaceValidationReport",
    "canonical_public_json",
    "public_surface_hash",
    "render_public_continuation",
    "strip_python_evaluator_hints",
    "validate_public_payload",
]

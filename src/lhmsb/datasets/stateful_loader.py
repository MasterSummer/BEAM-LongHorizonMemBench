"""Strict loader for frozen state-first Software vertical datasets."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path

from lhmsb.datasets.stateful_pipeline import (
    STATEFUL_SCHEMA_VERSION,
    StatefulDatasetError,
    StatefulManifest,
    StatefulVerifyReport,
    verify_stateful,
)
from lhmsb.families.software.vertical import SoftwareVerticalSpec
from lhmsb.longhorizon.render import surfaces_hash
from lhmsb.longhorizon.replay import plan_hash
from lhmsb.longhorizon.schema import ActionSpec, EpisodePlan


def load_software_vertical_specs(
    frozen: Path,
    *,
    verify: bool = True,
) -> tuple[SoftwareVerticalSpec, ...]:
    """Load Software vertical specs without invoking the dataset generator."""
    root = frozen.resolve()
    if verify:
        report = verify_stateful(root)
        if not report.ok:
            raise StatefulDatasetError(_verification_message(report))
    manifest = _read_manifest(root)
    if manifest.schema_version != STATEFUL_SCHEMA_VERSION:
        raise StatefulDatasetError(
            f"unsupported stateful schema version: {manifest.schema_version}"
        )
    if manifest.family != "software":
        raise StatefulDatasetError(f"unsupported stateful family: {manifest.family}")
    records = _read_jsonl(root / "episodes.jsonl")
    specs = tuple(_spec_from_record(record) for record in records)
    _validate_records(specs, records, manifest)
    return specs


def _verification_message(report: StatefulVerifyReport) -> str:
    details: list[str] = []
    details.extend(f"checksum mismatch: {path}" for path, _, _ in report.mismatches)
    details.extend(f"missing file: {path}" for path in report.missing)
    return "stateful checksum verification failed: " + "; ".join(details)


def _read_manifest(root: Path) -> StatefulManifest:
    data = _read_json(root / "MANIFEST.json")
    try:
        return StatefulManifest.from_dict(data)
    except (KeyError, TypeError, ValueError) as exc:
        raise StatefulDatasetError(f"malformed stateful manifest: {exc}") from exc


def _read_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StatefulDatasetError(f"cannot read JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise StatefulDatasetError(f"expected JSON object: {path}")
    return {str(key): item for key, item in value.items()}


def _read_jsonl(path: Path) -> tuple[dict[str, object], ...]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise StatefulDatasetError(f"cannot read JSONL file {path}: {exc}") from exc
    records: list[dict[str, object]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise StatefulDatasetError(
                f"invalid JSONL record {path}:{line_number}: {exc}"
            ) from exc
        if not isinstance(value, dict):
            raise StatefulDatasetError(f"expected JSON object at {path}:{line_number}")
        records.append({str(key): item for key, item in value.items()})
    return tuple(records)


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise StatefulDatasetError(f"{label} must be a JSON object")
    return {str(key): item for key, item in value.items()}


def _sequence(value: object, label: str) -> Sequence[object]:
    if not isinstance(value, (list, tuple)):
        raise StatefulDatasetError(f"{label} must be a JSON array")
    return value


def _pairs(value: object, label: str) -> tuple[tuple[str, str], ...]:
    output: list[tuple[str, str]] = []
    for index, item in enumerate(_sequence(value, label)):
        pair = _sequence(item, f"{label}[{index}]")
        if len(pair) != 2:
            raise StatefulDatasetError(f"{label}[{index}] must contain exactly two values")
        output.append((str(pair[0]), str(pair[1])))
    return tuple(output)


def _spec_from_record(record: Mapping[str, object]) -> SoftwareVerticalSpec:
    try:
        plan = EpisodePlan.from_dict(_mapping(record["plan"], "plan"))
        actions = tuple(
            ActionSpec.from_dict(_mapping(item, "action"))
            for item in _sequence(record["actions"], "actions")
        )
        return SoftwareVerticalSpec(
            plan=plan,
            package_files=_pairs(record["package_files"], "package_files"),
            hidden_tests=_pairs(record["hidden_tests"], "hidden_tests"),
            actions=actions,
            surface_hash=str(record["surface_hash"]),
        )
    except StatefulDatasetError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise StatefulDatasetError(f"malformed frozen episode record: {exc}") from exc


def _validate_records(
    specs: tuple[SoftwareVerticalSpec, ...],
    records: tuple[dict[str, object], ...],
    manifest: StatefulManifest,
) -> None:
    if len(specs) != manifest.n_episodes:
        raise StatefulDatasetError(
            f"episode count mismatch: records={len(specs)} manifest={manifest.n_episodes}"
        )
    manifest_by_id: dict[str, Mapping[str, object]] = {}
    for entry in manifest.episodes:
        episode_id = str(entry.get("episode_id", ""))
        if not episode_id or episode_id in manifest_by_id:
            raise StatefulDatasetError(f"duplicate or empty manifest episode ID: {episode_id!r}")
        manifest_by_id[episode_id] = entry
    seen: set[str] = set()
    for spec, record in zip(specs, records, strict=True):
        episode_id = spec.plan.episode_id
        if episode_id in seen:
            raise StatefulDatasetError(f"duplicate frozen episode ID: {episode_id}")
        seen.add(episode_id)
        manifest_entry = manifest_by_id.get(episode_id)
        if manifest_entry is None:
            raise StatefulDatasetError(f"episode missing from manifest: {episode_id}")
        _validate_record(spec, record, manifest_entry)
    if seen != set(manifest_by_id):
        missing = sorted(set(manifest_by_id) - seen)
        raise StatefulDatasetError(f"manifest episodes missing from records: {missing}")


def _validate_record(
    spec: SoftwareVerticalSpec,
    record: Mapping[str, object],
    manifest_entry: Mapping[str, object],
) -> None:
    plan = spec.plan
    expected_scalars = {
        "episode_id": plan.episode_id,
        "semantic_seed": plan.semantic_seed,
        "trajectory_seed": plan.trajectory_seed,
        "n_sessions": plan.n_sessions,
    }
    for field, actual in expected_scalars.items():
        if record.get(field) != actual:
            raise StatefulDatasetError(
                f"{plan.episode_id} record {field} mismatch: "
                f"expected {actual!r}, got {record.get(field)!r}"
            )
        if manifest_entry.get(field) != actual:
            raise StatefulDatasetError(
                f"{plan.episode_id} manifest {field} mismatch: "
                f"expected {actual!r}, got {manifest_entry.get(field)!r}"
            )
    hashes = {
        "plan_hash": plan_hash(plan),
        "surface_hash": surfaces_hash(plan.sessions),
        "workspace_hash": _hash_json([asdict(item) for item in plan.workspaces]),
    }
    labels = {
        "plan_hash": "plan hash",
        "surface_hash": "surface hash",
        "workspace_hash": "workspace hash",
    }
    for field, actual in hashes.items():
        if str(record.get(field, "")) != actual:
            raise StatefulDatasetError(
                f"{plan.episode_id} record {labels[field]} mismatch: "
                f"expected {actual}, got {record.get(field)!r}"
            )
        if str(manifest_entry.get(field, "")) != actual:
            raise StatefulDatasetError(
                f"{plan.episode_id} manifest {labels[field]} mismatch: "
                f"expected {actual}, got {manifest_entry.get(field)!r}"
            )
    if spec.surface_hash != hashes["surface_hash"]:
        raise StatefulDatasetError(
            f"{plan.episode_id} spec surface hash mismatch: "
            f"expected {hashes['surface_hash']}, got {spec.surface_hash}"
        )


def _hash_json(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


__all__ = ["load_software_vertical_specs"]

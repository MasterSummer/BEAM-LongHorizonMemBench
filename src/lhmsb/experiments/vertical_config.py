"""Configuration and immutable task identities for the vertical offline pilot."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import yaml

from lhmsb.families.software.vertical import SoftwareVerticalSpec
from lhmsb.longhorizon.runner import VerticalCondition

VERTICAL_EXPERIMENT_SCHEMA_VERSION = 1
_CONDITIONS = {"workspace_only", "oracle_current_state", "fake_native"}


class VerticalExperimentError(ValueError):
    """Raised when a vertical experiment contract is malformed or inconsistent."""


@dataclass(frozen=True)
class VerticalOfflineConfig:
    """Normalized, order-preserving offline matrix configuration."""

    schema_version: int
    experiment_id: str
    conditions: tuple[tuple[VerticalCondition, tuple[str | None, ...]], ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "experiment_id": self.experiment_id,
            "conditions": [
                {
                    "condition": condition,
                    "interventions": list(interventions),
                }
                for condition, interventions in self.conditions
            ],
        }

    @property
    def config_hash(self) -> str:
        return canonical_hash(self.to_dict())


@dataclass(frozen=True)
class GitSnapshot:
    """The code state bound into one experiment identity."""

    commit: str
    dirty: bool
    ref: str


@dataclass(frozen=True)
class VerticalTask:
    """One independently executable episode/condition/intervention cell."""

    task_index: int
    task_id: str
    episode_id: str
    condition: VerticalCondition
    intervention_state_id: str | None
    run_identity: str
    task_payload_hash: str

    def __post_init__(self) -> None:
        if self.task_index < 0:
            raise VerticalExperimentError("task_index must be non-negative")
        if not self.task_id or not self.episode_id:
            raise VerticalExperimentError("task_id and episode_id must be non-empty")
        if self.condition not in _CONDITIONS:
            raise VerticalExperimentError(f"unknown vertical condition: {self.condition!r}")
        if self.condition != "fake_native" and self.intervention_state_id is not None:
            raise VerticalExperimentError("interventions are supported only for fake_native")
        if not self.run_identity or not self.task_payload_hash:
            raise VerticalExperimentError("task identities must be non-empty")

    def to_dict(self) -> dict[str, object]:
        return {
            "task_index": self.task_index,
            "task_id": self.task_id,
            "episode_id": self.episode_id,
            "condition": self.condition,
            "intervention_state_id": self.intervention_state_id,
            "run_identity": self.run_identity,
            "task_payload_hash": self.task_payload_hash,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> VerticalTask:
        condition = str(data.get("condition", ""))
        if condition not in _CONDITIONS:
            raise VerticalExperimentError(f"unknown vertical condition: {condition!r}")
        intervention = data.get("intervention_state_id")
        if intervention is not None and not isinstance(intervention, str):
            raise VerticalExperimentError("intervention_state_id must be a string or null")
        try:
            task_index = _integer(data["task_index"], "task_index")
            task_id = str(data["task_id"])
            episode_id = str(data["episode_id"])
            run_identity = str(data["run_identity"])
            task_payload_hash = str(data["task_payload_hash"])
        except (KeyError, TypeError, ValueError) as exc:
            raise VerticalExperimentError(f"malformed vertical task: {exc}") from exc
        return cls(
            task_index=task_index,
            task_id=task_id,
            episode_id=episode_id,
            condition=cast(VerticalCondition, condition),
            intervention_state_id=intervention,
            run_identity=run_identity,
            task_payload_hash=task_payload_hash,
        )


class _UniqueKeyLoader(yaml.SafeLoader):  # type: ignore[misc]
    """Safe YAML loader that rejects duplicate mapping keys."""


def _construct_unique_mapping(
    loader: _UniqueKeyLoader,
    node: yaml.nodes.MappingNode,
    deep: bool = False,
) -> dict[object, object]:
    loader.flatten_mapping(node)
    output: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in output:
            raise VerticalExperimentError(f"duplicate key in YAML configuration: {key!r}")
        output[key] = loader.construct_object(value_node, deep=deep)
    return output


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def load_vertical_offline_config(path: Path) -> VerticalOfflineConfig:
    """Parse and normalize one offline pilot YAML configuration."""
    try:
        raw = yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    except VerticalExperimentError:
        raise
    except (OSError, yaml.YAMLError) as exc:
        raise VerticalExperimentError(f"cannot read vertical config {path}: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise VerticalExperimentError("vertical config must be a YAML mapping")
    schema_version = _integer(raw.get("schema_version"), "schema_version")
    if schema_version != VERTICAL_EXPERIMENT_SCHEMA_VERSION:
        raise VerticalExperimentError(
            f"unsupported vertical experiment schema version: {schema_version}"
        )
    experiment_id = raw.get("experiment_id")
    if not isinstance(experiment_id, str) or not experiment_id.strip():
        raise VerticalExperimentError("experiment_id must be a non-empty string")
    raw_conditions = raw.get("conditions")
    if not isinstance(raw_conditions, Mapping) or not raw_conditions:
        raise VerticalExperimentError("conditions must be a non-empty mapping")
    conditions: list[tuple[VerticalCondition, tuple[str | None, ...]]] = []
    for raw_condition, raw_interventions in raw_conditions.items():
        condition = str(raw_condition)
        if condition not in _CONDITIONS:
            raise VerticalExperimentError(f"unknown vertical condition: {condition!r}")
        interventions = _interventions(raw_interventions, condition)
        conditions.append((cast(VerticalCondition, condition), interventions))
    return VerticalOfflineConfig(
        schema_version=schema_version,
        experiment_id=experiment_id.strip(),
        conditions=tuple(conditions),
    )


def _integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise VerticalExperimentError(f"{label} must be an integer")
    return value


def _interventions(
    value: object,
    condition: str,
) -> tuple[str | None, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise VerticalExperimentError(f"{condition} interventions must be a YAML array")
    if not value:
        raise VerticalExperimentError(f"{condition} has an empty intervention matrix")
    output: list[str | None] = []
    for intervention in value:
        if intervention is not None and not isinstance(intervention, str):
            raise VerticalExperimentError("interventions must contain only string or null values")
        if intervention in output:
            raise VerticalExperimentError(
                f"duplicate intervention for {condition}: {intervention!r}"
            )
        output.append(intervention)
    if condition != "fake_native" and output != [None]:
        raise VerticalExperimentError("interventions are supported only for fake_native")
    return tuple(output)


def build_vertical_tasks(
    specs: Sequence[SoftwareVerticalSpec],
    config: VerticalOfflineConfig,
    *,
    run_identity: str,
) -> tuple[VerticalTask, ...]:
    """Expand episodes and the ordered matrix into stable atomic tasks."""
    tasks: list[VerticalTask] = []
    for spec in specs:
        state_ids = {state.state_id for state in spec.plan.state_units}
        for condition, interventions in config.conditions:
            for intervention in interventions:
                if intervention is not None and intervention not in state_ids:
                    raise VerticalExperimentError(
                        f"unknown intervention state {intervention!r} "
                        f"for episode {spec.plan.episode_id}"
                    )
                index = len(tasks)
                payload = {
                    "task_index": index,
                    "episode_id": spec.plan.episode_id,
                    "condition": condition,
                    "intervention_state_id": intervention,
                    "run_identity": run_identity,
                }
                task_hash = canonical_hash(payload)
                task_id = (
                    f"{index:05d}-{_slug(spec.plan.episode_id)}-"
                    f"{condition.replace('_', '-')}-{_slug(intervention or 'baseline')}"
                )
                tasks.append(
                    VerticalTask(
                        task_index=index,
                        task_id=task_id,
                        episode_id=spec.plan.episode_id,
                        condition=condition,
                        intervention_state_id=intervention,
                        run_identity=run_identity,
                        task_payload_hash=task_hash,
                    )
                )
    if not tasks:
        raise VerticalExperimentError("vertical task matrix is empty")
    return tuple(tasks)


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_.-]+", "-", value).strip("-")
    if not slug:
        raise VerticalExperimentError(f"cannot build task ID from {value!r}")
    return slug


def canonical_json(value: object) -> str:
    """Serialize JSON-compatible data with stable cross-process bytes."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def canonical_hash(value: object) -> str:
    """Return SHA-256 over :func:`canonical_json`."""
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


__all__ = [
    "GitSnapshot",
    "VERTICAL_EXPERIMENT_SCHEMA_VERSION",
    "VerticalExperimentError",
    "VerticalOfflineConfig",
    "VerticalTask",
    "build_vertical_tasks",
    "canonical_hash",
    "canonical_json",
    "load_vertical_offline_config",
]

"""Prepare and audit resumable AAAI rebuttal online experiment runs.

The manifest bridge is deliberately stricter than a filesystem copy.  Every
reused result must match the frozen matrix identity, configuration, dataset,
repository commit, decision count, and measurement-integrity gates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

from lhmsb.longhorizon.online_cli import online_result_id
from lhmsb.qualification.config import load_qualification_config
from lhmsb.qualification.schema import SystemBackend, SystemsQualificationConfig

REBUTTAL_MANIFEST_SCHEMA_VERSION = "memhorizon-aaai-rebuttal-manifest-v1"
INTEGRITY_FIELDS = (
    "online_long_horizon",
    "causal_chain_verified",
    "workflow_complete",
    "project_chain_verified",
    "substantive_work_verified",
    "memory_selection_verified",
    "memory_count_verified",
    "efficiency_measurement_verified",
)


@dataclass(frozen=True)
class ResultCandidate:
    result_id: str
    source: Path
    sha256: str
    task_index: int
    episode_id: str

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["source"] = str(self.source)
        return payload


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _audit_repository_snapshot() -> dict[str, object]:
    root = Path(__file__).resolve().parents[3]
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return {"code_commit": "unknown", "code_dirty": None}
    return {"code_commit": commit, "code_dirty": dirty}


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _load_tasks(run_dir: Path) -> tuple[dict[str, object], ...]:
    path = run_dir / "tasks.jsonl"
    tasks = tuple(
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    )
    if not tasks or not all(isinstance(task, dict) for task in tasks):
        raise ValueError(f"invalid or empty task ledger: {path}")
    normalized = tuple({str(key): value for key, value in task.items()} for task in tasks)
    for expected_index, task in enumerate(normalized):
        if task.get("task_index") != expected_index:
            raise ValueError(
                f"task ledger is not contiguous at {expected_index}: {task.get('task_index')!r}"
            )
        if not isinstance(task.get("episode_id"), str):
            raise ValueError(f"task {expected_index} lacks episode_id")
    return normalized


def install_frozen_plan(source_dir: Path, run_dir: Path) -> None:
    """Install once, then prove later resumes use the same frozen identities."""
    source_matrix = _read_json(source_dir / "MATRIX.json")
    source_tasks = (source_dir / "tasks.jsonl").read_text(encoding="utf-8")
    run_dir.mkdir(parents=True, exist_ok=True)
    destination_matrix = run_dir / "MATRIX.json"
    destination_tasks = run_dir / "tasks.jsonl"
    if destination_matrix.exists():
        current = _read_json(destination_matrix)
        immutable_fields = (
            "schema_version",
            "config_hash",
            "dataset_release",
            "dataset_manifest_sha256",
            "policy_profile",
            "conditions",
            "expected_result_count",
            "expected_policy_calls",
            "expected_result_ids",
            "repository",
        )
        changed = [
            field for field in immutable_fields if current.get(field) != source_matrix.get(field)
        ]
        if changed:
            raise ValueError(f"existing run uses a different frozen plan: {','.join(changed)}")
    else:
        _atomic_write_text(
            destination_matrix,
            json.dumps(source_matrix, sort_keys=True, ensure_ascii=False, indent=2) + "\n",
        )
    if destination_tasks.exists():
        if destination_tasks.read_text(encoding="utf-8") != source_tasks:
            raise ValueError("existing run task ledger differs from frozen source")
    else:
        _atomic_write_text(destination_tasks, source_tasks)


def rebind_frozen_matrix_paths(
    run_dir: Path,
    *,
    dataset: Path,
    config_path: Path,
) -> dict[str, object]:
    """Move a frozen plan to another host without changing its identities."""
    manifest_path = dataset / "MANIFEST.json"
    if not manifest_path.is_file():
        raise ValueError(f"dataset manifest does not exist: {manifest_path}")
    matrix = _read_json(run_dir / "MATRIX.json")
    observed_dataset_hash = _sha_file(manifest_path)
    if observed_dataset_hash != matrix.get("dataset_manifest_sha256"):
        raise ValueError("dataset manifest hash does not match frozen matrix")
    loaded = load_qualification_config(config_path)
    if not isinstance(loaded, SystemsQualificationConfig):
        raise ValueError("online matrix requires a schema-v2 systems config")
    if loaded.config_hash != matrix.get("config_hash"):
        raise ValueError("config hash does not match frozen matrix")
    if loaded.dataset_release != matrix.get("dataset_release"):
        raise ValueError("config dataset release does not match frozen matrix")
    rebound = dict(matrix)
    rebound["dataset"] = str(dataset.resolve())
    rebound["config"] = str(config_path.resolve())
    _atomic_write_text(
        run_dir / "MATRIX.json",
        json.dumps(rebound, sort_keys=True, ensure_ascii=False, indent=2) + "\n",
    )
    return rebound


def _identity_from_payload(payload: Mapping[str, object]) -> tuple[str, str]:
    result = payload.get("result")
    if not isinstance(result, Mapping):
        raise ValueError("payload lacks result object")
    episode_id = result.get("episode_id")
    condition = payload.get("condition", result.get("condition"))
    backend = payload.get("backend")
    policy_profile = payload.get("policy_profile")
    if not all(isinstance(value, str) for value in (episode_id, condition, policy_profile)):
        raise ValueError("payload lacks string episode, condition, or policy identity")
    if condition not in {"workspace_only", "memory"}:
        raise ValueError(f"unsupported online condition: {condition!r}")
    if condition == "memory" and backend not in {"flat_retrieval", "mem0", "amem", "memos"}:
        raise ValueError(f"unsupported memory backend: {backend!r}")
    if condition == "workspace_only" and backend is not None:
        raise ValueError("workspace-only payload must have null backend")
    computed = online_result_id(
        episode_id=cast(str, episode_id),
        condition=condition,
        backend=(None if backend is None else cast(SystemBackend, backend)),
        policy_profile=cast(str, policy_profile),
    )
    return computed, cast(str, episode_id)


def _candidate_errors(
    payload: Mapping[str, object],
    *,
    matrix: Mapping[str, object],
    expected_ids: set[str],
    episode_to_task: Mapping[str, int],
) -> tuple[list[str], str | None, str | None, int | None]:
    errors: list[str] = []
    try:
        computed_id, episode_id = _identity_from_payload(payload)
    except ValueError as exc:
        return [str(exc)], None, None, None
    stored_id = payload.get("result_id")
    if stored_id != computed_id:
        errors.append(f"stored result_id does not match computed identity: {stored_id!r}")
    if computed_id not in expected_ids:
        errors.append("result identity is not present in frozen matrix")
    execution_overrides = matrix.get("execution_overrides")
    result_config_hashes: Mapping[str, object] = {}
    if isinstance(execution_overrides, Mapping):
        raw_hashes = execution_overrides.get("result_config_hashes", {})
        if isinstance(raw_hashes, Mapping):
            result_config_hashes = cast(Mapping[str, object], raw_hashes)
    expected_config_hash = result_config_hashes.get(computed_id, matrix.get("config_hash"))
    if payload.get("config_hash") != expected_config_hash:
        errors.append(
            "config_hash mismatch: "
            f"{payload.get('config_hash')!r} != {expected_config_hash!r}"
        )
    if payload.get("dataset_release") != matrix.get("dataset_release"):
        errors.append("dataset_release mismatch")
    if payload.get("policy_profile") != matrix.get("policy_profile"):
        errors.append("policy_profile mismatch")
    repository = payload.get("repository")
    expected_repository = matrix.get("repository")
    if not isinstance(repository, Mapping) or not isinstance(expected_repository, Mapping):
        errors.append("missing repository provenance")
    else:
        result_repository_commits: Mapping[str, object] = {}
        if isinstance(execution_overrides, Mapping):
            raw_commits = execution_overrides.get("result_repository_commits", {})
            if isinstance(raw_commits, Mapping):
                result_repository_commits = cast(Mapping[str, object], raw_commits)
        expected_commit = result_repository_commits.get(
            computed_id, expected_repository.get("code_commit")
        )
        if repository.get("code_commit") != expected_commit:
            errors.append(
                "repository commit mismatch: "
                f"{repository.get('code_commit')!r} != {expected_commit!r}"
            )
        if repository.get("code_dirty") is not False:
            errors.append("result was produced from a dirty or unknown checkout")
    result = payload.get("result")
    if not isinstance(result, Mapping):
        errors.append("missing result object")
    else:
        policy_calls_total = matrix.get("expected_policy_calls")
        result_count = matrix.get("expected_result_count")
        if (
            isinstance(policy_calls_total, bool)
            or not isinstance(policy_calls_total, int)
            or isinstance(result_count, bool)
            or not isinstance(result_count, int)
        ):
            raise ValueError("matrix expected call/count fields must be integers")
        expected_calls = policy_calls_total // max(1, result_count)
        if expected_calls <= 0:
            expected_calls = 256
        if result.get("policy_calls") != expected_calls:
            errors.append(
                f"policy_calls mismatch: {result.get('policy_calls')!r} != {expected_calls}"
            )
        failed = [field for field in INTEGRITY_FIELDS if result.get(field) is not True]
        if failed:
            errors.append(f"failed measurement-integrity fields: {','.join(failed)}")
    task_index = episode_to_task.get(episode_id)
    if task_index is None:
        errors.append("episode is absent from task ledger")
    return errors, computed_id, episode_id, task_index


def _result_paths(root: Path) -> Iterable[Path]:
    if not root.exists():
        return ()
    return (
        path
        for path in sorted(root.rglob("*.json"))
        if path.name not in {"MANIFEST.json", "MATRIX.json", "report.json", "validation.json"}
    )


def audit_result_roots(
    run_dir: Path,
    legacy_roots: Iterable[Path] = (),
) -> dict[str, object]:
    """Classify all saved result candidates against the frozen matrix."""
    matrix = _read_json(run_dir / "MATRIX.json")
    tasks = _load_tasks(run_dir)
    expected_ids = {str(item) for item in matrix.get("expected_result_ids", [])}
    if len(expected_ids) != matrix.get("expected_result_count"):
        raise ValueError("matrix expected-result count is inconsistent")
    execution_overrides = matrix.get("execution_overrides")
    if execution_overrides is not None:
        if not isinstance(execution_overrides, Mapping):
            raise ValueError("matrix execution_overrides must be an object")
        raw_hashes = execution_overrides.get("result_config_hashes")
        if not isinstance(raw_hashes, Mapping) or not raw_hashes:
            raise ValueError(
                "matrix execution_overrides requires non-empty result_config_hashes"
            )
        invalid_ids = sorted(str(key) for key in raw_hashes if str(key) not in expected_ids)
        if invalid_ids:
            raise ValueError(
                "matrix execution_overrides contains unknown result IDs: "
                + ",".join(invalid_ids)
            )
        if any(not isinstance(value, str) or not value for value in raw_hashes.values()):
            raise ValueError("matrix execution override config hashes must be strings")
        raw_commits = execution_overrides.get("result_repository_commits")
        if not isinstance(raw_commits, Mapping) or set(raw_commits) != set(raw_hashes):
            raise ValueError(
                "matrix execution_overrides requires repository commits for exactly "
                "the overridden result IDs"
            )
        if any(not isinstance(value, str) or not value for value in raw_commits.values()):
            raise ValueError("matrix execution override repository commits must be strings")
    episode_to_task = {
        cast(str, task["episode_id"]): cast(int, task["task_index"]) for task in tasks
    }
    roots = (run_dir / "results", *tuple(legacy_roots))
    valid_by_id: dict[str, ResultCandidate] = {}
    rejected: list[dict[str, object]] = []
    identical_duplicates: list[dict[str, object]] = []
    divergent_duplicates: list[dict[str, object]] = []
    scanned_json_count = result_payload_count = 0
    for root in roots:
        for path in _result_paths(root):
            scanned_json_count += 1
            try:
                payload = _read_json(path)
            except (OSError, json.JSONDecodeError, ValueError) as exc:
                rejected.append({"source": str(path), "errors": [f"invalid JSON: {exc}"]})
                continue
            if not isinstance(payload.get("result"), Mapping):
                continue
            result_payload_count += 1
            errors, result_id, episode_id, task_index = _candidate_errors(
                payload,
                matrix=matrix,
                expected_ids=expected_ids,
                episode_to_task=episode_to_task,
            )
            if errors or result_id is None or episode_id is None or task_index is None:
                rejected.append(
                    {
                        "source": str(path),
                        "result_id": result_id,
                        "errors": errors or ["candidate identity is incomplete"],
                    }
                )
                continue
            candidate = ResultCandidate(
                result_id=result_id,
                source=path.resolve(),
                sha256=_sha_file(path),
                task_index=task_index,
                episode_id=episode_id,
            )
            previous = valid_by_id.get(result_id)
            if previous is None:
                valid_by_id[result_id] = candidate
            elif previous.sha256 == candidate.sha256:
                identical_duplicates.append(
                    {"result_id": result_id, "kept": str(previous.source), "ignored": str(path)}
                )
            else:
                divergent_duplicates.append(
                    {
                        "result_id": result_id,
                        "first": previous.to_dict(),
                        "second": candidate.to_dict(),
                    }
                )
    valid_ids = set(valid_by_id)
    missing_ids = sorted(expected_ids - valid_ids)
    missing_tasks = sorted(
        {
            episode_to_task[episode_id]
            for episode_id in episode_to_task
            if any(result_id.startswith(f"{episode_id}__") for result_id in missing_ids)
        }
    )
    return {
        "schema_version": REBUTTAL_MANIFEST_SCHEMA_VERSION,
        "ok": not rejected and not divergent_duplicates,
        "complete": not missing_ids and not rejected and not divergent_duplicates,
        "expected_result_count": len(expected_ids),
        "valid_result_count": len(valid_by_id),
        "missing_result_count": len(missing_ids),
        "missing_result_ids": missing_ids,
        "missing_task_count": len(missing_tasks),
        "missing_task_indices": missing_tasks,
        "scanned_json_count": scanned_json_count,
        "result_payload_count": result_payload_count,
        "rejected": rejected,
        "identical_duplicates": identical_duplicates,
        "divergent_duplicates": divergent_duplicates,
        "valid_candidates": [
            candidate.to_dict()
            for candidate in sorted(valid_by_id.values(), key=lambda item: item.result_id)
        ],
    }


def _install_candidate(candidate: ResultCandidate, run_dir: Path) -> str:
    destination = (
        run_dir / "results" / f"task-{candidate.task_index:04d}" / f"{candidate.result_id}.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if _sha_file(destination) != candidate.sha256:
            raise ValueError(f"refusing to overwrite divergent result: {destination}")
        return "already_present"
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        try:
            os.link(candidate.source, temporary)
            method = "hardlink"
        except OSError:
            shutil.copy2(candidate.source, temporary)
            method = "copy"
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return method


def prepare_rebuttal_run(
    run_dir: Path,
    legacy_roots: Iterable[Path] = (),
) -> dict[str, object]:
    """Import compatible evidence and write an executable missing-work ledger."""
    initial = audit_result_roots(run_dir, legacy_roots)
    if not initial["ok"]:
        raise ValueError(
            "candidate result audit failed; inspect compatibility_report.json before running"
        )
    import_counts = {"hardlink": 0, "copy": 0, "already_present": 0}
    raw_candidates = initial["valid_candidates"]
    assert isinstance(raw_candidates, list)
    for raw in raw_candidates:
        assert isinstance(raw, Mapping)
        candidate = ResultCandidate(
            result_id=str(raw["result_id"]),
            source=Path(str(raw["source"])),
            sha256=str(raw["sha256"]),
            task_index=int(raw["task_index"]),
            episode_id=str(raw["episode_id"]),
        )
        import_counts[_install_candidate(candidate, run_dir)] += 1
    final = audit_result_roots(run_dir)
    final["import_counts"] = import_counts
    final["source_compatibility"] = {
        key: value
        for key, value in initial.items()
        if key not in {"valid_candidates", "missing_result_ids", "missing_task_indices"}
    }
    tasks = _load_tasks(run_dir)
    missing_task_set = set(cast(list[int], final["missing_task_indices"]))
    missing_task_records = [task for task in tasks if task["task_index"] in missing_task_set]
    _atomic_write_text(
        run_dir / "EXPECTED_RESULTS.jsonl",
        "".join(
            json.dumps({"result_id": result_id}, sort_keys=True) + "\n"
            for result_id in _read_json(run_dir / "MATRIX.json")["expected_result_ids"]
        ),
    )
    _atomic_write_text(
        run_dir / "missing_tasks.jsonl",
        "".join(json.dumps(task, sort_keys=True) + "\n" for task in missing_task_records),
    )
    _atomic_write_text(
        run_dir / "compatibility_report.json",
        json.dumps(final, sort_keys=True, ensure_ascii=False, indent=2) + "\n",
    )
    manifest = {
        "schema_version": REBUTTAL_MANIFEST_SCHEMA_VERSION,
        "matrix": str((run_dir / "MATRIX.json").resolve()),
        "expected_result_count": final["expected_result_count"],
        "reused_result_count": final["valid_result_count"],
        "missing_result_count": final["missing_result_count"],
        "missing_task_indices": final["missing_task_indices"],
        "ready_to_execute": final["ok"],
        "complete": final["complete"],
        "import_counts": import_counts,
        "audit_repository": _audit_repository_snapshot(),
    }
    _atomic_write_text(
        run_dir / "EXPERIMENT_MANIFEST.json",
        json.dumps(manifest, sort_keys=True, ensure_ascii=False, indent=2) + "\n",
    )
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--source-plan", type=Path, default=None)
    parser.add_argument("--legacy-results", type=Path, action="append", default=[])
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--audit-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.source_plan is not None:
        install_frozen_plan(args.source_plan, args.run_dir)
    if (args.dataset is None) != (args.config is None):
        raise SystemExit("--dataset and --config must be supplied together")
    if args.dataset is not None and args.config is not None:
        rebind_frozen_matrix_paths(
            args.run_dir,
            dataset=args.dataset,
            config_path=args.config,
        )
    if args.audit_only:
        report = audit_result_roots(args.run_dir, args.legacy_results)
        print(json.dumps(report, sort_keys=True, ensure_ascii=False))
        return 0 if report["ok"] else 1
    try:
        manifest = prepare_rebuttal_run(args.run_dir, args.legacy_results)
    except ValueError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, sort_keys=True))
        return 1
    print(json.dumps(manifest, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "INTEGRITY_FIELDS",
    "REBUTTAL_MANIFEST_SCHEMA_VERSION",
    "ResultCandidate",
    "audit_result_roots",
    "install_frozen_plan",
    "main",
    "prepare_rebuttal_run",
    "rebind_frozen_matrix_paths",
]

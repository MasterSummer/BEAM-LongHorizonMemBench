# Vertical Offline Pilot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a frozen-data-only offline pilot driver that runs the six Software vertical tasks locally or as independent Slurm array jobs and emits reproducible per-SCEU results.

**Architecture:** A strict loader reconstructs `SoftwareVerticalSpec` from a verified frozen directory. A configuration module creates an immutable run identity and deterministic task list. A shared atomic worker powers both sequential local execution and one-task cluster execution, while a deterministic aggregator produces JSONL records and pilot summaries.

**Tech Stack:** Python 3.11+, frozen dataclasses, PyYAML, argparse, canonical JSON/SHA-256, pytest, existing LHMSB vertical runner.

---

## File Map

| File | Responsibility |
|---|---|
| `src/lhmsb/datasets/stateful_loader.py` | Verify and reconstruct frozen Software vertical specs |
| `src/lhmsb/datasets/__init__.py` | Export the frozen loader |
| `src/lhmsb/experiments/vertical_config.py` | Parse/normalize YAML, define task/run identity types |
| `src/lhmsb/experiments/vertical_runner.py` | Plan, atomically execute, resume, and aggregate tasks |
| `src/lhmsb/experiments/vertical.py` | `plan`, `run`, `run-task`, and `aggregate` CLI |
| `src/lhmsb/experiments/__init__.py` | Public experiment API exports |
| `configs/vertical_offline_pilot.yaml` | Pinned six-task offline matrix |
| `tests/datasets/test_stateful_loader.py` | Frozen loader integrity tests |
| `tests/experiments/conftest.py` | Four-session frozen fixture and config fixture |
| `tests/experiments/test_vertical_config.py` | Config validation and deterministic task identity tests |
| `tests/experiments/test_vertical_runner.py` | Planning, worker, resume, and aggregation tests |
| `tests/experiments/test_vertical_cli.py` | End-to-end CLI tests |

### Task 1: Frozen Software spec loader

**Files:**

- Create: `src/lhmsb/datasets/stateful_loader.py`
- Modify: `src/lhmsb/datasets/__init__.py`
- Create: `tests/datasets/test_stateful_loader.py`

- [ ] **Step 1: Write loader round-trip and tamper tests**

```python
def _frozen(tmp_path: Path) -> Path:
    stage = tmp_path / "stage"
    frozen = tmp_path / "software_v1"
    generate_stateful_to_staging(
        stage, family="software", seeds=(42,), n_episodes=1, n_sessions=4
    )
    freeze_stateful(stage, frozen)
    return frozen


def test_load_frozen_spec_without_regeneration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frozen_vertical = _frozen(tmp_path)
    monkeypatch.setattr(
        SoftwareVerticalFamily,
        "generate",
        classmethod(lambda cls, *args, **kwargs: (_ for _ in ()).throw(AssertionError())),
    )
    specs = load_software_vertical_specs(frozen_vertical)
    assert len(specs) == 1
    assert specs[0].plan.episode_id == "software-42"
    assert plan_hash(specs[0].plan) == _record(frozen_vertical)["plan_hash"]


def test_load_rejects_tampered_frozen_file(tmp_path: Path) -> None:
    frozen_vertical = _frozen(tmp_path)
    episodes = frozen_vertical / "episodes.jsonl"
    episodes.write_text(episodes.read_text() + "\n", encoding="utf-8")
    with pytest.raises(StatefulDatasetError, match="checksum"):
        load_software_vertical_specs(frozen_vertical)


def test_load_rejects_record_hash_drift(tmp_path: Path) -> None:
    frozen = _frozen(tmp_path)
    record = json.loads((frozen / "episodes.jsonl").read_text().splitlines()[0])
    record["plan_hash"] = "0" * 64
    (frozen / "episodes.jsonl").write_text(
        json.dumps(record, sort_keys=True) + "\n", encoding="utf-8"
    )
    with pytest.raises(StatefulDatasetError, match="plan hash"):
        load_software_vertical_specs(frozen, verify=False)
```

- [ ] **Step 2: Run the loader tests and confirm RED**

Run:

```bash
PYTHONPATH="$PWD/src" pytest tests/datasets/test_stateful_loader.py -q
```

Expected: collection fails because `lhmsb.datasets.stateful_loader` does not exist.

- [ ] **Step 3: Implement strict reconstruction**

Implement these public and internal interfaces:

```python
def load_software_vertical_specs(
    frozen: Path,
    *,
    verify: bool = True,
) -> tuple[SoftwareVerticalSpec, ...]:
    if verify:
        report = verify_stateful(frozen)
        if not report.ok:
            raise StatefulDatasetError(_verification_message(report))
    manifest = _read_manifest(frozen)
    if manifest.schema_version != STATEFUL_SCHEMA_VERSION:
        raise StatefulDatasetError(
            f"unsupported stateful schema version: {manifest.schema_version}"
        )
    if manifest.family != "software":
        raise StatefulDatasetError(f"unsupported stateful family: {manifest.family}")
    records = _read_jsonl(frozen / "episodes.jsonl")
    specs = tuple(_spec_from_record(record) for record in records)
    _validate_records(specs, records, manifest)
    return specs
```

`_spec_from_record()` must use:

```python
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
```

Validation must recompute `plan_hash`, `surfaces_hash`, and workspace SHA-256
from `[asdict(item) for item in plan.workspaces]`, then compare episode ID,
semantic/trajectory seeds, session count, and all hashes against both
`episodes.jsonl` and `MANIFEST.json`.

- [ ] **Step 4: Export loader and verify GREEN**

Add `load_software_vertical_specs` to `lhmsb.datasets.__all__`, then run:

```bash
PYTHONPATH="$PWD/src" pytest tests/datasets/test_stateful_loader.py \
  tests/datasets/test_stateful_pipeline.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit Task 1**

```bash
git add src/lhmsb/datasets tests/datasets/test_stateful_loader.py
git commit -m "feat: load frozen software vertical specs"
```

### Task 2: Configuration, code snapshot, and deterministic task identities

**Files:**

- Create: `src/lhmsb/experiments/__init__.py`
- Create: `src/lhmsb/experiments/vertical_config.py`
- Create: `configs/vertical_offline_pilot.yaml`
- Create: `tests/experiments/conftest.py`
- Create: `tests/experiments/test_vertical_config.py`

- [ ] **Step 1: Create shared frozen/config fixtures**

```python
@pytest.fixture
def frozen_vertical(tmp_path: Path) -> Path:
    stage = tmp_path / "stage"
    frozen = tmp_path / "software_v1"
    generate_stateful_to_staging(
        stage, family="software", seeds=(42,), n_episodes=1, n_sessions=4
    )
    freeze_stateful(stage, frozen)
    return frozen


@pytest.fixture
def offline_config(tmp_path: Path) -> Path:
    path = tmp_path / "vertical.yaml"
    path.write_text(
        "schema_version: 1\n"
        "experiment_id: software-vertical-offline-pilot\n"
        "conditions:\n"
        "  workspace_only: [null]\n"
        "  oracle_current_state: [null]\n"
        "  fake_native: [null, P2, C1, U1]\n",
        encoding="utf-8",
    )
    return path
```

- [ ] **Step 2: Write config normalization and task-order tests**

```python
def test_default_matrix_has_six_ordered_tasks(
    frozen_vertical: Path, offline_config: Path
) -> None:
    config = load_vertical_offline_config(offline_config)
    specs = load_software_vertical_specs(frozen_vertical)
    tasks = build_vertical_tasks(specs, config, run_identity="r" * 64)
    assert [(task.condition, task.intervention_state_id) for task in tasks] == [
        ("workspace_only", None),
        ("oracle_current_state", None),
        ("fake_native", None),
        ("fake_native", "P2"),
        ("fake_native", "C1"),
        ("fake_native", "U1"),
    ]
    assert [task.task_index for task in tasks] == list(range(6))
    assert len({task.task_id for task in tasks}) == 6


def test_config_rejects_intervention_outside_fake_native(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(
        "schema_version: 1\nexperiment_id: x\n"
        "conditions:\n  workspace_only: [P2]\n",
        encoding="utf-8",
    )
    with pytest.raises(VerticalExperimentError, match="fake_native"):
        load_vertical_offline_config(path)
```

- [ ] **Step 3: Run config tests and confirm RED**

Run:

```bash
PYTHONPATH="$PWD/src" pytest tests/experiments/test_vertical_config.py -q
```

Expected: import fails because `lhmsb.experiments.vertical_config` does not exist.

- [ ] **Step 4: Implement frozen config/task dataclasses**

Define:

```python
@dataclass(frozen=True)
class VerticalOfflineConfig:
    schema_version: int
    experiment_id: str
    conditions: tuple[tuple[VerticalCondition, tuple[str | None, ...]], ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "experiment_id": self.experiment_id,
            "conditions": [
                {"condition": condition, "interventions": list(interventions)}
                for condition, interventions in self.conditions
            ],
        }

    @property
    def config_hash(self) -> str:
        return canonical_hash(self.to_dict())


@dataclass(frozen=True)
class GitSnapshot:
    commit: str
    dirty: bool
    ref: str


@dataclass(frozen=True)
class VerticalTask:
    task_index: int
    task_id: str
    episode_id: str
    condition: VerticalCondition
    intervention_state_id: str | None
    run_identity: str
    task_payload_hash: str
```

`VerticalTask.from_dict()` must validate the condition and reconstruct all
fields. `build_vertical_tasks()` creates payload hashes before constructing
each task and uses a filesystem-safe ID:

```python
slug = re.sub(r"[^a-zA-Z0-9_.-]+", "-", value).strip("-")
task_id = (
    f"{index:05d}-{slug(spec.plan.episode_id)}-"
    f"{condition.replace('_', '-')}-{slug(intervention or 'baseline')}"
)
```

- [ ] **Step 5: Implement YAML validation and canonical hashing**

`load_vertical_offline_config()` must reject:

- schema versions other than `1`;
- empty experiment IDs;
- unknown or duplicate conditions;
- empty intervention lists;
- duplicate interventions;
- interventions for workspace/oracle;
- non-string/non-null intervention values.

Use canonical JSON:

```python
def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def canonical_hash(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()
```

- [ ] **Step 6: Add the pinned default config and verify GREEN**

Create:

```yaml
schema_version: 1
experiment_id: software-vertical-offline-pilot
conditions:
  workspace_only: [null]
  oracle_current_state: [null]
  fake_native: [null, P2, C1, U1]
```

Run:

```bash
PYTHONPATH="$PWD/src" pytest tests/experiments/test_vertical_config.py -q
```

Expected: all tests pass.

- [ ] **Step 7: Commit Task 2**

```bash
git add configs/vertical_offline_pilot.yaml src/lhmsb/experiments \
  tests/experiments/conftest.py tests/experiments/test_vertical_config.py
git commit -m "feat: define vertical pilot configuration and tasks"
```

### Task 3: Immutable planning and atomic task execution

**Files:**

- Create: `src/lhmsb/experiments/vertical_runner.py`
- Modify: `src/lhmsb/experiments/__init__.py`
- Modify: `tests/experiments/conftest.py`
- Create: `tests/experiments/test_vertical_runner.py`

- [ ] **Step 1: Write planning identity and idempotence tests**

```python
def test_plan_is_idempotent_and_binds_dataset_code_and_config(
    frozen_vertical: Path, offline_config: Path, tmp_path: Path
) -> None:
    run_dir = tmp_path / "run"
    first = plan_vertical_run(
        frozen_vertical, offline_config, run_dir, allow_dirty=True
    )
    manifest_bytes = (run_dir / "run_manifest.json").read_bytes()
    tasks_bytes = (run_dir / "tasks.jsonl").read_bytes()
    second = plan_vertical_run(
        frozen_vertical, offline_config, run_dir, allow_dirty=True
    )
    assert first.run_identity == second.run_identity
    assert (run_dir / "run_manifest.json").read_bytes() == manifest_bytes
    assert (run_dir / "tasks.jsonl").read_bytes() == tasks_bytes
    assert first.dataset_manifest_sha256
    assert first.config_hash
    assert first.code_commit
```

- [ ] **Step 2: Run planner tests and confirm RED**

Run:

```bash
PYTHONPATH="$PWD/src" pytest \
  tests/experiments/test_vertical_runner.py::test_plan_is_idempotent_and_binds_dataset_code_and_config \
  -q
```

Expected: import fails because `vertical_runner.py` does not exist.

- [ ] **Step 3: Implement run manifest and planning**

Define `VerticalRunManifest` with `to_dict()`/`from_dict()` and these fields:

```python
schema_version: int
run_identity: str
experiment_id: str
code_commit: str
code_dirty: bool
code_ref: str
dataset_path: str
dataset_manifest_sha256: str
dataset_schema_version: int
dataset_generator_version: str
dataset_family: str
dataset_episodes: tuple[dict[str, object], ...]
config_path: str
config_hash: str
config: dict[str, object]
task_count: int
created_at_utc: str
environment: dict[str, str]
```

`plan_vertical_run()` sequence:

1. load config;
2. verify/load frozen specs;
3. read dataset manifest;
4. read `GitSnapshot`;
5. reject dirty checkout unless allowed;
6. compute run identity;
7. build tasks;
8. handle new/idempotent/force output directory;
9. atomically write `run_config.yaml`, `run_manifest.json`, and `tasks.jsonl`.

The run identity payload is exactly:

```python
{
    "schema_version": 1,
    "code_commit": snapshot.commit,
    "code_dirty": snapshot.dirty,
    "dataset_manifest_sha256": manifest_sha,
    "dataset_schema_version": dataset_manifest.schema_version,
    "dataset_generator_version": dataset_manifest.generator_version,
    "config_hash": config.config_hash,
}
```

- [ ] **Step 4: Add planned-run fixtures**

Now that planning exists, extend `tests/experiments/conftest.py` with:

```python
@pytest.fixture
def planned_run(
    frozen_vertical: Path, offline_config: Path, tmp_path: Path
) -> Path:
    run_dir = tmp_path / "run"
    plan_vertical_run(frozen_vertical, offline_config, run_dir, allow_dirty=True)
    return run_dir


@pytest.fixture
def two_planned_runs(
    frozen_vertical: Path, offline_config: Path, tmp_path: Path
) -> tuple[Path, Path]:
    first = tmp_path / "first"
    second = tmp_path / "second"
    plan_vertical_run(frozen_vertical, offline_config, first, allow_dirty=True)
    plan_vertical_run(frozen_vertical, offline_config, second, allow_dirty=True)
    return first, second
```

Use these concrete cases for changed identity, unrelated directories, and dirty
checkout rejection:

```python
def test_plan_rejects_changed_config(
    frozen_vertical: Path, offline_config: Path, tmp_path: Path
) -> None:
    run_dir = tmp_path / "run"
    plan_vertical_run(frozen_vertical, offline_config, run_dir, allow_dirty=True)
    offline_config.write_text(
        offline_config.read_text().replace("U1]", "U1, G0]"),
        encoding="utf-8",
    )
    with pytest.raises(VerticalExperimentError, match="identity"):
        plan_vertical_run(frozen_vertical, offline_config, run_dir, allow_dirty=True)


def test_plan_rejects_unrelated_nonempty_directory(
    frozen_vertical: Path, offline_config: Path, tmp_path: Path
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "user-data.txt").write_text("preserve me", encoding="utf-8")
    with pytest.raises(VerticalExperimentError, match="unrelated"):
        plan_vertical_run(frozen_vertical, offline_config, run_dir, allow_dirty=True)


def test_plan_requires_allow_dirty(
    frozen_vertical: Path,
    offline_config: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        vertical_runner,
        "current_git_snapshot",
        lambda: GitSnapshot(commit="a" * 40, dirty=True, ref="feature"),
    )
    with pytest.raises(VerticalExperimentError, match="dirty"):
        plan_vertical_run(frozen_vertical, offline_config, tmp_path / "run")
```

- [ ] **Step 5: Write independent worker and resume tests**

```python
def test_run_task_is_independent_and_skips_success(
    planned_run: Path
) -> None:
    result_path = run_vertical_task(planned_run, 3)
    first_bytes = result_path.read_bytes()
    first_mtime = result_path.stat().st_mtime_ns
    assert json.loads(first_bytes)["task"]["intervention_state_id"] == "P2"
    assert run_vertical_task(planned_run, 3) == result_path
    assert result_path.read_bytes() == first_bytes
    assert result_path.stat().st_mtime_ns == first_mtime
```

Add explicit invalid-index and stale-result cases:

```python
def test_run_task_rejects_invalid_index(planned_run: Path) -> None:
    with pytest.raises(VerticalExperimentError, match="task index"):
        run_vertical_task(planned_run, 99)


def test_run_task_requires_force_for_stale_result(planned_run: Path) -> None:
    result_path = run_vertical_task(planned_run, 0)
    payload = json.loads(result_path.read_text())
    payload["run_identity"] = "stale"
    result_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(VerticalExperimentError, match="stale"):
        run_vertical_task(planned_run, 0)
    repaired = run_vertical_task(planned_run, 0, force=True)
    assert json.loads(repaired.read_text())["run_identity"] != "stale"


def test_failed_task_can_retry(
    planned_run: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = vertical_runner.run_vertical_episode
    monkeypatch.setattr(
        vertical_runner,
        "run_vertical_episode",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("injected")),
    )
    with pytest.raises(RuntimeError, match="injected"):
        run_vertical_task(planned_run, 0)
    task = read_vertical_tasks(planned_run)[0]
    failure = planned_run / "tasks" / task.task_id / "failure.json"
    assert failure.is_file()
    monkeypatch.setattr(vertical_runner, "run_vertical_episode", original)
    result = run_vertical_task(planned_run, 0)
    assert result.is_file()
    assert not failure.exists()
```

- [ ] **Step 6: Implement atomic task execution**

Implement:

```python
def run_vertical_task(
    run_dir: Path,
    task_index: int,
    *,
    force: bool = False,
) -> Path:
    manifest, tasks = _load_run_contract(run_dir)
    _verify_current_code(manifest)
    task = _task_at(tasks, task_index)
    result_path = run_dir / "tasks" / task.task_id / "result.json"
    if result_path.exists() and not force:
        _validate_result_file(result_path, task, manifest)
        return result_path
    specs = load_software_vertical_specs(Path(manifest.dataset_path))
    spec = _select_spec(specs, task.episode_id)
    try:
        run_result = run_vertical_episode(
            spec,
            task.condition,
            intervention_state_id=task.intervention_state_id,
        )
        _atomic_json(result_path, _result_envelope(manifest, task, run_result))
        (result_path.parent / "failure.json").unlink(missing_ok=True)
        return result_path
    except Exception as exc:
        _atomic_json(
            result_path.parent / "failure.json",
            _failure_envelope(manifest, task, exc),
        )
        raise
```

`_result_envelope()` must explicitly include `behavior_score`,
`selected_actions`, nested SCEU results, hashes, and native trace. It must not
depend on dataclass property serialization.

- [ ] **Step 7: Verify Task 3 GREEN**

Run:

```bash
PYTHONPATH="$PWD/src" pytest tests/experiments/test_vertical_runner.py -q
```

Expected: planning and worker tests pass.

- [ ] **Step 8: Commit Task 3**

```bash
git add src/lhmsb/experiments tests/experiments/test_vertical_runner.py
git commit -m "feat: plan and execute atomic vertical pilot tasks"
```

### Task 4: Deterministic aggregation and leave-one-out summary

**Files:**

- Modify: `src/lhmsb/experiments/vertical_runner.py`
- Modify: `src/lhmsb/experiments/__init__.py`
- Modify: `tests/experiments/test_vertical_runner.py`

- [ ] **Step 1: Write complete, partial, and order-independence tests**

```python
def test_aggregate_reconstructs_chain_and_leave_one_out(planned_run: Path) -> None:
    for index in (5, 2, 0, 4, 1, 3):
        run_vertical_task(planned_run, index)
    aggregate = aggregate_vertical_run(planned_run)
    assert aggregate.complete
    assert aggregate.completed_tasks == 6
    summary = json.loads((planned_run / "summary.json").read_text())
    assert summary["chain_coverage"]["complete_rows"] > 0
    assert {item["intervention_state_id"] for item in summary["leave_one_out"]} == {
        "P2", "C1", "U1"
    }
    p2 = next(item for item in summary["leave_one_out"] if item["intervention_state_id"] == "P2")
    assert p2["action_changes"] > 0 or p2["score_delta"] != 0
```

Add these partial-run and order-independence assertions:

```python
def test_partial_aggregate_reports_missing_tasks(planned_run: Path) -> None:
    run_vertical_task(planned_run, 0)
    aggregate = aggregate_vertical_run(planned_run)
    assert not aggregate.complete
    assert aggregate.completed_tasks == 1
    assert aggregate.missing_tasks == 5


def test_aggregate_is_independent_of_execution_order(
    two_planned_runs: tuple[Path, Path]
) -> None:
    sequential, shuffled = two_planned_runs
    for index in range(6):
        run_vertical_task(sequential, index)
    for index in (5, 2, 0, 4, 1, 3):
        run_vertical_task(shuffled, index)
    aggregate_vertical_run(sequential)
    aggregate_vertical_run(shuffled)
    for name in ("task_results.jsonl", "sceu_results.jsonl", "summary.json"):
        assert (sequential / name).read_bytes() == (shuffled / name).read_bytes()
```

- [ ] **Step 2: Run aggregation tests and confirm RED**

Run:

```bash
PYTHONPATH="$PWD/src" pytest \
  tests/experiments/test_vertical_runner.py -k aggregate -q
```

Expected: fails because `aggregate_vertical_run` is missing.

- [ ] **Step 3: Implement `VerticalAggregate` and flattening**

```python
@dataclass(frozen=True)
class VerticalAggregate:
    run_identity: str
    planned_tasks: int
    completed_tasks: int
    failed_tasks: int
    missing_tasks: int
    complete: bool
```

Flatten each SCEU into the fields fixed by the design specification. Sort
tasks by `task_index`, preserve SCEU order inside each task, and write canonical
JSONL with a trailing newline.

- [ ] **Step 4: Implement deterministic summary**

Compute:

```python
group_key = f"{condition}:{intervention_state_id or 'baseline'}"
mean_behavior_score = round(sum(scores) / len(scores), 6)
complete_chain = bool(stored and retrieved and visible and used)
score_delta = round(intervention_score - baseline_score, 6)
action_changes = sum(a != b for a, b in zip(intervention_actions, baseline_actions))
```

The summary must contain counts for planned/completed/failed/missing tasks,
group means, selected actions, drift flags, fake-native chain coverage, and
one leave-one-out record per non-null fake-native intervention.

- [ ] **Step 5: Verify deterministic aggregation GREEN**

Run:

```bash
PYTHONPATH="$PWD/src" pytest tests/experiments/test_vertical_runner.py -q
```

Expected: all experiment runner tests pass.

- [ ] **Step 6: Commit Task 4**

```bash
git add src/lhmsb/experiments tests/experiments/test_vertical_runner.py
git commit -m "feat: aggregate vertical pilot results"
```

### Task 5: Local and Slurm-compatible CLI

**Files:**

- Create: `src/lhmsb/experiments/vertical.py`
- Create: `tests/experiments/test_vertical_cli.py`

- [ ] **Step 1: Write CLI behavior tests**

```python
def test_cli_run_completes_default_matrix(
    frozen_vertical: Path, offline_config: Path, tmp_path: Path
) -> None:
    run_dir = tmp_path / "pilot"
    status = main([
        "run",
        "--dataset", str(frozen_vertical),
        "--config", str(offline_config),
        "--out", str(run_dir),
        "--allow-dirty",
    ])
    assert status == 0
    assert json.loads((run_dir / "summary.json").read_text())["complete"] is True


def test_cli_partial_aggregate_returns_nonzero(
    frozen_vertical: Path, offline_config: Path, tmp_path: Path
) -> None:
    run_dir = tmp_path / "pilot"
    assert main([
        "plan", "--dataset", str(frozen_vertical),
        "--config", str(offline_config), "--out", str(run_dir), "--allow-dirty",
    ]) == 0
    assert main(["run-task", "--run-dir", str(run_dir), "--task-index", "0"]) == 0
    assert main(["aggregate", "--run-dir", str(run_dir)]) == 1
```

- [ ] **Step 2: Run CLI tests and confirm RED**

Run:

```bash
PYTHONPATH="$PWD/src" pytest tests/experiments/test_vertical_cli.py -q
```

Expected: import fails because `lhmsb.experiments.vertical` does not exist.

- [ ] **Step 3: Implement argparse commands**

`main(argv: Sequence[str] | None = None) -> int` dispatches:

```python
if args.command == "plan":
    plan_vertical_run(
        args.dataset, args.config, args.out,
        allow_dirty=args.allow_dirty, force=args.force,
    )
    return 0
if args.command == "run":
    aggregate = run_vertical_matrix(
        args.dataset, args.config, args.out,
        allow_dirty=args.allow_dirty, force=args.force,
    )
    return 0 if aggregate.complete else 1
if args.command == "run-task":
    run_vertical_task(args.run_dir, args.task_index, force=args.force)
    return 0
if args.command == "aggregate":
    aggregate = aggregate_vertical_run(args.run_dir)
    return 0 if aggregate.complete else 1
```

Catch `VerticalExperimentError`, `StatefulDatasetError`, `OSError`,
`ValueError`, and `KeyError`; print `<command> FAILED: <type>: <message>` to
stderr and return `2`. Do not catch `KeyboardInterrupt`.

- [ ] **Step 4: Verify CLI GREEN**

Run:

```bash
PYTHONPATH="$PWD/src" pytest tests/experiments/test_vertical_cli.py -q
```

Expected: all CLI tests pass.

- [ ] **Step 5: Commit Task 5**

```bash
git add src/lhmsb/experiments/vertical.py tests/experiments/test_vertical_cli.py
git commit -m "feat: add vertical offline pilot CLI"
```

### Task 6: Integration gate, documentation check, and offline pilot

**Files:**

- Modify: `tests/experiments/test_vertical_cli.py`
- Verify: `configs/vertical_offline_pilot.yaml`
- Verify: `docs/superpowers/specs/2026-07-16-vertical-offline-pilot-design.md`

- [ ] **Step 1: Add subprocess module-entry integration test**

```python
def test_python_module_entry_is_reproducible(
    frozen_vertical: Path, tmp_path: Path
) -> None:
    run_dir = tmp_path / "pilot"
    command = [
        sys.executable, "-m", "lhmsb.experiments.vertical", "run",
        "--dataset", str(frozen_vertical),
        "--config", str(PROJECT_ROOT / "configs/vertical_offline_pilot.yaml"),
        "--out", str(run_dir),
        "--allow-dirty",
    ]
    env = {**os.environ, "PYTHONPATH": str(PROJECT_ROOT / "src")}
    first = subprocess.run(command, env=env, text=True, capture_output=True, check=False)
    assert first.returncode == 0, first.stderr
    before = {
        name: (run_dir / name).read_bytes()
        for name in ("task_results.jsonl", "sceu_results.jsonl", "summary.json")
    }
    second = subprocess.run(command, env=env, text=True, capture_output=True, check=False)
    assert second.returncode == 0, second.stderr
    assert before == {
        name: (run_dir / name).read_bytes()
        for name in before
    }
```

- [ ] **Step 2: Run the complete new test surface**

```bash
PYTHONPATH="$PWD/src" pytest \
  tests/datasets/test_stateful_loader.py \
  tests/experiments \
  tests/longhorizon \
  tests/families/test_software_vertical.py \
  tests/datasets/test_stateful_pipeline.py -q
```

Expected: all tests pass.

- [ ] **Step 3: Run static checks**

```bash
ruff check \
  src/lhmsb/datasets/stateful_loader.py \
  src/lhmsb/experiments \
  tests/datasets/test_stateful_loader.py \
  tests/experiments

mypy \
  src/lhmsb/datasets/stateful_loader.py \
  src/lhmsb/experiments
```

Expected: both commands exit zero.

- [ ] **Step 4: Run a real offline pilot from the sealed exemplar**

Because implementation leaves the worktree dirty until committed, use
`--allow-dirty` for this development acceptance run:

```bash
PYTHONPATH="$PWD/src" python -m lhmsb.experiments.vertical run \
  --dataset runs/releases/software-vertical-v0.1.0/software_v1 \
  --config configs/vertical_offline_pilot.yaml \
  --out runs/pilots/software-vertical-offline-dev \
  --allow-dirty
```

Expected: six tasks complete and `summary.json` reports `"complete": true`.

- [ ] **Step 5: Verify sealed input remains intact**

```bash
cd runs/releases/software-vertical-v0.1.0
shasum -a 256 -c software_v1-6b4edbf.tar.gz.sha256
cd ../../..
PYTHONPATH="$PWD/src" python -m lhmsb.datasets verify-stateful \
  --frozen runs/releases/software-vertical-v0.1.0/software_v1
```

Expected: archive and all 44 frozen files pass.

- [ ] **Step 6: Run the repository regression suite**

```bash
PYTHONPATH="$PWD/src" pytest -q \
  -k "not test_resource_module_is_linux and not test_results_persisted_as_jsonl_and_parquet_keyed_by_track"
```

Expected on the current macOS environment: all selected tests pass; the two
known platform/optional-parquet tests are deselected explicitly.

- [ ] **Step 7: Commit the integration gate**

```bash
git add src/lhmsb configs/vertical_offline_pilot.yaml \
  tests/datasets/test_stateful_loader.py tests/experiments
git commit -m "test: verify vertical offline pilot workflow"
```

- [ ] **Step 8: Record the post-implementation release boundary**

Create an annotated tag only after all checks pass:

```bash
git tag -a software-vertical-offline-pilot-v0.1.0 \
  -m "Frozen-loader and offline pilot driver for software-vertical-v0.1.0"
```

Do not move or replace `software-vertical-v0.1.0`.

# Vertical Offline Pilot Driver Design

**Status:** Approved for implementation

**Date:** 2026-07-16

**Base release:** `software-vertical-v0.1.0`

**Base code:** `6b4edbf108336dba945dbff7825c59fea447ec60`

## 1. Objective

Build the smallest reproducible experiment layer needed to run the frozen
Software Project vertical slice locally or as independent cluster jobs.

The experiment layer must:

1. load `SoftwareVerticalSpec` objects only from a verified frozen dataset;
2. create the fixed offline condition and intervention matrix;
3. execute either the whole matrix sequentially or one atomic task by index;
4. resume safely without silently overwriting successful work;
5. aggregate task outputs into per-SCEU records and a compact pilot summary;
6. bind every result to the code commit, dataset manifest, and canonical config.

The frozen `software-vertical-v0.1.0` tag and its dataset archive are immutable
inputs. This feature is developed in a new commit after that release.

## 2. Scope

### Included

- Frozen Software vertical dataset loading and integrity verification.
- A versioned YAML configuration for the offline pilot.
- Deterministic task planning.
- Local sequential execution.
- Slurm-compatible atomic execution by task index.
- Conditions:
  - `workspace_only`;
  - `oracle_current_state`;
  - `fake_native`.
- Fake-native leave-one-memory-out interventions:
  - baseline;
  - remove `P2`;
  - remove `C1`;
  - remove `U1`.
- Immutable run identity and safe resumption.
- Per-task, per-SCEU, and aggregate JSON/JSONL outputs.
- Programmatic and CLI integration tests.

### Excluded

- Local LLM inference on A100.
- vLLM or Transformers model policies.
- Mem0, Letta, Graphiti, Cognee, or other real memory backends.
- LLM-based alignment between native memory objects and evaluator state IDs.
- Formal statistical analysis or paper tables.
- Regeneration or modification of frozen datasets during experiment execution.

## 3. Architecture

The implementation uses a shared execution core with four CLI operations:

```text
frozen dataset
    │
    ▼
verify + load specs
    │
    ▼
canonical config + immutable run identity
    │
    ▼
tasks.jsonl
    ├── local: run every task sequentially
    └── cluster: run one task by array index
              │
              ▼
      tasks/<task_id>/result.json
              │
              ▼
      aggregate completed task results
              │
              ├── task_results.jsonl
              ├── sceu_results.jsonl
              └── summary.json
```

The same atomic worker is used by local sequential runs and Slurm array jobs.
No cluster-specific experiment logic is permitted.

## 4. Component Boundaries

### 4.1 Frozen loader

Create `src/lhmsb/datasets/stateful_loader.py`.

Public interface:

```python
def load_software_vertical_specs(
    frozen: Path,
    *,
    verify: bool = True,
) -> tuple[SoftwareVerticalSpec, ...]:
    ...
```

The loader:

- calls `verify_stateful()` before reading by default;
- rejects checksum mismatches, missing files, unsupported schema versions, and
  non-Software families;
- reads `episodes.jsonl`;
- reconstructs `EpisodePlan` with `EpisodePlan.from_dict()`;
- reconstructs actions with `ActionSpec.from_dict()`;
- restores package files and hidden tests from the frozen record;
- checks episode ID, seed, session count, plan hash, surface hash, and workspace
  hash against the record and manifest;
- returns specs in the order stored in `episodes.jsonl`;
- never imports or calls `SoftwareVerticalFamily.generate()`.

Skipping verification is an internal/testing escape hatch, not exposed by the
experiment CLI.

### 4.2 Experiment configuration and identities

Create `src/lhmsb/experiments/vertical_config.py`.

Core frozen types:

```python
VerticalOfflineConfig
VerticalTask
VerticalRunIdentity
VerticalRunManifest
```

The default configuration is stored at
`configs/vertical_offline_pilot.yaml`:

```yaml
schema_version: 1
experiment_id: software-vertical-offline-pilot
conditions:
  workspace_only: [null]
  oracle_current_state: [null]
  fake_native: [null, P2, C1, U1]
```

`null` means no intervention. Unknown conditions, duplicate matrix entries,
empty matrices, and interventions on non-`fake_native` conditions are rejected.

The config hash is SHA-256 over canonical JSON produced from the parsed config,
not over YAML formatting. The normalized JSON represents `conditions` as an
ordered list of condition/intervention records, so changing matrix order also
changes the config hash and task indices.

The run identity is SHA-256 over:

```text
experiment schema version
code commit SHA
dirty-worktree flag
dataset MANIFEST.json SHA-256
dataset schema/generator versions
canonical config hash
```

The dataset path, output path, host name, and timestamps are recorded in the run
manifest but do not affect identity.

### 4.3 Task planner and atomic worker

Create `src/lhmsb/experiments/vertical_runner.py`.

Public interfaces:

```python
def plan_vertical_run(...) -> VerticalRunManifest:
    ...

def run_vertical_task(
    run_dir: Path,
    task_index: int,
    *,
    force: bool = False,
) -> Path:
    ...

def aggregate_vertical_run(run_dir: Path) -> VerticalAggregate:
    ...

def run_vertical_matrix(...) -> VerticalAggregate:
    ...
```

Task ordering is deterministic:

1. episode order in the frozen dataset;
2. condition order in the YAML configuration;
3. intervention order in each condition.

For the single frozen exemplar, the default plan contains six tasks:

```text
workspace_only / baseline
oracle_current_state / baseline
fake_native / baseline
fake_native / remove-P2
fake_native / remove-C1
fake_native / remove-U1
```

Each `VerticalTask` records:

- zero-based task index;
- stable task ID;
- episode ID;
- condition;
- intervention state ID or `null`;
- run identity;
- task payload hash.

The worker reloads and verifies the frozen dataset, selects the recorded
episode, calls `run_vertical_episode()`, and writes:

```text
tasks/<task_id>/result.json
```

It writes to a temporary sibling file and commits with `os.replace()`. A worker
failure writes `failure.json` using the same atomic pattern and exits non-zero.
No two tasks write the same file.

Before execution, the worker also verifies that the current checkout SHA and
dirty-worktree state match `run_manifest.json`. A task planned from one code
version cannot be executed by another checkout.

### 4.4 CLI

Create:

```text
src/lhmsb/experiments/__init__.py
src/lhmsb/experiments/vertical.py
```

Commands:

```bash
python -m lhmsb.experiments.vertical plan \
  --dataset DATASET --config CONFIG --out RUN_DIR

python -m lhmsb.experiments.vertical run \
  --dataset DATASET --config CONFIG --out RUN_DIR

python -m lhmsb.experiments.vertical run-task \
  --run-dir RUN_DIR --task-index INDEX

python -m lhmsb.experiments.vertical aggregate \
  --run-dir RUN_DIR
```

Common mutation semantics:

- A new run directory receives `run_config.yaml`, `run_manifest.json`, and
  `tasks.jsonl`.
- Replanning an existing directory with the same identity is idempotent.
- A different code SHA, dataset manifest, or config hash is rejected.
- A successful task is skipped by default.
- A failed or interrupted task may be rerun.
- `--force` permits replacement of an existing plan or successful task.
- A dirty Git worktree is rejected by default.
- `--allow-dirty` is accepted only by `plan` and `run`, marks the manifest, and
  becomes part of the run identity.

`run` is exactly `plan`, followed by `run-task` for every index, followed by
`aggregate`.

Planning with `--force` clears only files managed by this experiment format and
only when the directory already contains a recognized vertical run manifest. A
non-empty unrelated directory is always rejected.

### 4.5 Aggregation

Aggregation reads only complete `result.json` files whose run identity and task
payload hash match `tasks.jsonl`.

Outputs:

```text
task_results.jsonl
sceu_results.jsonl
summary.json
```

`task_results.jsonl` contains one canonical record per completed task.

`sceu_results.jsonl` contains one flattened record per SCEU with at least:

```text
run_identity
task_id
task_index
episode_id
condition
intervention_state_id
sceu_id
opportunity_id
stored_state_ids
retrieved_state_ids
model_visible_state_ids
used_state_ids
selected_action
behavior_score
is_correct
violated_state_ids
drift_flags
workspace_snapshot_hash
prefix_hash
transcript_hash
```

`summary.json` records:

- planned, completed, failed, and missing task counts;
- whether the run is complete;
- mean behavior score by condition/intervention;
- selected-action counts;
- drift-flag counts;
- stored/retrieved/visible/used chain coverage;
- leave-one-out action and score deltas relative to fake-native baseline.

Aggregation always writes a summary for completed tasks. Its CLI exits non-zero
when planned tasks are missing or failed, allowing schedulers to detect an
incomplete run without losing partial diagnostics.

Aggregate files contain no creation timestamp and are sorted by task index and
SCEU order. Re-aggregation from the same task results is byte-identical.

## 5. Run Directory Contract

```text
<run_dir>/
  run_config.yaml
  run_manifest.json
  tasks.jsonl
  tasks/
    <task_id>/
      result.json
      failure.json              # absent after a successful rerun
  task_results.jsonl
  sceu_results.jsonl
  summary.json
```

`run_manifest.json` includes:

- run identity and manifest schema;
- code SHA and dirty flag;
- branch/tag when available;
- dataset path and `MANIFEST.json` SHA-256;
- dataset schema version, generator version, and episode hashes;
- canonical config and config hash;
- Python, package, OS, host, and timestamp metadata.

Secrets and environment-variable values are never written.

## 6. Resume and Failure Semantics

### Existing successful task

If `result.json` matches the task and run identity, return it without rewriting
the file. This preserves modification time and gives observable skip semantics.

### Existing stale or corrupt result

Reject it unless `--force` is supplied. Never treat malformed JSON as completed.

### Failed task

Write the exception class and message to `failure.json`, without credentials,
traceback-local values, or environment contents. A later invocation may retry
the task without `--force`. Success removes the matching `failure.json`.

### Identity mismatch

Planning or execution stops before writing experimental results. `--force` on
planning may replace the run directory metadata and task outputs. It does not
modify the dataset.

### Dataset mutation

Every plan and task execution verifies the frozen dataset. Any checksum change
causes a hard failure before calling the vertical runner.

## 7. Testing Strategy

### Loader tests

- A valid frozen directory round-trips to the original spec.
- Corrupting one file causes loading to fail.
- Stored plan/surface/workspace hash mismatches are rejected.
- The loader succeeds when the generator is monkeypatched to raise, proving it
  does not regenerate data.

### Planner tests

- One exemplar produces exactly six deterministically ordered tasks.
- Replanning with identical inputs is idempotent.
- Changed config or dataset identity is rejected.
- Dirty worktrees require explicit `--allow-dirty`.

### Worker tests

- Any task index can execute independently.
- Re-running a successful task leaves its result unchanged.
- A failed task can be retried.
- `--force` replaces a stale result.
- P2 leave-one-out changes behavior in the expected direction.

### Aggregation tests

- Sequential and shuffled task execution produce byte-identical aggregate
  outputs.
- Partial runs produce a useful but incomplete summary and non-zero CLI status.
- Flattened rows reconstruct
  `stored → retrieved → visible → used → behavior`.
- Leave-one-out deltas are computed against the matching fake-native baseline.

### Integration gate

The final test creates a fresh frozen fixture and runs:

```bash
python -m lhmsb.experiments.vertical run \
  --dataset <fixture>/software_v1 \
  --config configs/vertical_offline_pilot.yaml \
  --out <fixture>/pilot \
  --allow-dirty
```

Acceptance conditions:

- six tasks complete;
- aggregate output is marked complete;
- all three baseline conditions are present;
- P2, C1, and U1 interventions are present;
- at least one intervention changes action or behavior score;
- every output records the same run identity;
- a second invocation skips successful tasks and produces identical aggregates;
- the source frozen dataset remains byte-identical.

## 8. Server Workflow After Implementation

Local/CPU:

```bash
python -m lhmsb.experiments.vertical run \
  --dataset /data/lhmsb/datasets/software_v1 \
  --config configs/vertical_offline_pilot.yaml \
  --out /data/lhmsb/runs/offline-pilot-001
```

Slurm planning:

```bash
python -m lhmsb.experiments.vertical plan \
  --dataset /data/lhmsb/datasets/software_v1 \
  --config configs/vertical_offline_pilot.yaml \
  --out /data/lhmsb/runs/offline-pilot-001
```

Array execution:

```bash
python -m lhmsb.experiments.vertical run-task \
  --run-dir /data/lhmsb/runs/offline-pilot-001 \
  --task-index "$SLURM_ARRAY_TASK_ID"
```

Aggregation:

```bash
python -m lhmsb.experiments.vertical aggregate \
  --run-dir /data/lhmsb/runs/offline-pilot-001
```

This offline pilot is CPU-only. A100 allocation begins only after replacing the
deterministic policy with a pinned model policy in a separate design and release.

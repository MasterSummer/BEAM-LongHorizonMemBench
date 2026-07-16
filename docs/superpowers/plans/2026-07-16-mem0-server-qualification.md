# Mem0 Server Qualification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver every repository-side prerequisite needed to copy a frozen, leak-free Software benchmark plus a pinned Mem0 runtime to an A100 server, execute the Controlled and Native qualification tracks, and emit validated write/retrieval/use/behavior metrics.

**Architecture:** Preserve the legacy deterministic vertical slice as a byte-identical regression fixture. Add a separate v0.2 public/evaluator dataset boundary, then build a new `lhmsb.qualification` runtime around immutable task identities, provider-neutral policy calls, Mem0 2.0.12 lifecycle traces, evaluator-side attribution, counterfactual readouts, aggregation, and deployment gates. Docker Compose and Slurm invoke the same Python CLI and data-root layout.

**Tech Stack:** Python 3.11, frozen dataclasses, PyYAML, HTTPX, OpenAI/Anthropic SDKs, Mem0 OSS 2.0.12, Qdrant, Hugging Face Text Embeddings Inference, pytest, mypy, ruff, Docker Compose, Slurm.

---

## Task 1: Lock Dependencies and Protect the Legacy Baseline

**Files:**

- Modify: `pyproject.toml`
- Create: `constraints/mem0.lock.txt`
- Create: `tests/qualification/test_dependency_contract.py`
- Create: `datasets/releases/software-vertical-v0.1.0/RELEASE.json`
- Copy unchanged: `datasets/releases/software-vertical-v0.1.0/software_v1-6b4edbf.tar.gz`
- Copy unchanged: `datasets/releases/software-vertical-v0.1.0/software_v1-6b4edbf.tar.gz.sha256`

- [ ] **Step 1: Write the failing dependency and release tests**

```python
def test_mem0_extra_is_exactly_pinned() -> None:
    assert project_optional_dependency("mem0") == "mem0ai==2.0.12"


def test_legacy_release_archive_matches_declared_sha() -> None:
    release = load_release("software-vertical-v0.1.0")
    assert sha256(release.archive) == release.archive_sha256
```

- [ ] **Step 2: Run the tests and confirm the expected failures**

Run:

```bash
uv run pytest tests/qualification/test_dependency_contract.py -q
```

Expected: failures for the unpinned dependency and missing tracked release.

- [ ] **Step 3: Pin the runtime**

Use these exact contracts:

```text
mem0ai==2.0.12
wheel sha256 6b7e1afa466f6e14dd34b5e9222c159a69fad38f8d787e73adbf91dbb29e73e2
source commit 42cf18c4e6adb448e981aa1c7b55c1602b0cb670
```

Add a `qualification` optional dependency group containing exact major/minor
constraints for `httpx`, `openai`, `anthropic`, `qdrant-client`, `pandas`,
`pyarrow`, and `duckdb`. Keep imports lazy so ordinary CI does not require
provider credentials or live services.

- [ ] **Step 4: Copy and verify the legacy archive without regenerating it**

Run:

```bash
shasum -a 256 datasets/releases/software-vertical-v0.1.0/software_v1-6b4edbf.tar.gz
```

Expected: the value already stored in the copied `.sha256` file.

- [ ] **Step 5: Regenerate the uv lock and run the tests**

Run:

```bash
uv lock
uv run pytest tests/qualification/test_dependency_contract.py -q
```

Expected: pass.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock constraints datasets/releases/software-vertical-v0.1.0 tests/qualification/test_dependency_contract.py
git commit -m "build: lock Mem0 qualification dependencies"
```

## Task 2: Define the Public/Evaluator Surface Boundary

**Files:**

- Create: `src/lhmsb/longhorizon/public_surface.py`
- Modify: `src/lhmsb/longhorizon/__init__.py`
- Create: `tests/longhorizon/test_public_surface.py`

- [ ] **Step 1: Write failing serialization and leak tests**

Cover:

- frozen `PublicActionOption`, `PublicContinuation`, and
  `EvaluatorContinuation`;
- canonical JSON and stable hashes;
- opaque action mapping round trip;
- rejection of state IDs, latent action IDs, evaluator field names, validity
  labels, answer-revealing phrases, comments, and docstrings;
- recursive scanning of both keys and string values.

```python
with pytest.raises(SurfaceLeakError, match="valid_action_ids"):
    validate_public_payload({"valid_action_ids": ["safe_v2_offline"]}, policy)
```

- [ ] **Step 2: Run the test**

```bash
uv run pytest tests/longhorizon/test_public_surface.py -q
```

Expected: import failure.

- [ ] **Step 3: Implement immutable public types**

Required public API:

```python
@dataclass(frozen=True)
class PublicActionOption:
    option_id: str
    files: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class PublicContinuation:
    opportunity_id: str
    checkpoint_session: int
    request: str
    options: tuple[PublicActionOption, ...]


@dataclass(frozen=True)
class EvaluatorContinuation:
    opportunity_id: str
    option_to_action: tuple[tuple[str, str], ...]
```

Implement comment/docstring stripping with Python `tokenize`/`ast`, not regular
expressions alone. Permutation must be seeded from
`sha256(episode_id + opportunity_id + semantic_seed)`.

- [ ] **Step 4: Implement `validate_public_payload`**

Return a structured report on success and raise `SurfaceLeakError` on any
forbidden match. The forbidden vocabulary is supplied by the dataset profile
and includes all gold state IDs and latent action IDs.

- [ ] **Step 5: Run focused checks**

```bash
uv run pytest tests/longhorizon/test_public_surface.py -q
uv run mypy src/lhmsb/longhorizon/public_surface.py
uv run ruff check src/lhmsb/longhorizon/public_surface.py tests/longhorizon/test_public_surface.py
```

Expected: pass.

- [ ] **Step 6: Commit**

```bash
git add src/lhmsb/longhorizon tests/longhorizon/test_public_surface.py
git commit -m "feat: add leak-safe public continuation boundary"
```

## Task 3: Generate the Leak-Free Software v0.2 Template

**Files:**

- Create: `src/lhmsb/families/software/mem0_vertical.py`
- Modify: `src/lhmsb/families/software/__init__.py`
- Create: `tests/families/test_software_mem0_vertical.py`

- [ ] **Step 1: Write failing semantic tests**

Assert:

- template ID is `software-project-mem0-v2`;
- `G0` does not imply offline operation;
- `C1` is the only state that forbids cloud services;
- public boilerplate is neutral;
- explicit/derivable/absent variants share latent state;
- raw workspaces are not automatically copied into write transcripts;
- public action option labels are opaque;
- all evaluator annotations remain private;
- 4-session and 16-session plans have the same schema.

- [ ] **Step 2: Run the test**

```bash
uv run pytest tests/families/test_software_mem0_vertical.py -q
```

Expected: import failure.

- [ ] **Step 3: Implement `SoftwareMem0VerticalFamily`**

Reuse state/replay/checker primitives, but do not mutate
`SoftwareVerticalFamily`. Use:

```text
G0: Build a reproducible and auditable experiment pipeline.
C1: Pipeline execution must remain completely offline; do not call cloud services.
C2: The held-out test set must never be modified.
```

Keep branch replacement, revocation, local-scope conflict, valid-update,
fresh-reminder, early/late matched opportunity, and drift controls.

- [ ] **Step 4: Implement public session rendering**

Only explicit artifact reads become public tool results and future Mem0 write
input. The current workspace itself is still delivered to every policy
condition. Never put `source_event_ids`, recoverability labels, state IDs, or
future values in public records.

- [ ] **Step 5: Run focused checks**

```bash
uv run pytest tests/families/test_software_mem0_vertical.py tests/longhorizon/test_public_surface.py -q
uv run mypy src/lhmsb/families/software/mem0_vertical.py
uv run ruff check src/lhmsb/families/software/mem0_vertical.py tests/families/test_software_mem0_vertical.py
```

Expected: pass.

- [ ] **Step 6: Commit**

```bash
git add src/lhmsb/families/software tests/families/test_software_mem0_vertical.py
git commit -m "feat: add leak-free Software Mem0 vertical"
```

## Task 4: Freeze and Verify the v0.2 Dataset Release

**Files:**

- Create: `src/lhmsb/datasets/mem0_stateful_pipeline.py`
- Modify: `src/lhmsb/datasets/cli.py`
- Modify: `src/lhmsb/datasets/__init__.py`
- Create: `tests/datasets/test_mem0_stateful_pipeline.py`
- Create: `datasets/releases/software-vertical-mem0-v0.2.0/RELEASE.json`
- Generate: `datasets/releases/software-vertical-mem0-v0.2.0/software_mem0_v2.tar.gz`
- Generate: `datasets/releases/software-vertical-mem0-v0.2.0/software_mem0_v2.tar.gz.sha256`

- [ ] **Step 1: Write failing release-pipeline tests**

Test:

- deterministic staging/freeze/verify/regen;
- separate `public/` and `evaluator/` trees;
- archive has reproducible member order, mode, uid/gid, and mtime;
- manifest includes generator, plan, public surface, workspace, evaluator, and
  archive hashes;
- leak scan and recoverability audit are mandatory freeze gates;
- v0.1 code paths and hashes are unchanged.

- [ ] **Step 2: Run the tests**

```bash
uv run pytest tests/datasets/test_mem0_stateful_pipeline.py -q
```

Expected: import failure.

- [ ] **Step 3: Add CLI profile**

Commands:

```bash
python -m lhmsb.datasets generate-mem0-stateful ...
python -m lhmsb.datasets freeze-mem0-stateful ...
python -m lhmsb.datasets verify-mem0-stateful ...
python -m lhmsb.datasets regen-check-mem0-stateful ...
```

The frozen tree must include public sessions/workspaces/continuations,
evaluator state/events/signatures/mappings/SCEUs, hashes, and dataset card.

- [ ] **Step 4: Generate the tracked seed-42 release**

Use 16 sessions and one episode. Build the tar archive deterministically with
gzip timestamp zero.

- [ ] **Step 5: Verify both releases**

```bash
uv run python -m lhmsb.datasets verify-mem0-stateful --frozen runs/vertical/software_mem0_v2
uv run python -m lhmsb.datasets regen-check-mem0-stateful --frozen runs/vertical/software_mem0_v2
shasum -a 256 -c datasets/releases/software-vertical-mem0-v0.2.0/software_mem0_v2.tar.gz.sha256
shasum -a 256 -c datasets/releases/software-vertical-v0.1.0/software_v1-6b4edbf.tar.gz.sha256
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/lhmsb/datasets tests/datasets/test_mem0_stateful_pipeline.py datasets/releases/software-vertical-mem0-v0.2.0
git commit -m "feat: freeze leak-free Mem0 qualification dataset"
```

## Task 5: Define Qualification Configuration and Immutable Task Identities

**Files:**

- Create: `src/lhmsb/qualification/__init__.py`
- Create: `src/lhmsb/qualification/config.py`
- Create: `src/lhmsb/qualification/schema.py`
- Create: `tests/qualification/test_config.py`
- Create: `configs/models/claude-opus-4-8.yaml`
- Create: `configs/models/deepseek-v4-pro.yaml`
- Create: `configs/models/gpt-5.6-sol.yaml`
- Create: `configs/models/bge-m3.yaml`
- Create: `configs/models/bge-reranker-v2-m3.yaml`
- Create: `configs/systems/mem0/controlled.yaml`
- Create: `configs/systems/mem0/native.yaml`
- Create: `configs/experiments/mem0_qualification.yaml`
- Create: `configs/systems.lock.yaml`

- [ ] **Step 1: Write failing config tests**

Cover unique-key YAML loading, exact model IDs, exact BGE revisions, candidate
20/visible 5, track separation, no deferred systems, secret environment names
without secret values, and deterministic 12-task/15-result expansion.

- [ ] **Step 2: Run the tests**

```bash
uv run pytest tests/qualification/test_config.py -q
```

Expected: import failure.

- [ ] **Step 3: Implement frozen config dataclasses**

Required top-level types:

```python
PolicyProfile
RetrievalProfile
Mem0Profile
QualificationConfig
QualificationTask
ScoredCondition
RunIdentityInputs
```

Controlled task expansion yields paired native/common-rerank result branches
from one store. Reject any duplicated model, condition, result ID, or output
namespace.

- [ ] **Step 4: Add canonical hashing and redaction**

All effective non-secret values enter run identity. Environment variable values
are redacted; required variable names enter the manifest.

- [ ] **Step 5: Run focused checks**

```bash
uv run pytest tests/qualification/test_config.py -q
uv run mypy src/lhmsb/qualification/config.py src/lhmsb/qualification/schema.py
uv run ruff check src/lhmsb/qualification tests/qualification/test_config.py
```

Expected: pass.

- [ ] **Step 6: Commit**

```bash
git add src/lhmsb/qualification configs tests/qualification/test_config.py
git commit -m "feat: define Mem0 qualification matrix"
```

## Task 6: Implement Provider-Neutral Structured Policy Clients

**Files:**

- Create: `src/lhmsb/qualification/providers.py`
- Create: `tests/qualification/test_providers.py`

- [ ] **Step 1: Write fake-transport contract tests**

Test successful tool calls, usage normalization, retries, one format repair,
model mismatch, unavailable model, timeout/rate-limit classification, endpoint
redaction, and no provider fallback.

- [ ] **Step 2: Run the tests**

```bash
uv run pytest tests/qualification/test_providers.py -q
```

Expected: import failure.

- [ ] **Step 3: Implement the common API**

```python
class PolicyClient(Protocol):
    def submit_action(self, request: PolicyRequest) -> PolicyResponse: ...


@dataclass(frozen=True)
class PolicyUsage:
    input_tokens: int | None
    output_tokens: int | None
    cached_tokens: int | None
    reasoning_tokens: int | None
```

Implement OpenAI-compatible and Anthropic clients. DeepSeek uses the
OpenAI-compatible transport with its own endpoint/profile. Accept injected
transports for tests.

- [ ] **Step 4: Hash the exact request**

Record effective parameters after unsupported fields are removed. Persist the
opaque selected option before evaluator mapping.

- [ ] **Step 5: Run focused checks**

```bash
uv run pytest tests/qualification/test_providers.py -q
uv run mypy src/lhmsb/qualification/providers.py
uv run ruff check src/lhmsb/qualification/providers.py tests/qualification/test_providers.py
```

Expected: pass.

- [ ] **Step 6: Commit**

```bash
git add src/lhmsb/qualification/providers.py tests/qualification/test_providers.py
git commit -m "feat: add structured policy provider clients"
```

## Task 7: Implement Local Embedding and Reranker Service Clients

**Files:**

- Create: `src/lhmsb/qualification/tei.py`
- Create: `tests/qualification/test_tei.py`

- [ ] **Step 1: Write fake-server tests**

Verify embedding dimension 1024, batching, deterministic rerank ordering,
response validation, content hashes, service health, latency fields, and
failure classification.

- [ ] **Step 2: Run the tests**

```bash
uv run pytest tests/qualification/test_tei.py -q
```

Expected: import failure.

- [ ] **Step 3: Implement `EmbeddingClient` and `RerankerClient`**

Use HTTPX with injected transports. Preserve native candidate order separately
from common-rerank order. Tie-break equal reranker scores by native rank.

- [ ] **Step 4: Run focused checks**

```bash
uv run pytest tests/qualification/test_tei.py -q
uv run mypy src/lhmsb/qualification/tei.py
uv run ruff check src/lhmsb/qualification/tei.py tests/qualification/test_tei.py
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add src/lhmsb/qualification/tei.py tests/qualification/test_tei.py
git commit -m "feat: add pinned TEI retrieval clients"
```

## Task 8: Add a Mem0 2.0.12 Qualification Adapter

**Files:**

- Create: `src/lhmsb/adapters/mem0_qualification.py`
- Modify: `src/lhmsb/adapters/__init__.py`
- Create: `tests/contract/test_mem0_qualification.py`

- [ ] **Step 1: Write a fake Mem0 2.0.12 backend**

Exercise:

```python
add(messages, user_id=..., run_id=..., metadata=..., infer=True)
search(query, filters=..., top_k=20, threshold=0.0, rerank=False)
get_all(filters=..., top_k=10000)
history(memory_id)
```

Cover ADD/UPDATE/DELETE/NONE rows, bare-list and `{"results": ...}` responses,
inventory/history delta, candidate ordering, shortfall, task namespace, and
telemetry disablement.

- [ ] **Step 2: Run the tests**

```bash
uv run pytest tests/contract/test_mem0_qualification.py -q
```

Expected: import failure.

- [ ] **Step 3: Implement a new adapter without altering legacy semantics**

Expose:

```python
write_session(...)
snapshot_inventory(...)
search_candidates(...)
history_delta(...)
```

Build explicit Controlled and Native Mem0 configs. Controlled uses the policy
model plus local BGE-M3/Qdrant. Native uses `gpt-5-mini`,
`text-embedding-3-small`, Qdrant, and no reranker.

- [ ] **Step 4: Instrument internal provider calls**

Accept a trace hook/client factory. Never invent observed usage. If usage is
absent, emit a separately named estimate with `observed=False`.

- [ ] **Step 5: Run focused checks**

```bash
uv run pytest tests/contract/test_mem0_qualification.py tests/contract/test_mem0.py -q
uv run mypy src/lhmsb/adapters/mem0_qualification.py
uv run ruff check src/lhmsb/adapters/mem0_qualification.py tests/contract/test_mem0_qualification.py
```

Expected: pass, including all legacy Mem0 adapter tests.

- [ ] **Step 6: Commit**

```bash
git add src/lhmsb/adapters tests/contract/test_mem0_qualification.py
git commit -m "feat: trace Mem0 2 lifecycle"
```

## Task 9: Implement Programmatic State Attribution

**Files:**

- Create: `src/lhmsb/longhorizon/attribution.py`
- Create: `tests/longhorizon/test_attribution.py`

- [ ] **Step 1: Write signature and provenance tests**

Cover exact signature, allowed variants, polarity/negation, version, authority,
scope, unique provenance, multiple matches, contradictions, and zero matches.
Ambiguous objects must not improve coverage or precision.

- [ ] **Step 2: Run the tests**

```bash
uv run pytest tests/longhorizon/test_attribution.py -q
```

Expected: import failure.

- [ ] **Step 3: Implement evaluator-only types**

```python
FactSignature
MemoryAttribution
AttributionMethod = Literal["exact_signature", "unique_provenance", "ambiguous"]
```

Normalize Unicode, whitespace, case, and punctuation deterministically. Do not
use an LLM judge or embedding threshold for gold assignment.

- [ ] **Step 4: Integrate signature generation into the v0.2 freeze**

Frozen evaluator records must include source sessions/events, canonical anchors,
allowed variants, negative anchors, and state predicates.

- [ ] **Step 5: Run focused checks**

```bash
uv run pytest tests/longhorizon/test_attribution.py tests/datasets/test_mem0_stateful_pipeline.py -q
uv run mypy src/lhmsb/longhorizon/attribution.py
```

Expected: pass.

- [ ] **Step 6: Commit**

```bash
git add src/lhmsb/longhorizon/attribution.py src/lhmsb/datasets tests/longhorizon/test_attribution.py
git commit -m "feat: add programmatic memory-state attribution"
```

## Task 10: Implement Causal-Use and Drift Classification

**Files:**

- Create: `src/lhmsb/longhorizon/interventions.py`
- Create: `src/lhmsb/longhorizon/drift.py`
- Create: `tests/longhorizon/test_interventions_and_drift.py`

- [ ] **Step 1: Write failing classification tests**

Cover:

- stable repeated baseline;
- unstable baseline;
- stable leave-one-out benefit/harm;
- stale/conflict replacement;
- unstable intervention;
- unchanged action but changed checker outcome;
- `constraint_loss`, `plan_deviation`, `stale_state`, and
  `local_over_global` drift flags;
- matched early/late decay.

- [ ] **Step 2: Run the tests**

```bash
uv run pytest tests/longhorizon/test_interventions_and_drift.py -q
```

Expected: import failure.

- [ ] **Step 3: Implement pure classifiers**

Classification consumes persisted outcomes and never calls a model itself.
Require agreement within each repeated pair before assigning causal use.

- [ ] **Step 4: Run focused checks**

```bash
uv run pytest tests/longhorizon/test_interventions_and_drift.py -q
uv run mypy src/lhmsb/longhorizon/interventions.py src/lhmsb/longhorizon/drift.py
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add src/lhmsb/longhorizon tests/longhorizon/test_interventions_and_drift.py
git commit -m "feat: classify causal memory use and drift"
```

## Task 11: Build the Real Qualification Runner

**Files:**

- Create: `src/lhmsb/qualification/runner.py`
- Create: `src/lhmsb/qualification/storage.py`
- Create: `tests/qualification/test_runner.py`

- [ ] **Step 1: Write fake end-to-end runner tests**

Use fake policy, Mem0, checker, embedding, and reranker components. Assert:

- 12 atomic tasks and 15 scored results;
- identical workspace across conditions;
- fresh working context each session;
- no raw-workspace auto-ingestion;
- isolated user/run/collection/history IDs;
- Controlled readouts share one store/candidate set;
- continuation answers never enter later prefix writes;
- model-visible hashes match exact policy requests;
- opaque option mapping happens only after response persistence;
- expected failures are typed and atomic.

- [ ] **Step 2: Run the tests**

```bash
uv run pytest tests/qualification/test_runner.py -q
```

Expected: import failure.

- [ ] **Step 3: Implement atomic prefix execution**

Persist each session write, inventory, and API trace before advancing. Make the
checkpoint immutable before continuation readouts.

- [ ] **Step 4: Implement continuation branches**

For each SCEU:

- workspace only;
- oracle current state;
- Mem0 native top 5;
- Controlled common-reranked top 5;
- required repeated baselines and interventions.

Use the existing programmatic Software checker after mapping the opaque option
to a latent action.

- [ ] **Step 5: Implement safe resume**

Completed cells reopen only if identity and trace hashes match. A failed
reranker branch does not invalidate the Controlled native branch. A failed or
incomplete Mem0 trace is unscoreable.

- [ ] **Step 6: Run focused checks**

```bash
uv run pytest tests/qualification/test_runner.py -q
uv run mypy src/lhmsb/qualification/runner.py src/lhmsb/qualification/storage.py
uv run ruff check src/lhmsb/qualification/runner.py src/lhmsb/qualification/storage.py tests/qualification/test_runner.py
```

Expected: pass.

- [ ] **Step 7: Commit**

```bash
git add src/lhmsb/qualification tests/qualification/test_runner.py
git commit -m "feat: execute resumable Mem0 qualification tasks"
```

## Task 12: Compute Metrics and Produce the Scorecard

**Files:**

- Create: `src/lhmsb/qualification/metrics.py`
- Create: `src/lhmsb/qualification/report.py`
- Create: `tests/qualification/test_metrics.py`
- Create: `tests/qualification/test_report.py`

- [ ] **Step 1: Write formula tests with hand-computed fixtures**

Cover all design metrics:

- write coverage/selectivity;
- current-state precision/recall/F1;
- stale retention and duplicate rate;
- update/delete responsiveness;
- write-to-continuation alignment;
- candidate/retrieval/visible metrics;
- native/common-rerank delta;
- causal use and intervention labels;
- four drift components and aggregate drift;
- workspace gain and oracle-gap closure;
- memory object counts;
- cost, latency, retry, and reliability totals.

- [ ] **Step 2: Run the tests**

```bash
uv run pytest tests/qualification/test_metrics.py tests/qualification/test_report.py -q
```

Expected: import failure.

- [ ] **Step 3: Implement denominator-safe metric functions**

Every ratio carries numerator, denominator, and nullable value. Never convert
an undefined metric to zero.

- [ ] **Step 4: Emit required artifacts**

```text
run_manifest.json
tasks.jsonl
task_results.jsonl
sceu_results.jsonl
memory_events.jsonl
memory_inventory.jsonl
retrieval_trace.jsonl
interventions.jsonl
api_usage.jsonl
metrics.json
summary.json
scorecard.csv
scorecard.md
```

Sort rows deterministically and hash every artifact.

- [ ] **Step 5: Run focused checks**

```bash
uv run pytest tests/qualification/test_metrics.py tests/qualification/test_report.py -q
uv run mypy src/lhmsb/qualification/metrics.py src/lhmsb/qualification/report.py
```

Expected: pass.

- [ ] **Step 6: Commit**

```bash
git add src/lhmsb/qualification tests/qualification/test_metrics.py tests/qualification/test_report.py
git commit -m "feat: export Mem0 lifecycle and drift metrics"
```

## Task 13: Add CLI, Preflight, and Artifact Validation

**Files:**

- Create: `src/lhmsb/qualification/cli.py`
- Create: `src/lhmsb/qualification/__main__.py`
- Create: `src/lhmsb/qualification/preflight.py`
- Create: `src/lhmsb/qualification/validate.py`
- Create: `tests/qualification/test_cli.py`
- Create: `tests/qualification/test_preflight.py`

- [ ] **Step 1: Write failing CLI/preflight tests**

Commands:

```text
plan
run-task
run-matrix
aggregate
validate
preflight
smoke
```

Test stop-on-first-failure, JSON reports, dry-run mode, live-gate
environment variables, dirty-tree policy, identity mismatch, and redaction.

- [ ] **Step 2: Run the tests**

```bash
uv run pytest tests/qualification/test_cli.py tests/qualification/test_preflight.py -q
```

Expected: import failure.

- [ ] **Step 3: Implement ordered preflight gates**

Implement all repository-verifiable gates locally and service/provider gates as
typed probes. No formal run starts automatically after preflight.

- [ ] **Step 4: Implement artifact schema/hash validation**

Reject missing trace layers, unknown IDs, candidate/retrieved/visible ordering
violations, incomplete interventions, or mismatched run identity.

- [ ] **Step 5: Run focused checks**

```bash
uv run pytest tests/qualification/test_cli.py tests/qualification/test_preflight.py -q
uv run python -m lhmsb.qualification --help
uv run mypy src/lhmsb/qualification
```

Expected: pass.

- [ ] **Step 6: Commit**

```bash
git add src/lhmsb/qualification tests/qualification/test_cli.py tests/qualification/test_preflight.py
git commit -m "feat: add qualification CLI and preflight gates"
```

## Task 14: Package Docker Compose and Slurm Execution

**Files:**

- Create: `docker/mem0-worker.Dockerfile`
- Create: `deploy/compose.mem0.yaml`
- Create: `deploy/slurm/mem0_preflight.sbatch`
- Create: `deploy/slurm/mem0_qualification.sbatch`
- Create: `.env.example`
- Create: `tests/qualification/test_deploy_assets.py`

- [ ] **Step 1: Write static deployment tests**

Assert:

- worker, Qdrant, embedding TEI, and reranker TEI services exist;
- Qdrant and model services are not exposed publicly by default;
- GPU 0/1 placement is explicit;
- `/data/lhmsb` is the shared default root;
- image tags require digest variables;
- health checks exist;
- Slurm commands invoke the same CLI/config and do not alter semantics;
- only declared provider destinations may be configured.

- [ ] **Step 2: Run the tests**

```bash
uv run pytest tests/qualification/test_deploy_assets.py -q
```

Expected: missing-file failures.

- [ ] **Step 3: Implement the worker image**

Install from the lock/wheelhouse, use an unprivileged user, disable Mem0
telemetry, and make `python -m lhmsb.qualification` the worker entry point.

- [ ] **Step 4: Implement Compose and Slurm wrappers**

Compose allocates two A100-class GPUs when available. Slurm defaults to
`--gres=gpu:a100:2` and accepts an override without changing run config.

- [ ] **Step 5: Validate syntax**

```bash
docker compose -f deploy/compose.mem0.yaml config
uv run pytest tests/qualification/test_deploy_assets.py -q
```

Expected: pass; if Docker is unavailable locally, the pytest static validation
still passes and records Docker validation as a server gate.

- [ ] **Step 6: Commit**

```bash
git add docker deploy .env.example tests/qualification/test_deploy_assets.py
git commit -m "ops: add Mem0 Docker and Slurm deployment"
```

## Task 15: Build Bootstrap and Offline Dependency Bundle Workflows

**Files:**

- Create: `scripts/bootstrap_server.sh`
- Create: `scripts/build_offline_bundle.sh`
- Create: `scripts/preflight_mem0.sh`
- Create: `scripts/run_mem0_smoke.sh`
- Create: `scripts/run_mem0_qualification.sh`
- Create: `tests/qualification/test_scripts.py`

- [ ] **Step 1: Write static and dry-run script tests**

Test shell strict mode, argument validation, path quoting, no embedded secrets,
idempotent directory creation, exact archive verification, image/model/wheel
manifests, and matching Compose/Slurm worker commands.

- [ ] **Step 2: Run the tests**

```bash
uv run pytest tests/qualification/test_scripts.py -q
```

Expected: missing-file failures.

- [ ] **Step 3: Implement online bootstrap**

Create the data-root tree, verify releases, install the locked environment,
download exact model revisions, record every file SHA-256, pull images and
record digests, then run repository-only preflight.

- [ ] **Step 4: Implement offline bundle**

Bundle:

- source archive/commit manifest;
- wheelhouse and Mem0 wheel;
- OCI image archives/digests;
- BGE model snapshots/file hashes;
- both dataset releases;
- configuration/lock hashes;
- signed or SHA-256 bundle manifest.

Provider credentials are never bundled.

- [ ] **Step 5: Run script checks**

```bash
shellcheck scripts/*.sh
uv run pytest tests/qualification/test_scripts.py -q
```

Expected: pass. If `shellcheck` is absent, record it as an environment
prerequisite and run `bash -n scripts/*.sh`.

- [ ] **Step 6: Commit**

```bash
git add scripts tests/qualification/test_scripts.py
git commit -m "ops: package Mem0 server bootstrap workflows"
```

## Task 16: Add End-to-End Mock Qualification and Migration Documentation

**Files:**

- Create: `tests/qualification/test_mem0_vertical_slice.py`
- Create: `docs/mem0-server-workflow.md`
- Modify: `README.md`

- [ ] **Step 1: Write the end-to-end acceptance test**

Generate/freeze/load a 4-session v0.2 fixture, execute the 12-task matrix with
fakes, produce 15 scored results, reconstruct at least one complete
`write → inventory → candidate → retrieved → visible → causal use → behavior`
chain, aggregate metrics, resume without rerunning valid cells, and validate
every artifact.

- [ ] **Step 2: Run the acceptance test and fix only root causes**

```bash
uv run pytest tests/qualification/test_mem0_vertical_slice.py -q
```

Expected: pass after integration.

- [ ] **Step 3: Document the exact migration workflow**

Include:

```text
local verification
bundle build
rsync/scp transfer
server unpack
credential setup
Compose preflight
Slurm preflight
4-session smoke
16-session qualification
resume
aggregate
validate
copy results back
```

List every required environment variable, expected directory, command, output,
and failure location.

- [ ] **Step 4: Run the complete quality gate**

```bash
uv run pytest -q -k 'not test_resource_module_is_linux'
uv run ruff check src tests
uv run mypy src/lhmsb
uv run python -m lhmsb.datasets verify-mem0-stateful --frozen runs/vertical/software_mem0_v2
uv run python -m lhmsb.datasets regen-check-mem0-stateful --frozen runs/vertical/software_mem0_v2
uv run python -m lhmsb.qualification preflight --repository-only --json runs/preflight-local.json
```

Expected:

- all platform-neutral tests pass locally;
- the Linux-only resource test is run without exclusion in the worker image;
- ruff and mypy pass;
- both dataset release checks pass;
- repository-only preflight returns success.

- [ ] **Step 5: Run Linux container verification**

```bash
docker build -f docker/mem0-worker.Dockerfile -t lhmsb-mem0-worker:test .
docker run --rm lhmsb-mem0-worker:test uv run pytest -q
```

Expected: all tests pass, including `test_resource_module_is_linux`.

- [ ] **Step 6: Scan for incomplete implementation**

```bash
rg -n 'TODO|FIXME|TBD|NotImplemented|placeholder|dummy' \
  src/lhmsb/qualification src/lhmsb/adapters/mem0_qualification.py \
  src/lhmsb/families/software/mem0_vertical.py deploy scripts configs \
  docs/mem0-server-workflow.md
```

Expected: no implementation placeholders.

- [ ] **Step 7: Commit**

```bash
git add tests/qualification/test_mem0_vertical_slice.py docs/mem0-server-workflow.md README.md
git commit -m "docs: finalize Mem0 server qualification workflow"
```

## Task 17: Final Review, Integration, and Server Handoff

- [ ] **Step 1: Verify design-to-implementation coverage**

Check every requirement in
`docs/superpowers/specs/2026-07-16-mem0-server-qualification-design.md` against
code, tests, configs, traces, deployment, and documentation. Record any
deliberate deviation in the design document before completion.

- [ ] **Step 2: Review the diff**

```bash
git diff 0b6ec36...HEAD --stat
git diff 0b6ec36...HEAD --check
git log --oneline --decorate 0b6ec36..HEAD
```

Expected: no whitespace errors; only Mem0 qualification scope.

- [ ] **Step 3: Run final verification from a clean checkout**

Clone or create a clean worktree at `HEAD`, install from `uv.lock`, and repeat
the platform-neutral full gate. Confirm no generated output outside declared
ignored run directories is required.

- [ ] **Step 4: Merge and push only after verification**

Use the repository's normal integration path. Preserve unrelated changes in the
original working tree.

- [ ] **Step 5: Deliver the server handoff**

Report:

- commit SHA and branch;
- release archive hashes;
- bundle command and resulting bundle hash;
- exact server bootstrap/preflight/smoke/full commands;
- local verification counts;
- gates that can only run with A100/Docker/provider credentials;
- expected scorecard and trace paths.

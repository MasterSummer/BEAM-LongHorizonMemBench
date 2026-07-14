# Extending LongHorizonMemSysBench

This document covers how to add new memory adapters and task families to LHMSB, documents the deferred dimensions as extension points, and explains the reproducibility contract.

## Adding a New Memory Adapter

A memory adapter is the single integration point between the benchmark harness and a memory system under test. Adding one involves four steps.

### 1. Implement the Adapter

Create a new module in `src/lhmsb/adapters/` (e.g., `src/lhmsb/adapters/my_backend.py`). Subclass `MemorySystemAdapter` from `src/lhmsb/adapters/base.py` and implement all six required methods:

- `initialize(user_id, session_id=None, **config)` -- set up the backend for a user.
- `reset(user_id)` -- delete all memory for a user (idempotent, between episodes).
- `add_memory(content, *, user_id, session_id=None, metadata=None)` -- ingest content, return a unique `memory_id`.
- `search(query, *, user_id, session_id=None, top_k=10, **filters)` -- retrieve relevance-ranked memories.
- `update_memory(memory_id, *, content=None, metadata=None)` -- update an existing entry.
- `delete_memory(memory_id)` -- remove an entry from future search results.

All six methods have exact signatures documented in `spec/05-systems.md` section 1.1. The base class in `src/lhmsb/adapters/base.py` defines them as abstract methods -- match those signatures exactly.

### 2. Instrument Internal Costs

If your adapter calls an LLM or embedding model internally (for extraction, summarization, enrichment, etc.), those tokens MUST be counted for honest Memory ROI. Wrap internal calls inside `memory_scope()`:

```python
with self.cost_meter.memory_scope():
    self.cost_meter.add_memory_internal_tokens(in_tokens, out_tokens)
    self.cost_meter.add_embedding(embedding_tokens, num_calls)
```

Use the scope-INDEPENDENT direct accumulators (`add_memory_internal_tokens`, `add_embedding`, `add_reflection_tokens`, `add_storage_bytes`, `record_latency`, `incr_retrieval`) for backend calls whose token counts you derive from content (e.g., a content-derived proxy when the backend does not expose usage). This keeps `strict_instrumentation=True` clean -- the scope-aware `record_llm_call` is what raises when unscoped, not the direct accumulators.

See existing adapters for patterns: `src/lhmsb/adapters/chroma.py` (offline embedding counting), `src/lhmsb/adapters/mem0_adapter.py` (content-derived proxy, internal LLM not observable), `src/lhmsb/adapters/cognee_adapter.py` (dual `mem_internal_*` / `reflection_tokens` split).

### 3. Pass the Contract Suite

Every adapter must pass the generic contract suite in `tests/contract/adapter_contract.py`. The suite verifies:

- `add_memory` returns a stable, globally-unique `memory_id`.
- `search` returns the stored entry within the requested `top_k`.
- `update_memory` changes content while keeping the same `memory_id`.
- `delete_memory` removes the entry from search results.
- `reset` clears all stored memories.

Add a test file at `tests/contract/test_my_backend.py` that subclasses `AdapterContractTests` from `tests/contract/__init__.py` and provides an `adapter_factory()` static method. The `no_memory` control is EXEMPT from round-trip checks (it deliberately stores nothing); all storing backends must pass the full suite.

### 4. Register in the Runner

Add your condition name and import to `src/lhmsb/runner/adapters.py`:

1. Add a constant (e.g., `MY_BACKEND = "my_backend"`) near the existing condition-name constants.
2. Add the name to `LEADERBOARD_CONDITIONS` (or `SENSITIVITY_CONDITIONS` for calibration-only).
3. Add a branch in `build_adapter()` that lazily imports your adapter module and returns an instance wired with the per-cell `cost_meter`.

The adapter must be importable lazily (only inside `build_adapter` or the adapter's own `initialize`), so a missing optional dependency fails only that cell, not the whole matrix. Do NOT re-export the adapter class from `src/lhmsb/adapters/__init__.py`.

## Adding a New Task Family

Task families are the benchmark's content generators. Each family produces procedurally generated episodes with a fixed, seed-derived world schedule (inject/change/retract events) and aligned probes. Two families ship in v1: `research` (autonomous research over a synthetic evidence world) and `software` (evolving spec with sandboxed `pytest` grading).

### 1. Implement the Generator

Create a new subpackage in `src/lhmsb/families/` (e.g., `src/lhmsb/families/my_family/`). Implement a generator class that produces `FamilyContent` (from `src/lhmsb/sim/core.py`) -- a list of `WorldEvent`s and `ProbeSpec`s. Follow the patterns in `src/lhmsb/families/research/generator.py` and `src/lhmsb/families/software/generator.py`:

- **Seeded determinism**: use `seeded_rng(seed)` from `lhmsb.types` for all randomness.
- **Cascade semantics**: if retracting a parent fact cascades to dependent facts, emit multiple `retract` events at the same `step` so the cascade is observable as a group drop in `WorldState.valid_facts_at`.
- **Session tagging**: stamp `"session": <int>` into every event payload so the harness can assign session boundaries.
- **Cross-session probing**: set `ProbeSpec.cross_session = True` for probes whose correct answer depends on facts from an earlier session.

### 2. Implement the Checker

Create a checker that grades agent answers against the world state. Subclass `Checker` (a `Protocol` from `src/lhmsb/sim/core.py`) and implement `check(probe, answer) -> CheckResult`. The `CheckResult` carries a `score` (0.0 to 1.0), `is_correct` (boolean), `drift_flags` (for the drift metric), and `metadata` (for downstream consumers).

Follow the patterns in `src/lhmsb/families/research/checker.py` (factual/behavioral/synthesis probes) and `src/lhmsb/families/software/checker.py` (sandboxed `pytest` grading).

### 3. Register in Dataset Pipeline and Pilot Config

- Add your family name to `_FAMILIES` and scale-override keys to the per-family tuples in `src/lhmsb/datasets/pipeline.py`.
- Update the dispatch in `_resolve()` to call your generator.
- Add the family to the `families:` list in `configs/pilot.yaml`.

The dataset CLI (`python -m lhmsb.datasets`) automatically discovers new families through the pipeline registry. The pilot reads the families list from its YAML config.

## Deferred Dimensions (v2 Extension Points)

The following dimensions are documented as extension points and are **not implemented in v1**. They do not appear in the v1 scorecard and do not contribute to the Memory ROI headline. When a future version implements a dimension, its metric is added as an independent row in the scorecard.

| Dimension | v1 Status | Extension Point |
|-----------|-----------|-----------------|
| **Dim 1: Memory Evolution** | Deferred to v2 | `ReflectionCapability` mixin and `apply_decay` hooks exist in `src/lhmsb/adapters/base.py`. An evolution scoring module can be added independently. |
| **Dim 5: Temporal Reasoning** | Deferred to v2 | Probe types for explicit temporal queries exist in the types contract. Saturated in prior art; v2 may add as a secondary axis. |
| **Dim 6: Robustness** | Deferred to v2 | Adversarial probe generation hooks are stubbed in `src/lhmsb/sim/core.py`. A robustness scoring module can be added. |
| **Dim 8: Scalability** | Deferred to v2 | Dataset size is a config parameter in `src/lhmsb/datasets/pipeline.py` (via `n_episodes`, `seeds`, and `ScaleParams`). Scaling experiments are a separate v2 study. |
| **Dim 9: Abstraction** | Deferred to v2 | Conceptual-rollup probe types are defined in the schema (`ProbeSpec.derivation` and `Probe.kind`). A scoring module is deferred. |

Extension points in the codebase:

- **Capability mixins**: `ReflectionCapability`, `ForgettingCapability`, and `SessionCapability` in `src/lhmsb/adapters/base.py` are optional mixins that backends opt into. They are the hooks for Dim 1 (Memory Evolution) and session-aware scalability.
- **Dataset scale parameters**: `ScaleParams` (fact count bounds) in `src/lhmsb/sim/core.py` and per-family scale overrides (`min_events`, `max_sessions`, etc.) in `src/lhmsb/datasets/pipeline.py` are the hooks for Dim 8 (Scalability).
- **Adversarial probe generation**: the `ProbeSpec` structure in `src/lhmsb/sim/core.py` and the `Checker` protocol support adversarial probe kinds. A robustness module can inject adversarial probes without architectural changes.
- **Temporal query probes**: `Probe.kind` already supports `"factual"`, `"synthesis"`, and `"behavioral"`; a `"temporal"` kind (Dim 5) can be added without schema changes.

See `spec/01-overview.md` section "Extension Points (Deferred Dimensions)" for the formal deferral table.

## Reproducibility

LHMSB v1 guarantees reproducibility through three mechanisms:

1. **Frozen datasets**: every dataset is checksummed per file in `MANIFEST.json`. Run `python -m lhmsb.datasets verify --frozen <path>` to confirm integrity. The generation config (seeds, scale, generator version) is recorded so the dataset is regeneratable from seeds alone.

2. **Seeded regeneration**: `python -m lhmsb.datasets regen-check --frozen <path>` regenerates each episode from its stored seed and asserts identical `world_event_hash` and `episode_hash`, proving the recipe is reproducible without the frozen files.

3. **Pinned judge**: the sparse LLM judge is pinned as `lordx64/Qwable-v1` by revision hash in `configs/pilot.yaml`. The hash is recorded in `run_manifest.json`. The smoke run uses a deterministic `StubJudge` (token-level Jaccard) so no live model is needed for reproducibility checks.

The smoke pilot injects a zero-clock so latency-derived cost fields (`retrieval_latency_ms`, `write_latency_ms`, `update_latency_ms`) are always zero, and a deterministic stub agent so agent tokens are identical across runs. Two smoke runs to different output directories produce byte-identical `scorecard.json`.

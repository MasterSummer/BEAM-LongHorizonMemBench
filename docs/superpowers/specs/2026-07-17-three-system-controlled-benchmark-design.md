# Three-System Controlled Long-Horizon Benchmark Design

**Date:** 2026-07-17

**Status:** Approved for autonomous implementation

**Scope:** Server-ready controlled track for MemOS, A-MEM, Mem0, and four controls

## 1. Objective

Build one reproducible, trace-complete benchmark matrix that can be bootstrapped on
an A100 server and directly run without editing source code. The headline matrix is:

1. `workspace_only`
2. `full_context`
3. `oracle_current_state`
4. `flat_retrieval`
5. `mem0`
6. `amem`
7. `memos`

All seven conditions receive the identical current workspace and continuation
request. They differ only in the additional historical information supplied to the
policy model.

This release implements the controlled track. A future native-profile track may use
each system's recommended model and retrieval stack, but native results must not be
mixed into the controlled leaderboard.

## 2. Experimental interpretation

| Condition | Role | Additional policy-visible information |
|---|---|---|
| `workspace_only` | lower-bound control | none |
| `full_context` | context control | every prior public trajectory unit |
| `oracle_current_state` | upper-bound control | minimal evaluator-derived current state, without gold IDs |
| `flat_retrieval` | retrieval baseline | top-k raw trajectory objects after common retrieval/reranking |
| `mem0` | memory system | Mem0-managed memory objects |
| `amem` | memory system | A-MEM structured notes and evolved links |
| `memos` | memory system | MemOS-Tree graph-organized memory objects |

`workspace_only` does not mean that all persistence is removed. The benchmark's
workspace is deliberately persistent. This condition measures the marginal value of
an external long-term memory system beyond that workspace.

`full_context` is not an oracle. It exposes the complete public past, including stale
and conflicting information, and therefore tests whether raw context alone is enough.

## 3. Frozen Flat Retrieval definition

Flat retrieval stores deterministic raw trajectory units and nothing else.

- Each public observation and tool result is one object.
- Future task families may also expose canonical public action units; evaluator-only
  actions are never eligible.
- Object IDs are deterministic hashes of episode, session, kind, ordinal, and content.
- The object text is stored unchanged.
- There is no LLM extraction, summarization, merge, update, deletion, linking, or
  consolidation.
- BGE-M3 produces embeddings; Qdrant performs candidate retrieval; the common BGE
  reranker produces the model-visible top-k.
- Duplicate text from distinct public events remains distinct because the events have
  different provenance.
- `candidate_k=20` and `visible_k=5` are shared with the three memory systems.

Flat retrieval therefore answers whether memory lifecycle management provides value
beyond ordinary semantic retrieval over the raw history.

## 4. Controlled variables

### 4.1 Continuation policy models

The three policy models remain:

- `claude-opus-4-8` through OpenCode Zen Messages;
- `deepseek-v4-pro` through the official DeepSeek Chat Completions endpoint;
- `gpt-5.6-sol` through OpenCode Zen Responses.

The action prompt, current workspace, option set, temperature policy, maximum output,
and structured-output contract are identical across conditions.

The schema-v2 sampling contract freezes `temperature=0`,
`max_output_tokens=512`, `baseline_repeats=2`,
`intervention_repeats=2`, the profile-specific single format-repair allowance,
and an explicit null provider seed where a common seed is unsupported. These
values participate in run identity.

### 4.2 Memory writer

All three managed systems use one fixed DeepSeek writer profile. The tested policy
model does not write its own prefix memory in the controlled track. This prevents
different policy models from receiving different memory stores because of writer
randomness or provider-specific structured-output behavior.

The systems still write memory through their native algorithms:

- Mem0 performs its normal extraction and ADD/UPDATE/DELETE decision process;
- A-MEM stores the public transcript as its native note content and runs the pinned
  implementation's related-note linking/evolution path;
- MemOS uses its tree-memory reader, graph organization, and graph-expanded retrieval.

The benchmark supplies only the public session transcript. It never writes an ideal
summary or evaluator state into a system.

### 4.3 Retrieval models

All controlled profiles use the same local BGE-M3 embedding service. Native candidate
selection remains part of the tested system. For diagnostic comparability, every
managed system additionally exposes a `common_rerank` readout over its frozen native
candidate set using BGE-Reranker-v2-M3.

Headline system scores use the native readout. Common-rerank scores are diagnostic and
must be labeled separately. Flat retrieval has only the common-rerank readout.

Memory-object count is the primary scale variable. Character count, provider tokens,
latency, and storage bytes are auxiliary costs and never replace object count.

## 5. Two-stage execution architecture

Memory construction and policy evaluation are separated.

### Stage A: prefix preparation

For each episode, exactly four preparation tasks run:

```text
flat_retrieval
mem0
amem
memos
```

Each preparation task replays the public sessions in order. At every session boundary
it records native writes and a complete inventory. At every SCEU checkpoint it records
the native candidate set and the common-reranked order. The output is an immutable,
hash-addressed `MemoryPrefixArtifact`.

The same prefix artifact is reused by all three continuation policy models. A policy
evaluation never calls the memory writer or mutates the prepared store.

At every SCEU, the eligible prefix contains exactly public sessions
`0 .. checkpoint_session - 1`. Retrieval is performed before the checkpoint's current
surface could be written. This ordering is shared by Flat, Mem0, A-MEM, and MemOS-Tree
and is checked at session/write boundaries.

### Stage B: continuation evaluation

The core worker evaluates the three policies over the seven headline conditions. For
one episode this produces 21 condition tasks. The three managed systems each have a
native and common-rerank result, producing 30 scored cells in total:

```text
3 policies × (3 controls + 1 flat + 3 systems × 2 readouts) = 30 cells
```

Evaluation task identity includes the exact prefix-artifact hash. Changing a writer,
source commit, embedding model, system configuration, or prepared trace necessarily
changes the run identity.

Planning therefore has two explicit schema types. Initial planning emits stable
`EvaluationTaskTemplate` rows without artifact hashes. After all four required
preparations are verified, `finalize-evaluation-plan` materializes immutable
`EvaluationTask` rows in the same stable index order and binds each retrieval
condition to its exact artifact hash. Templates are never executable.

## 6. Public context construction

### 6.1 Workspace-only

The policy sees the current `SessionSurface`, current workspace snapshot, continuation
request, and opaque action options. No prior session transcript or external memory is
included.

### 6.2 Full-context

The policy sees the canonical public transcripts for sessions
`0 .. checkpoint_session - 1`, followed by the same current surface used by every
condition. The prefix contains observations and tool results only. It excludes:

- future sessions;
- evaluator state IDs or validity labels;
- hidden checker results;
- prior probe answers generated by a tested policy;
- evaluator-only action mappings.

The full prefix is never silently truncated. Planning/preflight fails if a frozen
episode exceeds the configured `full_context_max_chars=100000` limit. The current
16-session Software exemplar is far below that gate.

### 6.3 Oracle-current-state

The oracle block serializes the minimal current state values and dependency closure
needed at the checkpoint. It strips state IDs, gold labels, invalidation annotations,
future validity windows, and valid-action identities.

## 7. Generic memory contract

Mem0-owned trace DTOs move behind a backend-neutral contract while retaining backward
compatible re-exports for the existing Mem0 qualification.

```python
MemoryMutationEvent
MemoryObject
InventorySnapshot
RetrievalCandidate
CandidateSearch
ProviderUsageEvent
WriteSessionResult
MemoryPrefixCheckpoint
MemoryPrefixArtifact
MemoryRuntime
```

`MemoryRuntime` exposes:

```python
write_session(public_messages, session_index, metadata) -> WriteSessionResult
snapshot_inventory(checkpoint_session) -> InventorySnapshot
search_candidates(query, checkpoint_session) -> CandidateSearch
close() -> None
```

Normalized records preserve native IDs and native event names in addition to common
ADD/UPDATE/DELETE/OBSERVED classifications. An adapter may add provenance metadata but
may not rewrite memory content.

## 8. System profiles

### 8.1 Mem0

- Package: `mem0ai==2.0.12`, existing verified wheel.
- Storage: isolated Qdrant collection plus isolated SQLite history.
- Writer: fixed DeepSeek controlled profile.
- Embedding: common BGE-M3 TEI endpoint.
- Candidate retrieval: Mem0 search without its optional reranker.
- Existing history, inventory-delta, usage, and Responses bridge instrumentation remain
  available and backward compatible.

### 8.2 A-MEM

- Official source: `agiresearch/A-mem`.
- Pinned commit: `ceffb860f0712bbae97b184d440df62bc910ca8d`.
- Package identity: official `agentic-memory` source, not the unrelated PyPI `a-mem`
  package.
- Native object: one `MemoryNote` with its UUID, content, context, keywords, tags,
  category, timestamp, and links.
- Native operations: `add_note`, `read`, `search_agentic`, `update`, and `delete`.
- Embedding: the A-MEM vector boundary is adapted to the common BGE-M3 service without
  changing note generation or linking logic.
- Writer transport: a controlled DeepSeek bridge converts the official JSON-schema
  request into JSON-object output and performs local schema validation.

The official implementation resets a fixed collection during construction and has no
reliable cross-process namespace recovery. Therefore each A-MEM preparation runs in a
fresh process with its own storage directory, completes the entire episode, and writes
its prefix artifact atomically. Interrupted preparation is rerun from session zero;
the benchmark does not claim resumability that the upstream system does not provide.

The pinned code's `add_note` does not automatically invoke its unused
`analyze_content` helper, despite broader wording in the README. The adapter must not
silently add that call. It also must not repair dangling links, insertion-index neighbor
updates, or differences between in-memory note metadata and Chroma metadata. Those are
measured as link-validity and store-consistency diagnostics.

### 8.3 MemOS

- Official source: `MemTensor/MemOS`.
- Release profile: v2.0.23 commit
  `583b07b998afc4debb6c5078439b0b3896f5b097`.
- Frozen backend name: `memos_tree_controlled`; headline labels use
  `MemOS-Tree`, not an unqualified claim about every MemOS memory mode.
- Native object: a UUID-backed `TextualMemoryItem` graph node; graph edges are recorded
  separately and are not counted as memory objects.
- Reader: official `SimpleStructMemReader` over the public session transcript.
- Organization: `TreeTextMemory` with `reorganize=true`, synchronous add, internet
  retrieval disabled, and an explicit wait for the reorganizer after each session.
- Retrieval: vector/graph-expanded TreeTextMemory search in its frozen fast mode.
- Storage: one fresh Neo4j Community container volume per preparation task, created
  and destroyed by the serial preparation orchestrator.
- Writer: fixed DeepSeek controlled profile.
- Embedding: common BGE-M3 service.

Inventory snapshots include all live graph nodes that can participate in retrieval,
with structural/topic nodes labeled separately from extracted content nodes. Archived
or deleted nodes remain in history but not in `N_live`. Node-content/status changes,
MERGED_TO lineage, and edge additions/removals are derived from before/after graph
snapshots because the public API does not expose a complete mutation event stream.

The lighter `GeneralTextMemory` profile is intentionally rejected for the headline
condition: it is extraction plus vector storage and cannot test the benchmark's central
state-evolution claim. There is no fallback from TreeTextMemory to GeneralTextMemory.

## 9. Trace and causal-use semantics

Every external-memory condition reconstructs:

```text
public session units
→ native write call
→ native mutation events
→ live inventory
→ candidate set
→ selected/retrieved order
→ model-visible objects
→ policy action
→ programmatic checker behavior
```

`used` is never inferred from model self-report. It is classified from repeated
baseline calls plus leave-one-visible-memory-out interventions. For stale conflicting
objects, a replacement intervention substitutes a current-state-supporting candidate
when available.

The evaluator-side attribution layer may map object content to latent state IDs for
measurement, but those IDs never enter a writer, retrieval query, or policy prompt.

## 10. Metrics and outputs

The report keeps all existing trace files and adds backend-neutral grouping.

### Behavior and drift

- mean behavior score and correct rate;
- constraint-loss drift rate;
- current-plan deviation rate;
- local-over-global override rate;
- stale-state action rate.

### Storage and evolution

- `N_write` and `N_live` by checkpoint;
- ADD/UPDATE/DELETE/observed-delta counts;
- current-state coverage in live objects;
- stale-state retention and current/stale coexistence;
- memory count at each continuation.

### Retrieval, visibility, and use

- candidate and visible current-state coverage;
- stale candidate and stale visible exposure;
- candidate shortfall;
- retrieval-to-visible and visible-to-causal-use yield;
- beneficial, harmful, unused, unstable, and indeterminate intervention rates.

### Controlled comparisons

- each condition's gain beyond workspace;
- each condition's oracle-gap closure;
- system gain over flat retrieval;
- system gap to full context;
- native versus common-rerank delta;
- behavior and drift stratified by live memory-object count.

### Cost diagnostics

- writer, policy, embedding, and reranker calls;
- observed provider tokens where available;
- write/search/rerank/policy latency;
- store bytes and visible characters.

Required report artifacts include `metrics.json`, `metrics_by_cell.json`,
`scorecard.csv`, prefix manifests, mutation/inventory/retrieval traces, SCEU results,
interventions, provider usage, and validation output.

## 11. Container and Slurm topology

Dependency isolation is mandatory. The server bootstrap builds and archives four worker
images from the same benchmark commit:

```text
worker-core   controls, flat retrieval, policy evaluation, report/validation
worker-mem0   Mem0 prefix preparation
worker-amem   A-MEM prefix preparation
worker-memos  MemOS prefix preparation
```

Qdrant, BGE-M3, and BGE-Reranker-v2-M3 are shared infrastructure services. MemOS uses
the pinned official Neo4j Python driver against a fresh Neo4j Community service and
fresh volume for each serial preparation; no Enterprise-only multi-database feature
is assumed. Every other preparation uses a unique namespace or local store directory.
Images, per-worker hash-locked wheelhouses, and upstream source archives are pinned,
hashed, saved under the data root, restored on the assigned compute node, and run with
`pull_policy: never`.

The formal Slurm job requests two A100 GPUs. One serves embedding and one serves
reranking. Prefix preparations run serially in the first qualification release to
avoid provider-rate and shared-service races. Policy evaluation tasks may resume from
independently hashed cells.

Only workers have provider egress. Live calls and GPU/Docker acceptance run on the
server; local CI uses fake upstream backends and mocked HTTP transports.

## 12. Preflight and failure policy

Repository-only preflight verifies configuration, frozen dataset hashes, source pins,
container definitions, secret names, and trace contracts without a provider call.

Live server preflight verifies:

1. exact source/package identities for all three systems;
2. exact policy model identities and API surfaces;
3. two distinct A100 devices;
4. BGE embedding dimension and reranker ordering;
5. empty namespace/store at task start;
6. native add, inventory, search, update, and delete where supported;
7. reconstructable native IDs and model-visible blocks;
8. reset/isolation behavior;
9. each restored worker image starts offline and reports the expected benchmark
   commit, upstream package/source identity, CLI entrypoint, and prefix schema;
10. a four-session end-to-end smoke matrix.

Unsupported APIs, source drift, missing scores/IDs, model substitution, silent context
truncation, hidden-gold leakage, or non-empty starting stores are hard failures. A
failed backend produces an explicit failed preparation; it is never replaced by flat
retrieval or another system.

## 13. Backward compatibility

The existing frozen Software v0.2 release, Mem0-only configuration, Mem0 scripts, and
public Python imports remain valid. The multisystem experiment receives a new schema
version, configuration file, Compose file, scripts, run directory, and report manifest.
Old Mem0 results are not silently reinterpreted as the new controlled matrix.

## 14. Server-ready acceptance gate

Implementation is ready to migrate when all of the following are true:

- the 7-condition configuration expands to 21 tasks and 30 scored cells per episode;
- four prefix preparations have deterministic identities and complete normalized
  traces under fake/contract backends;
- Full-context contains every and only prior public trajectory unit;
- Flat retrieval performs no LLM write or lifecycle mutation;
- A-MEM, MemOS, and Mem0 adapters pass their normalized contract suites;
- source pins, image IDs, datasets, models, configuration, and code commit participate
  in the run identity;
- repository-only preflight and four-session dry-run/smoke planning pass locally;
- report aggregation and validation cover every backend and control;
- all existing tests remain green apart from explicitly platform/live-gated tests;
- one documented bootstrap command and one Slurm submission command are sufficient on
  the server.

The real Zen, DeepSeek, Docker, upstream-system, and A100 smoke remains a server
acceptance step and is not claimed from local mocks.

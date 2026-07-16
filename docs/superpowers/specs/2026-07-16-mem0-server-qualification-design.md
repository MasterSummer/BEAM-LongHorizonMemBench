# Mem0-Only Server Qualification Design

**Date:** 2026-07-16
**Status:** Approved design baseline
**Scope:** All prerequisites required to produce a leak-free frozen Software
qualification slice, move it to an A100 server, and run the first real Mem0
qualification. Letta, Graphiti, Hindsight, and MemOS are explicitly deferred.

## 1. Objective

Extend the existing Software vertical architecture from deterministic
`workspace_only`, `oracle_current_state`, and `fake_native` execution to a real,
auditable Mem0 experiment.

The implementation is complete when a clean server checkout can:

1. generate, freeze, and verify a leak-free real-policy dataset release without
   changing the legacy deterministic release;
2. bootstrap the required Docker services and local retrieval models;
3. validate all three policy-model credentials and model capabilities;
4. run Mem0 through its native write and retrieval lifecycle;
5. reconstruct the chain
   `session input → write → live inventory → candidate → visible → intervention → behavior`;
6. resume failed or interrupted tasks without contaminating other cells;
7. export programmatic write, retrieval, use, state-conflict, drift, cost, and
   behavior metrics;
8. refuse to score a run that silently changed a model, component, prompt,
   dataset, or dependency.

This phase does not run a formal large-scale experiment and does not add another
memory system.

### 1.1 Dataset prerequisite discovered during design review

The existing `software-vertical-v0.1.0` release is retained as a regression
fixture, but it is not valid input for a real policy model:

- continuation files contain evaluator state IDs, valid-action IDs, utilities,
  and satisfy/violate annotations;
- semantic action IDs and evaluative comments reveal which branch is safe,
  stale, or disallowed;
- recurring session boilerplate repeats the offline requirement;
- the README repeats "fully offline" even when `C1` is labelled absent from the
  workspace;
- `G0` and `C1` overlap semantically, so `C1` cannot be independently absent.

The deterministic stub did not read these natural-language leaks, which is why
the legacy tests could pass. A real LLM would read them. The Mem0 qualification
therefore uses a new `software-vertical-mem0-v0.2.0` release. Version 0.1.0 and
its hashes remain unchanged.

## 2. Scientific Boundary

The benchmark has two tracks that must never share a leaderboard.

### 2.1 Controlled track

The Controlled track asks:

> With the same frozen task, local embedding model, local reranker, memory-system
> implementation, and read budget, how do the policy model and Mem0 interact
> across writing, retrieval, use, and behavior?

Each policy-model profile drives its own Mem0 extraction model, store,
retrieval, and continuation policy. Stores are never shared across policy
models.

The three policy models are:

| Profile ID | Provider | Pinned API model ID |
|---|---|---|
| `opus_4_8` | Anthropic | `claude-opus-4-8` |
| `deepseek_v4_pro` | DeepSeek | `deepseek-v4-pro` |
| `gpt_5_6_sol` | OpenAI | `gpt-5.6-sol` |

For a Controlled Mem0 cell, the same policy model is also used by Mem0's
internal extraction operation. This is an end-to-end
`policy model × memory system` comparison, not a pure memory-use-only
comparison. Stage-specific metrics localize whether a difference arose during
write, retrieval, visibility, or continuation.

Mem0's official extraction prompts are preserved. No benchmark-authored
extraction, update, consolidation, or reflection prompt replaces them.

Controlled retrieval components are fixed:

| Component | Pinned choice |
|---|---|
| Embedding | `BAAI/bge-m3@5617a9f61b028005a4858fdac845db406aefb181` |
| Embedding dimension | `1024` |
| Common reranker | `BAAI/bge-reranker-v2-m3@953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e` |
| Native candidate budget | `20` |
| Model-visible memory budget | `5` |
| Vector store | Qdrant |

The release manifest locks these revisions plus every downloaded model-file
hash. Model names alone are not sufficient.

### 2.2 Native track

The Native track asks:

> What does the pinned Mem0 OSS library deliver when its recommended library
> defaults are used as a complete memory product?

The external policy still rotates across the three policy models so behavior can
be compared. Mem0's internal components are fixed independently:

| Component | Explicit native configuration |
|---|---|
| Internal LLM | OpenAI `gpt-5-mini` |
| Embedding | OpenAI `text-embedding-3-small` |
| Vector store | Qdrant |
| Reranker | disabled |
| Prompt | Mem0 2.0.12 built-in prompt |

These values are written explicitly into the configuration even though they
match the pinned library defaults. Calling `Memory()` with mutable defaults is
not reproducible enough for a frozen run.

Native and Controlled results are reported separately. Native performance
cannot be attributed to the Mem0 algorithm alone because the internal model and
embedding stack also differ.

The Qdrant provider is unchanged, but the benchmark supplies an isolated
server-backed collection instead of Mem0's shared `/tmp/qdrant` path. This
deployment-only override is declared in the Native manifest; it prevents
cross-task lock contention and does not change Mem0's retrieval implementation.

## 3. Mem0 Version and Compatibility Contract

The first qualification locks:

```text
package: mem0ai==2.0.12
source commit: 42cf18c4e6adb448e981aa1c7b55c1602b0cb670
wheel sha256: 6b7e1afa466f6e14dd34b5e9222c159a69fad38f8d787e73adbf91dbb29e73e2
```

The adapter targets the Mem0 2.0.12 OSS `Memory` API rather than preserving the
repository's older assumed API:

```python
memory.add(
    messages,
    user_id=user_id,
    run_id=run_id,
    metadata=metadata,
    infer=True,
)

memory.search(
    query,
    filters={"user_id": user_id, "run_id": run_id},
    top_k=20,
    threshold=0.0,
    rerank=False,
)

memory.get_all(
    filters={"user_id": user_id, "run_id": run_id},
    top_k=inventory_limit,
)

memory.history(memory_id)
```

The threshold is explicitly zero for evaluator retrieval accounting. The
benchmark, rather than a hidden default threshold, determines the visible
budget.

Mem0 2.0.12's default V3 extraction is additive. It can retain a stale object
after a state transition instead of mutating or deleting it. The benchmark does
not patch that behavior: stale retention is a target phenomenon for state
evolution and behavioral drift.

Mem0 telemetry is disabled in benchmark containers. Benchmark-native traces and
provider usage callbacks replace product telemetry.

Inventory uses `get_all(..., top_k=10000)` and independently checks the
task-local Qdrant collection count. A mismatch or a store larger than the
declared inventory limit is an `inventory_failure`, not a silently truncated
count.

## 4. Experimental Matrix

Each episode expands to twelve independently resumable tasks and fifteen scored
condition results.

### 4.1 Atomic execution tasks

For each of the three policy models:

1. `workspace_only`
2. `oracle_current_state`
3. `mem0_controlled`
4. `mem0_native`

This produces `3 × 4 = 12` atomic tasks.

### 4.2 Scored conditions

One `mem0_controlled` task writes one store and produces two paired readouts:

1. `mem0_controlled_native_readout`
2. `mem0_controlled_common_rerank`

Therefore the scored matrix contains:

```text
3 workspace-only
3 oracle-current-state
3 controlled native-readout
3 controlled common-rerank
3 native Mem0
= 15 condition results per episode
```

The two Controlled readouts branch from the same checkpoint and use the same
candidate set. A continuation probe and its answer are not written into the
future prefix. This prevents one readout from contaminating the other or later
SCEUs.

## 5. Episode and Session Semantics

The qualification dataset reuses the legacy state/event/checker architecture
but freezes new public surfaces under
`software-vertical-mem0-v0.2.0`. The legacy 0.1.0 generator and release remain
replayable.

The qualification template separates the overlapping goal and constraint:

```text
G0: Build a reproducible and auditable experiment pipeline.
C1: Pipeline execution must remain completely offline; do not call cloud services.
C2: The held-out test set must never be modified.
```

All other branch, leakage, update, local-scope, and validation semantics remain
the same. `G0` no longer entails `C1`, so workspace recoverability for `C1` can
be measured independently.

The v0.2 renderer enforces:

- neutral session boilerplate that does not repeat old goals or constraints;
- workspace content consistent with declared `explicit`, `derivable`, and
  `absent` recoverability;
- separate public and evaluator continuation records;
- no evaluator IDs or labels in any agent-readable file;
- no future values in an earlier surface;
- deterministic neutralization and permutation of action options.

For each session:

1. load the frozen public session surface;
2. create a fresh working context containing only the current session surface
   and workspace snapshot;
3. run the policy for the ordinary session interaction, if the surface requires
   a policy response;
4. send the evaluator-blind session transcript to
   `Mem0.add(..., infer=True)`;
5. snapshot the native memory inventory and history delta;
6. discard working context before the next session.

The write transcript contains public observations, public tool results, and
ordinary policy messages. It does not automatically copy the raw workspace
snapshot into Mem0. A workspace artifact enters the write transcript only when
an explicit public tool result shows that the agent read it. This prevents
automatic workspace ingestion from erasing the distinction between workspace
and memory.

The benchmark controller initiates the session-boundary write, while Mem0's
internal LLM decides which facts become memory objects. The write origin is
recorded as `system_managed_extraction`; it must not be described as an
agent-authored memory tool call.

At a continuation opportunity:

1. construct the natural-language continuation query from the public surface;
2. retrieve at most 20 Mem0 objects;
3. save the exact native candidate ordering;
4. derive native top 5 and common-reranked top 5;
5. serialize only the selected branch's five objects into the policy prompt;
6. invoke the policy's structured `submit_action` tool;
7. run the hidden Software checker;
8. run the required causal readout interventions from the same immutable
   checkpoint.

Every condition receives the identical current workspace. `workspace_only`
adds nothing, `oracle_current_state` adds the evaluator's minimal current state,
and a Mem0 condition adds its selected memory readout. Thus Mem0 and oracle
scores measure marginal value beyond workspace rather than replacing it.

### 5.1 Public action options

The latent actions keep their evaluator IDs and checker annotations. The policy
does not receive them. For each opportunity, the public renderer:

1. strips module and function docstrings, comments, utilities, satisfy/violate
   fields, valid-action fields, and evaluative descriptions;
2. presents only neutral candidate labels and the behaviorally meaningful patch
   content;
3. deterministically permutes the candidates from an opportunity-specific
   evaluator seed;
4. stores the opaque-label-to-latent-action mapping only in the evaluator
   record.

For example, the policy may see three unlabeled implementations whose returned
values differ in `version`, `offline`, and `heldout_modified`; it never sees
names such as `safe_v2_offline`, `stale_v1`, or `cloud_shortcut`.

## 6. Information Firewall

The policy model and Mem0 internal model may receive:

- current public observations;
- current public tool results;
- the current workspace snapshot;
- previous public conversation content passed to Mem0 at its native write
  boundary;
- natural-language memory text returned by Mem0.

They must never receive:

- latent state IDs;
- source event IDs;
- validity or supersession labels;
- evaluator dependency graphs;
- future events or future workspace values;
- valid-action IDs;
- action-to-state satisfy/violate annotations;
- hidden checker results;
- evaluator drift labels;
- leave-one-out target labels.

Evaluator metadata may be attached after a run through offline alignment. It
must not be inserted into Mem0 metadata before the continuation is complete.

A surface validator scans both keys and string values. The qualification
release is rejected if any public file contains a gold state ID, latent action
ID, validity label, evaluator field name, or configured answer-revealing phrase.

## 7. Retrieval and Use Semantics

The read lifecycle has four distinct layers:

```text
candidate → retrieved → model-visible → behaviorally-used
```

### 7.1 Candidate

A candidate is an object returned by the Mem0 search call before a benchmark
readout truncates or reorders it. The candidate trace records:

- memory ID;
- original rank;
- Mem0 score and optional score details;
- content hash;
- created and updated timestamps;
- query and query hash.

### 7.2 Retrieved

Retrieved objects are the branch-specific top five:

- native top five for native readout;
- common-reranked top five for reranker ablation;
- native top five for the Native track.

If fewer than 20 candidates exist, all candidates are retained and
`candidate_shortfall` is recorded. No synthetic candidate is added.

### 7.3 Model-visible

An object is model-visible only if its serialized natural-language content is
present in the exact request sent to the policy. The trace stores:

- memory IDs in prompt order;
- serialized block hashes;
- final request hash.

This prevents a successful search call from being mistaken for model exposure.

### 7.4 Programmatic state attribution

Storage and retrieval metrics do not use an LLM judge. The qualification
dataset includes evaluator-only fact signatures generated from the latent state
list. A signature contains:

- required canonical anchors;
- allowed surface variants;
- polarity and negation rules;
- version, scope, and authority predicates;
- the sessions and source events in which the fact could have been observed.

An extracted memory is attributed in ordered tiers:

1. `exact_signature`: its normalized text satisfies one state signature and no
   conflicting signature;
2. `unique_provenance`: the memory is a new inventory delta from a write whose
   public source introduced exactly one eligible state, its text matches at
   least one positive anchor for that state, and it does not match another
   state;
3. `ambiguous`: zero, multiple, or contradictory states remain possible.

Only the first two tiers contribute positive state coverage. Ambiguous objects
remain in `N_live` and count as unaligned storage, so uncertainty cannot improve
precision. Embedding similarity may be recorded as a diagnostic but cannot
decide the gold attribution.

The attribution process runs after the policy continuation and never writes
state IDs back into Mem0.

### 7.5 Behaviorally used

Textual mentions and chain-of-thought are not evidence of use. Use is inferred
by targeted counterfactual continuations.

For each evaluator-eligible visible memory:

1. run the full-visible continuation twice with byte-identical requests;
2. if the two baseline actions differ, mark `unstable_baseline` and do not make
   a causal-use claim;
3. rerun twice after removing that memory;
4. where a gold stale/conflict counterpart exists, rerun twice after replacing
   it;
5. require each intervention pair to agree before comparing selected action,
   checker score, violated state IDs, and drift flags.

A memory is `behaviorally_used` only when the intervention changes an outcome
consistently and the direction agrees with the state supported or contradicted
by that memory. Otherwise it is `visible_not_causally_used`,
`causal_direction_ambiguous`, or `intervention_unstable`.

Interventions never mutate the underlying Mem0 store. They modify the
model-visible readout only.

## 8. Policy Provider Contract

Provider adapters expose one benchmark-facing interface:

```python
PolicyClient.submit_action(request: PolicyRequest) -> PolicyResponse
```

The response is a structured tool call:

```text
submit_action(
  action_id,
  optional_patch,
  concise_rationale
)
```

The policy receives the neutral, permuted `PublicActionOption` records defined
in Section 5.1. The provider adapter translates the submitted opaque option ID
back to a latent action only after the model response has been persisted.

Every call records:

- provider and exact model ID;
- endpoint identity without credentials;
- request parameters and canonical request hash;
- response and tool-call hashes;
- provider request ID when available;
- input, output, cached, and reasoning tokens when returned;
- start time, end time, latency, retry count, and terminal error class.

No provider fallback is allowed. A structured-output failure permits one
format-repair attempt using the same model. A second failure is a scored model
failure.

Provider-specific unsupported parameters are omitted intentionally and recorded
in the effective request:

- Opus 4.8 uses Anthropic's native Messages API and its pinned default effort
  unless a qualification test proves an explicit supported effort setting.
- DeepSeek V4 Pro uses the official DeepSeek endpoint and its pinned default
  thinking behavior.
- GPT-5.6 Sol uses the OpenAI endpoint with GPT-5 reasoning-compatible
  parameters; unsupported sampling parameters are not sent.

Mem0's internal LLM calls are instrumented at the provider client boundary.
Returned provider usage is authoritative. A proxy token estimate may be emitted
only as a separately named diagnostic when the provider supplies no usage; it
must never overwrite measured cost.

## 9. Local Retrieval Services

Two Text Embeddings Inference services are used in the Controlled track:

```text
GPU 0: BAAI/bge-m3 embedding service
GPU 1: BAAI/bge-reranker-v2-m3 reranker service
```

If only one GPU is available, both services may share it for smoke tests, but
the manifest records the degraded placement. A formal qualification requires
the declared resource profile to match the actual placement.

TEI image tags and OCI digests, model revisions, model-file hashes, dtype,
served model name, device assignment, and startup arguments are pinned.

The embedding service exposes an OpenAI-compatible endpoint. Controlled Mem0
uses Mem0's OpenAI embedder client against that local endpoint because the
Mem0 2.0.12 Hugging Face provider imports the full `sentence-transformers`
runtime even when configured for remote inference. The served model remains
the pinned BGE-M3 revision. The reranker is benchmark-owned and is called
through TEI's rerank endpoint after Mem0 returns the native candidate list.

### 9.1 Network boundary

The task's "offline" constraint describes the software action selected by the
agent. It does not assert that the evaluation harness is physically
air-gapped. Policy inference and Native Mem0 extraction require the three
declared provider APIs.

At runtime:

- the policy sandbox has no arbitrary network tool;
- only benchmark-owned provider clients may make outbound HTTPS calls;
- outbound destinations are allowlisted and recorded;
- Qdrant, embedding, and reranking remain local;
- provider traffic is evaluator infrastructure and is never represented to the
  agent as an allowed project implementation choice.

The offline dependency bundle can install images, wheels, models, and the
dataset without internet. A qualification run still requires provider API
connectivity unless a future, separately named local-model track is introduced.

## 10. Storage Isolation

Every atomic task receives:

- a unique Qdrant collection namespace;
- a unique Mem0 history SQLite file;
- a unique user ID and run ID;
- a unique trace directory;
- a deterministic task ID derived from the immutable run identity.

No process reuses a store created by another policy model, track, episode, or
retry generation. A retry may reopen only its own validated task store.

The server data root is:

```text
${LHMSB_DATA_ROOT:-/data/lhmsb}/
├── datasets/
│   ├── software_v1/
│   └── software_mem0_v2/
├── models/
│   ├── bge-m3/
│   └── bge-reranker-v2-m3/
├── qdrant/                  # task isolation is by collection namespace
├── history/preflight/
├── hf-cache/
├── wheelhouse/
├── images/
├── manifests/
├── runs/
│   ├── preflight/latest.json
│   └── mem0/<run_name>/
│       └── cells/tasks/<task_id>/store/history.sqlite
├── logs/
└── bundles/
```

## 11. Repository Assets

The implementation adds:

```text
deploy/
  compose.mem0.yaml
  slurm/
    mem0_preflight.sbatch
    mem0_qualification.sbatch

configs/
  models/
    claude-opus-4-8.yaml
    deepseek-v4-pro.yaml
    gpt-5.6-sol.yaml
    bge-m3.yaml
    bge-reranker-v2-m3.yaml
  systems/
    mem0/
      controlled.yaml
      native.yaml
  experiments/
    mem0_qualification.yaml
  systems.lock.yaml

constraints/
  mem0.lock.txt

scripts/
  bootstrap_server.sh
  build_offline_bundle.sh
  preflight_mem0.sh

datasets/releases/
  software-vertical-v0.1.0/
    RELEASE.json
    software_v1-6b4edbf.tar.gz
    software_v1-6b4edbf.tar.gz.sha256
  software-vertical-mem0-v0.2.0/
    RELEASE.json
    software_mem0_v2.tar.gz
    software_mem0_v2.tar.gz.sha256

.env.example
```

The existing 0.1.0 release archive is copied from the verified local release
without changing its bytes or SHA-256. The new 0.2.0 release has a distinct
template ID, generator version, plan hash, surface hash, archive hash, and
dataset card documenting the leak fixes.

Deferred systems do not receive deployment services, dependency locks, source
checkouts, or experiment matrix entries. `systems.lock.yaml` may reserve an
extensible schema, but only Mem0 is enabled.

## 12. Trace and Result Schema

Each completed run emits:

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
metrics_by_cell.json
summary.json
scorecard.csv
scorecard.md
```

### 12.1 Memory event

Required fields include:

- run, task, episode, session, and operation IDs;
- operation type: `add`, `update`, `delete`, `none`, or observed inventory delta;
- Mem0 memory ID;
- old and new content hashes;
- native Mem0 event type;
- operation latency;
- provider call IDs attributable to the operation;
- evaluator-side aligned state IDs, added only after execution;
- alignment confidence and method.

### 12.2 Inventory

Each checkpoint records:

- `N_write`: cumulative native write events;
- `N_live`: currently retrievable native memory objects;
- every live memory ID and content hash;
- history length;
- created and updated timestamps;
- task-local store hash.

### 12.3 Retrieval

Each branch records:

- query and query hash;
- candidate IDs and native ranks;
- native and reranker scores;
- branch-specific retrieved IDs;
- model-visible IDs;
- candidate and visible counts;
- candidate shortfall;
- request hash;
- retrieval and reranking latency.

### 12.4 Behavior

Each SCEU records:

- selected action and optional patch hash;
- checker score and correctness;
- passed and failed hidden tests;
- violated state IDs;
- stale-state, constraint-loss, plan-deviation, and local-over-global flags;
- full-visible and intervention outcomes;
- behaviorally used IDs and ambiguity labels.

## 13. Metrics

### 13.1 Write and state maintenance

- write coverage;
- write selectivity;
- current-state storage precision, recall, and F1;
- stale-state retention rate;
- duplicate live-memory rate;
- update/delete responsiveness;
- write-to-continuation alignment;
- `N_write` and checkpoint-level `N_live`.

### 13.2 Retrieval and visibility

- candidate recall;
- retrieval precision, recall, F1, and false-positive rate;
- retrieval timeliness;
- candidate shortfall rate;
- visible sufficiency;
- visible contamination;
- stale retrieval rate;
- native-versus-common-rerank delta.

### 13.3 Causal use

- causal memory-use rate;
- retrieved-but-not-visible rate;
- visible-but-not-causally-used rate;
- beneficial, harmful, and ambiguous intervention rates;
- leave-one-memory-out action flip rate.

### 13.4 State evolution and drift

- state-conflict resolution accuracy;
- stale-state action rate;
- still-active constraint-loss rate;
- current-plan deviation rate;
- local-subgoal-over-global-goal rate;
- matched early-versus-late behavioral decay;
- aggregate drift index with component counts reported separately.

The conflict-resolution denominator includes scope conflicts, valid updates,
and matched-branch checkpoints only after an invalidated alternative exists.
The pre-update member of an early/late matched pair is a baseline, not a state
conflict.

### 13.5 Behavior and baselines

- programmatic behavior score;
- workspace-only score;
- oracle-current-state score;
- controlled native-order, controlled common-rerank, and native-track Mem0
  gain beyond workspace, reported separately;
- fraction of oracle gap closed for the same three cells;
- a clearly labeled macro-average across those three cells for overview only;
- common-rerank behavior delta.

### 13.6 Cost and reliability

- agent and memory-internal provider usage;
- embedding calls and processed inputs;
- reranker calls and candidate pairs;
- read/write/rerank/policy latency;
- retry and terminal-failure rates;
- Qdrant compressed collection-snapshot bytes and closed SQLite
  main/WAL/SHM bytes.

Token counts are cost diagnostics, not the RQ5 scale variable. RQ5 uses native
memory-object counts.

## 14. Run Identity and Reproducibility

Run identity hashes:

- code commit and dirty state;
- dataset manifest and archive hash;
- experiment config hash;
- resolved persistent data-root path;
- Mem0 package, source commit, and wheel hash;
- Python lock hash;
- Docker image digests;
- embedding and reranker revisions and file hashes;
- provider model IDs and effective request profiles;
- built-in Mem0 prompt hashes;
- candidate and visible budgets;
- retry and timeout policy;
- hardware and CUDA runtime snapshot.

Secrets are excluded and redacted. The manifest stores only the environment
variable names required.

An existing run directory may be resumed only if every identity component still
matches. Otherwise a new run identity is required.

## 15. Failure Handling

The runner classifies failures as:

- `preflight_failure`;
- `provider_auth_failure`;
- `provider_model_unavailable`;
- `provider_rate_limit`;
- `provider_timeout`;
- `structured_output_failure`;
- `mem0_write_failure`;
- `mem0_search_failure`;
- `inventory_failure`;
- `embedding_failure`;
- `reranker_failure`;
- `vector_store_failure`;
- `surface_leak`;
- `checker_failure`;
- `trace_incomplete`;
- `identity_mismatch`;
- `resource_measurement_failure`;
- `resource_cleanup_failure`.

Retries are bounded and recorded. No failure path may:

- switch provider or model;
- switch from Controlled to Native components;
- replace BGE-M3 with another embedder;
- skip the common reranker and label native order as reranked;
- estimate an unavailable trace field and label it observed;
- continue scoring after an information-firewall violation.

Reranker failure leaves the native branch valid but marks the paired reranker
branch failed. A trace-incomplete Mem0 task is not scoreable even if its final
action is correct.

## 16. Preflight and Qualification Gates

The server preflight must pass, in order:

1. verify repository commit policy and configuration schema;
2. verify the legacy 0.1.0 archive remains byte-identical;
3. regenerate the 0.2.0 plan and public surfaces from seed and match all frozen
   hashes;
4. run the public-surface leak scan and semantic workspace-recoverability audit;
5. verify the 0.2.0 archive SHA-256 and unpacked manifest;
6. verify Python lock and Mem0 wheel hash;
7. verify Docker, Compose, NVIDIA container runtime, GPU count, driver, disk,
   memory, and writable data root;
8. verify all OCI image digests;
9. verify local model revisions and every required model-file hash;
10. start Qdrant and pass write/search/list/delete/reset isolation checks;
11. start TEI embedding and require exactly 1024 output dimensions;
12. start TEI reranker and verify deterministic ordering on a fixed fixture;
13. verify all three provider credentials and exact model availability;
14. require a structured `submit_action` smoke response from each policy model;
15. require Mem0 extraction add/search/get_all/history for each Controlled
    internal model;
16. require the explicit Native Mem0 profile;
17. verify internal provider usage tracing and prompt hashing;
18. execute a four-session smoke episode;
19. execute the frozen sixteen-session qualification episode;
20. reconstruct at least one complete stored-to-behavior causal chain;
21. aggregate all required artifacts and validate their schemas and hashes.

The gate stops at the first failure and writes a machine-readable report. No
formal pilot is launched automatically.

## 17. Testing Strategy

### 17.1 Unit tests

- configuration parsing, duplicate-key rejection, and secret redaction;
- exact Mem0 2.0.12 request shapes;
- response normalization and native event parsing;
- inventory/history deltas;
- candidate/native/common-rerank branching;
- neutral action rendering and deterministic permutation;
- model-visible request hashing;
- fact-signature attribution and causal-use stability classification;
- metric formulas;
- run identity and resume validation.

### 17.2 Contract tests

- fake Mem0 backend implementing the 2.0.12 surface;
- fake policy providers with usage and failure cases;
- fake TEI embedding and rerank endpoints;
- no evaluator metadata crossing the information firewall;
- no semantic answer leakage through IDs, comments, boilerplate, or workspace
  content;
- no cross-task Qdrant or history contamination.

### 17.3 Live tests

Live tests are opt-in and separately gated:

- local Docker Qdrant;
- local TEI embedding;
- local TEI reranker;
- real Mem0 with one provider at a time;
- three-provider smoke qualification when all credentials are present.

### 17.4 End-to-end acceptance

A clean test environment must demonstrate:

- old v1 and existing vertical tests still pass;
- the v0.1.0 release remains byte-identical and reproducible;
- the v0.2.0 release is leak-free and its recoverability labels agree with what
  a policy can actually read;
- `workspace_only`, `oracle_current_state`, and real Mem0 are distinguishable;
- Controlled native and common-rerank share the exact store and candidate set;
- opaque public action options map back to the correct latent checker action;
- Mem0 writes and retrieves real native objects;
- an important memory intervention changes behavior in the expected direction;
- stale memory can be retained and measured without gold leakage;
- repeated deterministic local components produce stable hashes;
- interrupted tasks resume without rerunning valid completed cells;
- scorecard and trace validators pass.

## 18. Migration and Operator Workflow

Online bootstrap:

```text
git clone / git pull pinned commit
bootstrap data root
download and hash models
pull and record OCI image digests
build wheelhouse
verify dataset release
run preflight
run smoke qualification
run full qualification
aggregate and validate
```

Offline dependency bootstrap:

```text
copy signed bundle
verify bundle manifest
load OCI archives
install from wheelhouse
mount local model snapshots
unpack frozen dataset
run the same preflight and qualification commands
```

After offline dependency bootstrap, the qualification worker still needs
allowlisted HTTPS access to the three policy providers, as defined in
Section 9.1.

Docker Compose and Slurm invoke the same worker command and produce the same
run/task identities. Slurm is orchestration only; it must not change experiment
semantics.

## 19. Deferred Work

The following work is deliberately excluded until Mem0 passes qualification:

- Letta, Graphiti, Hindsight, and MemOS adapters or containers;
- large episode generation;
- large-scale statistics and survival analysis;
- formal paper-result runs;
- cross-system leaderboards;
- native memory-count caps;
- learned memory policies.

The next system is selected only after the Mem0 trace contract, server workflow,
and result schema are proven end to end.

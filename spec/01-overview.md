# 01 — Overview, Scope, Contributions & Glossary

## Problem Statement

Modern AI agents operate over long horizons: a personal assistant tracks user preferences across months, a research agent assimilates and updates a growing body of evidence, a software-development agent maintains and revises a multi-file codebase through evolving requirements. Memory management systems promise to give these agents persistent, evolvable recall beyond their context window. Yet the field lacks a standardized benchmark that answers the central question in deploying such systems: **does the memory system improve task performance enough to justify its full cost?**

Existing benchmarks fragment along partial axes. Retrieval quality (Dim 4) and temporal reasoning (Dim 5) are well-covered but saturated (22 benchmarks surveyed). Token/resource efficiency (Dim 7) and scalability (Dim 8) are commoditizing rapidly in 2026 with metrics like LongMemEval-V2 LAFS, LongMemCode $/query, and Cost-Accuracy-LTM CpSQ/AES/Pareto. No existing benchmark measures (a) whether a memory system stabilizes an agent's goal-directed behavior across long sessions, (b) the full lifecycle cost of using the memory system, not just query cost, or (c) how memory systems perform on procedural agentic tasks where the environment evolves through retractions and shifting requirements.

LongHorizonMemSysBench (LHMSB) v1 closes these gaps by introducing **Memory ROI**, a cross-cutting headline metric that normalizes performance gain by total memory-system cost, measured via a counterfactual replay protocol over two procedurally-generated agentic task families.

## v1 Scope: The Four-Dimension Core

v1 implements the novel, defensible core of LHMSB across four dimensions, with Memory ROI as the unifying headline:

| Dimension | v1 Status | Description |
|-----------|-----------|-------------|
| **Dim 2 — Goal-Directed Utilization** | **Implemented** | Does the agent use its memory to improve task completion? Scored via programmatic task completion rubrics + sparse judge. |
| **Dim 3 — Goal Drift & Behavioral Stability** | **Implemented** | Does the agent's behavior remain stable across sessions? Measured via programmatic invariants: `drift_index` over aligned probes. |
| **Dim 4 — Retrieval Quality** | **Implemented (supporting)** | Endogenous (agent-initiated) and oracle (fixed-query) retrieval metrics. Supports the other dims but is not the headline. |
| **Dim 7 — Token/Resource Efficiency** | **Implemented** | Full-lifecycle cost instrumentation feeding the cost denominator of Memory ROI. |

**Memory ROI** cross-cuts all four dimensions: `ROI = mean(normalized_gain) / mean(normalized_cost)` reported with bootstrap confidence intervals and Pareto analysis.

### Extension Points (Deferred Dimensions)

The following dimensions are deferred to future versions. This spec documents their extension points so later work can plug in without architectural redesign. They are explicitly NOT part of the v1 evaluation scorecard or headline metric:

| Dimension | Deferred Status | Extension Point |
|-----------|-----------------|-----------------|
| **Dim 1 — Memory Evolution** | Deferred to v2 | Adapter `ReflectionCapability` mixin and `apply_decay` hooks exist; evolution scoring module can be added. |
| **Dim 5 — Temporal Reasoning** | Deferred to v2 | Probe types for explicit temporal queries exist in the types contract. Saturated in prior art — v2 may add as secondary axis. |
| **Dim 6 — Robustness** | Deferred to v2 | Adversarial probe generation hooks are stubbed in the simulator core. |
| **Dim 8 — Scalability** | Deferred to v2 | Dataset size is a config parameter; scaling experiments are a separate v2 study. |
| **Dim 9 — Abstraction** | Deferred to v2 | Conceptual-rollup probe types are defined in the schema; scoring module deferred. |

v1 does NOT claim to be a "complete benchmark for all memory capabilities." It is the four-dimension novel core with a defensible headline metric and a clean architecture for extending to the full nine dimensions later.

## Contributions & Positioning

### Three Novelty Claims

LHMSB v1 makes three claims of novelty against the prior-art landscape (22 benchmarks surveyed; 6 closest competitors cited below):

1. **Full-lifecycle Memory ROI.** All prior efficiency-aware benchmarks define cost narrowly around retrieval/query tokens (LongMemEval-V2 LAFS) or per-query dollar cost (LongMemCode $/query, Cost-Accuracy-LTM's CpSQ). LHMSB is the first to define cost across the memory system's *entire lifecycle*: ingest, index, store, retrieve, update, and reflection. The cost vector captures agent tokens, memory-internal LLM tokens, embedding tokens and calls, storage bytes, retrieval latency, and write/update/reflection latency. The scalarization to tokens-equivalent uses a declared, pinned conversion sheet so the denominator is auditable and reproducible. The numerator normalizes gain against a no-memory counterfactual, producing a single ROI number that answers "did paying for this memory system help, and by how much?"

2. **Standardized cross-system goal drift in agentic tasks.** Existing benchmarks measure retrieval drift (did stored facts shift?) or embedding decay. LHMSB is the first to operationalize *behavioral* goal drift across memory systems in a standardized way, using programmatic invariants: (a) using a retracted or superseded fact, (b) violating a still-active constraint with no superseding event, (c) behavioral flip without a triggering event. The `drift_index` is a weighted violation rate over aligned probe points, identical across all conditions, so drift can be compared directly across memory systems. A sparse LLM judge (`lordx64/Qwable-v1`, pinned by revision hash) handles only the non-programmatically-decidable cases.

3. **Procedural agentic task-completion simulators.** The field's dominant paradigm is conversational QA: an agent answers isolated questions about a synthetic biography or document corpus (AMA-Bench, STALE, MemoryAgentBench). LHMSB introduces *procedural task-completion simulators*, where the agent must complete a goal over multiple sessions in an evolving evidence world. Two families ship in v1:
   - **Research Project**: an autonomous research task over a synthetic evidence world with retractions (facts that are later superseded). The checker maps every claim to a synthetic fact, so grading is programmatic and deterministic.
   - **Software Development**: an evolving specification with hidden requirements + a test suite. The task is a tiny synthetic Python package graded by deterministic `pytest`; no network access or package install during episodes.

   Both families share a common simulator core. The evidence world is *fixed and exogenous*: identical `world_event_hash` per (episode_id, seed) across all conditions. Agent actions do not mutate the world in v1, enabling clean counterfactual comparison.

### Positioning vs Prior Art

The following competitor benchmarks were identified in a 22-benchmark survey. Each is cited (arXiv ID where available) and briefly differentiated. No claim is made that these benchmarks are inadequate; they probe different, often complementary, aspects of memory systems.

| Competitor | arXiv / Ref | What It Measures | LHMSB v1 Differentiation |
|-----------|-------------|-----------------|--------------------------|
| **Cost-Accuracy-LTM** | 2601.07978 | Cost-Accuracy Pareto frontiers (CpSQ/AES/Pareto) for long-term memory retrieval. | Cost-Accuracy-LTM measures retrieval cost; LHMSB measures full-lifecycle cost (ingest + store + update + reflect in addition to retrieval) and normalizes over task-completion gain, not retrieval accuracy. |
| **LongMemCode** | — | Per-query dollar cost for code-memory retrieval. | LongMemCode targets code-retrieval cost per query; LHMSB targets agentic task completion with evolving requirements and a full lifecycle cost model. |
| **YCBench** | — | Conversational QA over structured user contexts, including a business-strategy family. | YCBench's business-strategy family is deliberately excluded from LHMSB's task space to avoid overlap. LHMSB uses procedural simulators, not conversational QA. |
| **AMA-Bench** | — | Conversational QA over a synthetic biography with memory operations. | AMA-Bench evaluates retrieval under single-session operations; LHMSB evaluates multi-session goal completion with retractions and drift measurement. |
| **STALE** | 2605.06527 | Staleness detection in memory-augmented QA. | STALE measures whether stored facts go stale; LHMSB measures whether the agent's *behavior* drifts when facts change, which is a different phenomenon (a fact can be current but the agent can still misapply it). |
| **MemoryAgentBench** | 2507.05257 | Multi-session memory evaluation with conversational tasks. | MemoryAgentBench is the closest in spirit but uses conversational QA tasks; LHMSB uses procedural task-completion simulators with programmatic grading, and adds goal drift + full-lifecycle ROI as headline metrics. |

## Glossary of Canonical Terms

All terms used throughout the LHMSB spec are defined here. Any term appearing in this glossary is used consistently across all spec files (`01-overview.md`, `02-metrics.md`, `03-protocol.md`, `04-datasets.md`, `05-systems.md`). Conversely, any term used in the spec that has a precise technical meaning within LHMSB MUST appear in this glossary.

### Core Entities

**episode**
A single complete run of a task family instance. An episode comprises a fixed, ordered sequence of world events (inject, change, retract) and aligned probe points, all derived from a seed. An episode is the unit of counterfactual replay: the same episode (identified by `episode_id` and `seed`) is run under every memory condition with identical events and probes. See also: `world_event_hash`, `session`.

**session**
One contiguous interaction window within an episode. An agent perceives evidence, takes actions, and may query or update its memory system during a session. Between sessions, context is cleared (only the memory system persists, if one exists). The no-memory control is stateless across sessions; all other conditions persist via their adapter. Sessions simulate the temporal gaps in long-horizon tasks (e.g., "continue this research tomorrow").

**probe**
A fixed measurement point inserted at a specific step in the episode schedule. Each probe has a `kind` (e.g., task-completion, drift-check, retrieval), a `query` presented to the agent, and `gold` (ground truth derived from the world state at that step = revealed-minus-retracted facts). Probes are *aligned*: the same probe appears at the same step in every condition, enabling paired comparison.

**condition**
A specific memory system configuration under which an episode is run. Leaderboard conditions in v1 include: `no_memory`, `chromadb`, `mem0`, `letta`, `graphiti`, `cognee`. Two sensitivity conditions (`fake_perfect`, `fake_bad`) are included for metric validation but are not leaderboard-visible. Each condition is paired with the same episodes, enabling the counterfactual comparison.

**world_event_hash**
A stable, deterministic hash computed over the ordered exogenous event+probe schedule of an episode. Identical `world_event_hash` per (episode_id, seed) across all conditions guarantees that every condition sees exactly the same evidence world. Changes to the event schedule (e.g., adding or removing a retraction) produce a different hash. This hash is the primary reproducibility and integrity check.

### Tracks

**native track**
The PRIMARY leaderboard track. Each memory system is deployed as-is: its own defaults, its own internal LLM (if any), its own indexing strategy. All internal LLM tokens, embedding calls, storage bytes, and retrieval latency are instrumented and counted in the system's cost vector. The native track answers: "When I install this memory system and run it, what's the ROI?" This track is reported separately and never mixed with the controlled track.

**controlled track**
A SECONDARY sub-study track, reported separately from the native track and never merged into the primary leaderboard. In the controlled track, where a memory system supports pinning its internal model, the same model is used across all systems (typically the same model as the agent). This isolates the effect of the memory architecture from the effect of the internal model quality. Not all systems support this (it is opt-in); the controlled track is always a secondary analysis.

### Cost & Measurement

**cost vector**
A structured record of all resource consumption incurred during an episode run under a given condition. Fields:

| Field | Description |
|-------|-------------|
| `agent_input_tokens` | Tokens fed to the agent model (context + tool results) |
| `agent_output_tokens` | Tokens generated by the agent model |
| `mem_internal_in_tokens` | Input tokens consumed by the memory system's internal LLM calls (add, search, reflect, summarize, etc.) |
| `mem_internal_out_tokens` | Output tokens generated by the memory system's internal LLM calls |
| `embedding_tokens` | Tokens consumed for generating embeddings (text → vector) |
| `embedding_calls` | Number of embedding API calls |
| `storage_bytes` | Bytes stored/consumed in the memory backend |
| `retrieval_latency_ms` | Total wall-clock milliseconds spent in retrieval/search operations |
| `write_latency_ms` | Total wall-clock milliseconds spent in write/upsert/update operations |
| `reflection_tokens` | Tokens consumed by reflection/summarization/consolidation (subset of `mem_internal_*` but tracked separately for analysis) |
| `num_retrieval_calls` | Total count of search/retrieval operations |

Cost vectors are scalarized to **tokens-equivalent** using a declared, pinned conversion sheet (`configs/cost_weights.yaml`). Latency and storage are converted to token equivalents via that sheet. Dollar cost is a secondary, separately reported column using a pinned price sheet. One-time dataset-generation cost and judge cost are excluded from system cost vectors unless the system itself triggers them.

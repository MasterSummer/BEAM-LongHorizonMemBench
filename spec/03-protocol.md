# 03 — Protocol & Methodology (legacy v1)

> **Status:** preserved for the legacy `WorldEvent`/`Probe` benchmark only. It
> is not the protocol for the active state-first Software qualification. The
> active contribution and evidence contract is
> [`docs/long-horizon-benchmark-contract.md`](../docs/long-horizon-benchmark-contract.md),
> and the canonical server procedure is
> [`docs/systems-server-workflow.md`](../docs/systems-server-workflow.md).

This document defines the experimental protocol and statistical methodology for
LongHorizonMemSysBench v1. It remains the reference for legacy episodes and
adapters; its factual-probe and Memory ROI claims must not be carried into the
active BEAM paper.

## 1. Counterfactual Replay Protocol

### 1.1 Fixed Exogenous World

The core experimental design is a **counterfactual replay**: every episode is run identically across all memory conditions, varying only the memory system attached to the agent. The evidence world is *fixed and exogenous* for v1.

Each episode is defined by an `(episode_id, seed)` pair. From this pair, the simulator core deterministically generates:

1. An ordered sequence of **world events** (`inject`, `change`, `retract`) that define what the agent perceives.
2. An aligned sequence of **probes** (`task-completion`, `drift-check`, `retrieval`) with gold answers derived from the world state at each probe step (revealed-minus-retracted facts).
3. A **world_event_hash**: a stable, deterministic hash over the ordered event+probe schedule.

The world_event_hash is **identical** per `(episode_id, seed)` across all conditions. This is the integrity guarantee: every memory system faces exactly the same evidence. If a hash mismatch is detected between conditions for the same `(episode_id, seed)`, the run is invalid and the experiment is aborted.

### 1.2 Agent Actions Do Not Mutate the World (v1)

In v1, agent actions are *observational only*: the agent perceives evidence, queries its memory, and answers probes, but its actions do not change `WorldState`. The world event schedule is entirely exogenous and proceeds identically regardless of what the agent does. This is a deliberate v1 simplification:

- It eliminates confounding from divergent agent behavior changing the evidence available to later probes (agent A makes a different decision, changing the future world, making counterfactual comparison impossible).
- It enables clean, paired statistical comparison across conditions.
- It keeps the simulator deterministic and reproducible.

Interactive worlds (agent actions mutate future evidence) are deferred to v2.

### 1.3 Aligned Probe Points

Probes are inserted at fixed steps in the episode schedule. The same probe (same `probe_id`, same `kind`, same `query`, same `gold`) appears at the same step in every condition run of a given episode. This makes comparison **paired** at the probe level.

Probe gold is derived programmatically from the `WorldState` at the probe step. Specifically:
- A fact that has been `inject`-ed and not yet `retract`-ed → present in gold.
- A fact that has been `retract`-ed → absent from gold (even if the agent's memory system still holds it).
- A fact that has been `change`-d → only the latest version is in gold.

The enriched schedule (events + probes + gold) is frozen as part of the dataset and checksummed.

### 1.4 Session Boundaries

An episode is divided into **sessions**, simulating the temporal gaps in long-horizon tasks. Session boundaries are deterministic (derived from the seed) and identical across conditions. At a session boundary:

- The agent's working context is cleared (the context window is reset).
- The memory system persists (for all conditions except `no_memory`).
- The next session begins with the next world event in the schedule.

This is the mechanism that forces the agent to rely on its memory system: information from earlier sessions is only accessible if the memory system has stored and can retrieve it.

## 2. No-Memory Control Policy

### 2.1 Definition

The **no-memory control** is a condition where the agent has NO persistent memory system. It is the counterfactual baseline against which all memory systems are compared.

The no-memory control is **stateless across sessions**. At each session boundary, the agent's entire state is discarded. The only information available to the agent in a session is:

1. The events revealed during that session (current-session context).
2. A fixed context budget **B** — the same working-set size available to all other conditions for their current-session context.

The no-memory control does NOT retain any information from prior sessions. It cannot look up past events, past agent reasoning, or past probe answers. This is the definition of the counterfactual: what performance is possible without memory?

### 2.2 Context Budget B

Every condition, including no-memory, receives the same context budget **B** for the working set (current-session events + agent reasoning). The budget is measured in tokens and enforced identically across conditions. This ensures that any performance difference between a memory condition and no-memory is attributable to the memory system's ability to provide information *beyond the context window*, not to a larger working set.

### 2.3 ROI Counterfactual Role

The no-memory condition provides the baseline `score(no_memory)` in the Memory ROI formula:

```
gain = score(system) − score(no_memory)
normalized_gain = clamp(gain / max(ε, max_achievable − score(no_memory)), −1, 1)
ROI = mean(normalized_gain) / mean(normalized_cost)
```

Without a no-memory counterfactual, it is impossible to distinguish a memory system that genuinely improves performance from one that merely correlates with an easier episode.

The no-memory condition's own ROI is `N/A` (not zero), because ROI is defined as a comparison against a baseline and the baseline cannot be compared to itself. This is enforced in all reporting and never displayed as `0` or omitted.

## 3. Track Policy: Native and Controlled

### 3.1 Native Track (PRIMARY)

The **native track** is the primary leaderboard. Each memory system is deployed in its default configuration:

- Its own internal LLM (if any), with its own model, prompt, and token usage.
- Its own indexing, embedding, retrieval, and storage strategies.
- Its own defaults for chunking, top-k, similarity thresholds, etc.

All internal resource consumption (LLM tokens, embedding calls, storage bytes, retrieval latency) is instrumented and counted in the system's cost vector.

The native track answers the deployment question: *"If I install this memory system as-is, what is my Memory ROI?"*

### 3.2 Controlled Track (SECONDARY)

The **controlled track** is a secondary sub-study, reported separately from the native track and never merged into the primary leaderboard.

In the controlled track, where a memory system supports pinning its internal model, the same model is used across all systems. This isolates the effect of the memory *architecture* (how facts are stored, organized, and retrieved) from the effect of the internal model *quality*. A memory system with a weaker internal LLM may underperform on the native track but match or exceed on the controlled track, revealing whether the architecture itself is sound.

The controlled track is **opt-in**: systems that do not support model pinning are simply absent from the controlled track. It is always a secondary analysis; conclusions are drawn from the native track.

### 3.3 Separation Rule (Never Mixed)

Native-track and controlled-track results are **never mixed** in any single leaderboard, table, or plot. They are presented in separate sections of the scorecard. Any aggregation, ranking, or statistical comparison operates within one track only. Violating this rule would conflate two different questions (system-as-deployed vs. architecture-only) and produce uninterpretable results.

## 4. Statistical Methodology

### 4.1 Experimental Design

Paired counterfactual design:

- **N episodes per family** (pilot default ≈20 per family, configurable).
- **K seeds** (pilot default ≥3, configurable), each producing a different episode instance.
- Every (episode_id, seed) is run under every condition, producing a matched observation per condition per episode.

The unit of analysis is the episode-condition pair. Metrics are aggregated first per episode (over probes within the episode), then per condition (over episodes), then compared pairwise (condition vs. no-memory).

### 4.2 Bootstrap Confidence Intervals

All headline metrics (Memory ROI, task score, drift_index, retrieval metrics) are reported with **bootstrap confidence intervals** (default 95% CI, ≥10,000 resamples, configurable).

Procedure:
1. Sample N episodes with replacement from the episode pool.
2. Compute the metric on the resampled set.
3. Repeat R times.
4. Report the 2.5th and 97.5th percentiles as the 95% CI.

Bootstrap CIs are preferred over parametric intervals because the sampling distribution of ratios (like ROI) and the distribution of episode difficulties are not guaranteed to be normal.

### 4.3 Effect Size

In addition to CIs, report a standardized **effect size** for the primary comparison (each condition vs. no-memory). The pilot is powered to detect a predeclared minimum effect size. The effect size metric and minimum are declared in the run config; default is Cohen's d on per-episode score deltas.

### 4.4 Failed Runs Included (Not Dropped)

A run that fails (timeout, crash, malformed output) is NOT dropped from the analysis. The failure is scored as:

- Task score = 0 for that episode (the agent failed to complete the task).
- Cost vector = actual cost incurred up to the point of failure (tokens consumed, API calls made, storage written before the crash).
- A failure flag is recorded in the episode result for diagnostic reporting.

Dropping failed runs would bias the results upward (systems that fail on hard episodes would appear stronger than they are). Including them ensures the results are honest about worst-case behavior. Partial successes (some probes answered before failure) are included with answered probes scored normally and unanswered probes scored as 0.

### 4.5 Pareto Analysis

Memory ROI alone can hide tradeoffs. A condition with high ROI but low absolute performance may not be useful; a condition with low ROI but the highest absolute scores may be preferred in some applications. Therefore every scorecard includes a **Pareto frontier** plot:

- X-axis: normalized cost (mean per episode).
- Y-axis: normalized gain (mean per episode).
- Each condition is a point; the Pareto frontier connects the non-dominated conditions.
- The no-memory control is plotted at (0, 0) by definition (its own ROI is N/A, but it serves as the reference point on the Pareto plane).

The Pareto plot is accompanied by both gain floor and cost ceiling lines so that the viewer can identify the region of practical interest. A bare leaderboard number without uncertainty intervals and Pareto context is never reported; this is a structural requirement enforced by the reporting module.

## 5. Failure Policy

### 5.1 Timeouts and Retries

Every episode run has a predeclared **timeout** (wall clock, configurable per family). If the agent does not complete all sessions within the timeout, the run is terminated and scored as a failure.

Each operation (LLM call, memory adapter call) has a bounded number of **retries** (default 2, configurable). If all retries are exhausted, the run is scored as a failure. Retries are counted in the cost vector.

### 5.2 Crash Policy

If the agent process, memory system, or harness crashes:
- The run is scored as a **task failure** (score = 0 for remaining probes).
- All cost incurred before the crash is **retained** in the cost vector (tokens charged, API calls counted, storage measured).
- A crash record is logged with stack trace and diagnostics.

### 5.3 Malformed Output

If the agent produces output that cannot be parsed by the programmatic checker:
- The probe is scored as 0.
- The malformed output is logged.
- No retry is attempted (to avoid inflating cost on an unbounded parse-retry loop).

### 5.4 Unsupported Adapter Operations

Memory systems vary in their feature sets. Some may not support update, delete, or reflection. When an adapter receives a call it does not support:

- It MUST raise a well-defined `UnsupportedOperation` (not crash silently).
- The harness catches this, logs it, and proceeds with the remaining operations.
- The capability gap is recorded in the condition's result (so reporting can note "system X does not support updates, which may affect drift scores").

Graceful capability degradation is preferred over crashing the entire run. However, if a missing capability makes a probe unanswerable (e.g., a drift probe requires checking whether a retracted fact is still retrievable), that probe is scored as 0 for that condition.

### 5.5 Malformed Memory State

If a memory system enters an inconsistent state (corrupted index, silent retrieval failure, duplicate IDs):
- The condition is flagged with a diagnostic marker.
- The run continues (to avoid biasing results by dropping hard cases).
- The incident is reported in the diagnostic appendix of the scorecard.

### 5.6 Exclusion Rule

The only case where an episode run is excluded from analysis entirely is a **world_event_hash mismatch** between conditions for the same `(episode_id, seed)`. This indicates a determinism bug in the simulator or dataset, not a memory system property, and the run is invalid data. Such exclusions must be rare (zero in normal operation) and are reported conspicuously.

# GPT-5.6 AAAI experiment protocol

This document freezes the confirmatory protocol for the first full controlled
track. It must be committed before dataset generation and may not be changed in
response to full-run outcomes. Any later deviation is recorded as exploratory.

## Experimental unit and dataset

- Dataset release: `software-vertical-mem0-v0.5.0`.
- 50 generated episodes, 16 sessions and 15 handoffs per episode.
- The episode is the primary repeated-trajectory unit. SCEUs within an episode
  are repeated measurements, not independent samples. Because episodes cross
  five semantic scenarios with ten schedules, scenario-clustered and
  leave-one-scenario-out sensitivity analyses accompany episode-level results;
  the 50 episodes are not described as 50 independent semantic templates.
- Five preregistered software scenarios and ten event schedules are crossed
  across semantic seeds. Explicit, derivable, and absent workspace variants are
  balanced across trajectory seeds.
- Every frozen episode, surface, workspace, seed, and generator revision is
  content-addressed. `verify` and `regen-check` must both pass.

## Compared conditions

The sole continuation policy is `gpt_5_6_sol_shengsuanyun`, pinned to routed
model ID `openai/gpt-5.6-sol`. The seven conditions are
workspace-only, full-context, oracle-current-state, flat retrieval, Mem0,
A-MEM, and MemOS. Native memory writers remain fixed backend components rather
than compared policy models. Common retrieval/reranking and native retrieval
readouts are reported separately.

## Primary outcomes

1. Programmatically checked continuation accuracy and behavior score.
2. State-evolution and conflict-resolution accuracy.
3. Eligible-denominator long-horizon behavioral drift, split into
   `constraint_loss`, `plan_deviation`, `stale_state`, and
   `local_over_global`.
4. Storage quality on two independent axes: lifecycle-event provenance
   (`native event`, `inventory-diff inferred`, or `unavailable`) and semantic
   state attribution (`exact_signature`, `unique_provenance`, `ambiguous`, or
   `unavailable`). Deterministic lexical signatures and supported `no_match`
   assignments are also retained explicitly. Inferred lifecycle events are a labeled sensitivity analysis;
   a native event is never described as semantically exact merely because its
   add/update/delete operation was observed exactly.
5. The four-stage retrieval chain: backend-retrieved, selected, model-visible,
   and behaviorally used state/memory objects.

The invariant drift estimate compares matched opportunities with the same
current latent-state signature. Opportunities whose state legitimately changes
are evaluated separately as state-evolution resolution, not counted as drift.

## Causal and scaling analyses

- Primary causal-use intervention: equal-count neutral replacement of one
  target memory object, preserving position and character length.
- Because one preregistered focal object is probed per SCEU, reports include
  probe coverage and the causal-use rate among probed objects. The used/visible
  ratio is labeled a lower bound, never an exhaustive utilization estimate.
- Negative control: the same replacement procedure applied to a non-target
  (sham) object.
- Leave-one-out is retained as a sensitivity analysis because it confounds
  content removal with memory count.
- Memory-count scaling uses pre-registered +1, +5, and +20 neutral-object arms
  within the same episode, checkpoint, SCEU, baseline evidence, and policy.
  Early/late checkpoint comparisons are not interpreted as memory scaling, and
  native live-store size remains an observational quantity.
- Native object count and attributed logical state-unit count are both reported;
  objects-per-state and unattributed-object rates quantify backend granularity.

## Statistical analysis

- Report episode means and episode-clustered 95% bootstrap confidence intervals
  using 10,000 deterministic resamples.
- Predeclared paired contrasts compare each memory system with workspace-only
  and flat retrieval, plus common-rerank with native retrieval when both exist.
- Use paired episode-level sign-flip permutation tests, Holm correction within
  each outcome family, paired Cohen's dz, and the observed-variance 80% power
  minimum detectable effect.
- Missing or failed tasks remain explicit. No complete-case replacement,
  opportunity-level pseudo-replication, or post-hoc denominator changes are
  permitted.

## Execution and acceptance gates

The run must use a clean commit, native venvs/services and Slurm, never Docker or
another container substitute. Prefix preparation produces 200 tasks (50
episodes by four memory backends); evaluation produces 350 tasks (50 episodes
by seven conditions). Before the full run:

- repository tests pass on Linux;
- a one-episode server smoke validates all artifacts;
- every write has exact or explicitly inferred provenance;
- every final memory object has an explicit semantic-attribution method, and
  ambiguous objects contribute no positive state coverage;
- future-state, stale-state, constraint-loss, and local-over-global fixtures
  each produce their expected positive and negative checks;
- no action supplies more than 50% of gold-valid opportunities and no opaque
  option supplies more than 40%;
- storage, causal-use, drift, and count-contrast metrics are finite where their
  denominators are non-zero.

The final artifact is acceptable only if all 350 evaluation tasks complete,
validation reports `ok: true`, code and dataset hashes match the frozen
manifests, and both per-episode and aggregate reports (including statistics and
limitations) are retained.

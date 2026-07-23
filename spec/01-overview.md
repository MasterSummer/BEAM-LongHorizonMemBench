# 01 — Overview, scope, contributions, and glossary

## Problem statement

Long-horizon tasks require an agent to pursue one persistent overall goal in a
bounded environment while progress, decisions, intermediate results, and
constraints remain causally relevant across hundreds to thousands of effective
transitions and context resets. A workspace preserves some task state, while a
task-level memory system must decide what else to write, update, retrieve, and
expose when a later session resumes.

BEAM requires goal persistence, a replayable pre-decision dependency chain,
delayed state dependence after context reset, and separately measured
workspace/memory channels. Session count, prompt length, and filler events do
not independently establish a long horizon.

The evaluation problem is therefore not simply whether an agent can answer a
question about a long history. It is whether the memory system maintains the
**current and authoritative task state**, contributes information beyond the
workspace, and changes the agent's next executable action in the right way.

BEAM evaluates memory-supported state maintenance and behavioral control at
critical continuation decisions sampled from replayable persistent-task
trajectories. The definition targets hundreds to thousands of mutually
dependent online Agent/environment steps; the current experiment is a
controlled critical-decision sampling protocol within that target setting.
The main v0.10 Software release does not claim that the tested continuation
policy executes the full multi-hundred-step trajectory online. Its
long-horizon claims are restricted to explicitly measured cross-session
handoffs, state age, dependency depth, state transitions, authority/scope
conflicts, and workspace recoverability. The supplementary v0.11 matched
release records at least 256 effective frozen/environment prefix transitions
per 16-session member. Its target decision is after the final prefix and 15
handoffs, so post-decision work cannot count toward the horizon. The policy is
still evaluated only at sparse continuation decisions.

Task length and evaluation density are distinct. Future releases may replay
hundreds or thousands of meaningful environment/agent transitions while
querying the policy only at selected critical decisions, but the manifest must
separately report effective transitions, policy-controlled decisions, replayed
segments, and generated segments. Padding events or repeated text do not count
toward the horizon.

An effective v0.11 transition must produce one unique semantic task effect,
consume the effects produced by its declared predecessors, and carry a
reproducible digest. The plan loader verifies both the semantic dependency
chain and the digest chain. The ≥200 threshold is applied to effective causal
ancestors of a scored continuation, not to episode length, post-decision work,
or a chain of unconsumed observations. Reports separately count
`policy_evaluated`, `frozen_replay`, and `environment_generated` steps; these
    categories may not be described interchangeably. The v0.13 longitudinal
    release applies the same anti-padding contract to 256 public prefix steps
    and 13 registered critical decisions (269 effective steps total).

Every report also derives a trajectory interaction tier. A
`replay_backed_critical_decision` has an audited causal prefix but no declared
policy-to-later-policy dependency. `sparse_closed_loop` requires at least one
policy decision to affect a later effective step and a later policy decision.
`online_long_horizon_agent_execution` additionally requires at least 200
causally linked policy-evaluated steps. Current v0.11/v0.12/v0.13 releases are in the
first tier. This prevents task-span evidence from being silently promoted to a
stronger online-rollout claim.

## Scope of the active qualification

The active GPT-only matrix contains:

- controls: `workspace_only`, `full_context`, and `oracle_current_state`;
- retrieval baseline: `flat_retrieval`;
- memory systems: `mem0`, `amem`, and `memos`;
- readouts: native and common reranking where defined;
- one fixed continuation/policy profile per experiment identity.

Memory systems write using their native lifecycle and representation. The
benchmark owns the frozen task trajectory, public workspace, continuation
request, opaque action catalog, executable checker, evaluator state graph, and
counterfactual interventions. Controlled and native readouts answer different
questions and are never pooled into one score.

## Contributions

### C1. Counterfactually controlled, workspace-aware long-horizon benchmark

Episodes are generated from latent `StateUnit` and `StateEvent` records before
public surfaces are rendered. State can be replaced, revoked, invalidated,
reopened, reprioritized, or scope-limited. The evaluation unit is a
State-Conditioned Evaluation Unit (SCEU): one checkpoint, current dependency
closure, workspace snapshot, opaque action catalog, and programmatically
checked continuation.

The workspace is a competing information channel rather than an accidental
part of the memory condition. Primary system value is measured on exactly
matched SCEUs as behavior gain beyond workspace-only and oracle-gap closure.
Full context diagnoses whether failure comes from task difficulty or memory
selection. Results are stratified by handoff, state age, event distance,
dependency depth, transition count, construct kind, and workspace
recoverability.

Every SCEU must also include every *current* state that can distinguish the
validity of any offered action. Stable task-governance rules—replacement and
revocation precedence, authority precedence, and scope non-generalization—are
part of the task contract shown to every condition. Oracle receives current
state facts, not privileged semantics. An incomplete current-action state
closure or condition-specific governance rule fails the design audit before
model calls, because otherwise an oracle/control error could be an evaluator
contract defect rather than a memory failure.

For the matched release, full context and oracle current state are not merely
reference bars. Every evaluated policy must solve every frozen
group/opportunity under all three history variants with each control. Oracle
failure means the terminal task is not established as solvable; full-context
failure means history interpretation is still a confound. Either failure
blocks a memory-channel claim for that report.

For construct identification, v0.11 generates `static`, `evolution`, and
`hierarchical_conflict` members with the same terminal request, checkpoint,
actions, gold action, opaque option map, continuation scope, checker-relevant
terminal predicates, and prefix/workspace shape. Across groups, terminal
archetypes balance current-v1, current-v2, and valid scoped-exception gold
actions, while their opaque option positions are balanced separately. The
primary mechanism outcomes are the evolution and conflict penalties relative
to matched static **after subtracting the corresponding workspace-only
penalty**. Raw performance, correctness, and endpoint drift-violation
differences relative to static are secondary; they cannot identify a
memory-channel effect when the history manipulation also changes workspace
recoverability.

These roles, the counterfactual-group analysis unit, the workspace adjustment,
effect direction, paired test, multiplicity scope, and endpoint-only drift
boundary are frozen in `experiment_design_audit.analysis_contract` before any
writer or continuation-policy call. The contract is content-addressed in the
run identity and revalidated from report artifacts. It is an internal pre-call
specification rather than a claim of external public preregistration.

A supplementary v0.12 same-decision horizon panel uses 4, 8, and 16 sessions (65,
129, and 257 effective transitions; 3, 7, and 15 handoffs). For each history
construct it fixes the terminal current state, workspace semantics, action
catalog, gold action, opaque option map, package, and hidden checker. The
primary diagnostic is the change from short to long in the workspace-adjusted
evolution/conflict penalty. This is a joint transition/handoff dose, not a pure
handoff manipulation. Generation, freeze/regeneration, planning, panel-level
statistics, and artifact validation are implemented; the policy/backend
calibration remains a separate, not-yet-completed evidence stream. Each nine-
member panel contributes one statistical unit, and physical-member inference is
suppressed by the report contract.

### C2. Goal-relative long-horizon behavioral drift

BEAM operationalizes four state-grounded action failures:

- `constraint_loss`: a persistent constraint loses behavioral influence;
- `plan_deviation`: action departs from the currently valid plan, including
  premature adoption of a future plan;
- `stale_state`: a revoked or superseded state still controls action;
- `local_over_global`: a local goal or scoped exception overrides a global goal
  or higher-authority constraint.

Rates use category-specific eligible denominators. A checked single-decision
error is first reported as a drift-compatible violation. It becomes an observed
longitudinal drift event only after adherence to the same category and the same
state lineage was observed at an earlier distinct eligible checkpoint.
First-observation errors therefore have no drift onset, and adherence to one
constraint cannot anchor a later failure of another plan. Longitudinal outputs
include adherence-anchored first-drift
handoff, drift-free survival, persistence across distinct checkpoints, and
recovery after a valid update or fresh reminder. The episode is the statistical
unit; SCEUs and state lineages are repeated observations within an episode.
Category-only legacy trajectories are descriptive and cannot satisfy the C2
gate. Oracle-current-state and full-context trajectories must cover the same
lineages without drift before an onset can be attributed to memory. The
single-endpoint matched release reports violation excess, not longitudinal
onset.

One SCEU/category must identify one focal lineage. A continuation whose same
category refers to multiple state lineages is split into separate SCEUs; the
report fails closed instead of applying one observed category flag to every
eligible lineage.

BEAM does not claim that behavioral drift as a general concept is new. The
contribution is a reproducible operationalization tied to versioned task state,
workspace controls, and executable behavior under ordinary task evolution.

### C3. Decision-aligned memory-to-behavior attribution

For the same SCEU, the evaluator reconstructs:

```text
stored -> backend-retrieved -> model-visible -> intervention evidence -> behavior
```

The earliest supported failure is localized as:

1. storage failure: required workspace-absent state is not represented in an
   observed store;
2. retrieval failure: stored required state is missing from backend retrieval;
3. exposure failure: backend-retrieved state is removed before the model sees
   it;
4. utilization/decision failure: required state is visible but the executable
   action is wrong.

Decision-layer failures are further separated by targeted interventions:
`visible_without_detected_unique_causal_effect`,
`visible_causally_influential_but_wrong`, or
`visible_use_evidence_incomplete`. This prevents “visible” from being silently
equated with “used,” and prevents a causally influential but wrong memory from
being mislabeled as non-use.

An unavailable lifecycle or semantic attribution is reported as
`storage_evidence_unavailable`, never silently converted into a storage
failure. Native/exact and inventory-inferred lifecycle provenance are separated.

Successful behavior does not prove memory use. A memory object is labelled
causally used only when a repeat-stable, state-targeted replacement or
leave-one-out intervention changes the action or checker result. This is a
lower bound on unique causal influence; correct behavior without detected
effect remains `behavior_success_without_detected_unique_causal_effect` or
`behavior_success_unprobed`. No detected effect does not exclude redundant or
compensated use.

The report additionally pairs memory conditions at an identical policy,
readout, episode, SCEU, opportunity, checkpoint, current-state contract, and
selected action. `outcome_equivalent_fault_profile_divergence` is the fraction
of these outcome-equivalent pairs whose earliest supported stage or
intervention-grounded utilization subtype differs. It demonstrates when the
same end-task outcome hides a different repair target. The pair rows are
dependent descriptive diagnostics rather than independent statistical units;
zero divergence is retained and is not interpreted as equivalence.

## Primary evidence and metrics

| Claim | Primary evidence |
| --- | --- |
| Memory helps beyond available task artifacts | behavior gain beyond workspace-only; gap to full context; oracle-gap closure |
| Current state is maintained | current-state storage precision/recall/F1; stale retention; update/delete responsiveness; current-state and conflict resolution |
| Writes are useful and selective | write coverage, write selectivity, write-to-continuation alignment |
| Retrieval and prompt exposure are distinct | stored-to-retrieved and retrieved-to-visible conditional yields |
| Memory affects behavior | state-targeted detected-unique-effect lower bound and first-failure distribution |
| End-task outcomes conceal mechanism differences | outcome-equivalent fault-profile divergence on the same decision and action |
| Behavior drifts over the trajectory | eligible drift rate, onset, survival, persistence, and recovery |
| Memory volume affects selection | matched within-SCEU memory-object-count interventions; not tokens or unmatched checkpoints |
| Construct effect is not a default-action artifact | paired penalties by terminal archetype plus always-action and always-option shortcut gates |
| A mechanism is specifically horizon-amplified | same-decision short/medium/long change in the workspace-adjusted evolution/conflict penalty |

Undefined denominators remain null. Lifecycle provenance and semantic alignment
are separate axes. Measurement-readiness gates are separate from artifact/hash
validation.

## Positioning and claim boundary

BEAM must not claim to introduce memory lifecycle evaluation or behavioral
drift as concepts. Adjacent work already covers incremental memory
competencies, dynamic state questions, arbitrary-length agent trajectories,
task-level memory benefits, module-level lifecycle analysis, and adversarial
memory drift. Recent work also directly studies behavioral state decay and
paired write/retrieval/utilization diagnostic profiles. BEAM's defensible
novelty is therefore not any one of those labels; it is their joint
decision-level
instrumentation:

> evolving authoritative task state + workspace control + executable
> continuation + causal storage/retrieval/exposure/utilization localization.

The resulting estimand is construct-specific degradation of a delayed task
state-control channel, not recall accuracy under a longer prompt. Static,
evolution, and conflict histories are paired at a final executable decision;
workspace supplies a measured alternative information path; native lifecycle
traces locate the earliest supported failure; and longitudinal drift records
its behavioral consequence. Removing this matched identification structure
would reduce the benchmark to descriptive “MemoryBench at longer context.”

The v0.10 construct-stratified scorecards remain descriptive. A causal claim
that state evolution itself changes performance is reserved for the separate
v0.11 matched mechanism experiment, which fixes the terminal decision contract
and prefix/workspace shape across static, evolution, and hierarchical-conflict
variants. Memory-object exposure is still an observed backend outcome and is
reported rather than asserted to be perfectly fixed across native systems.

The detailed contribution/evidence contract and related-work links are in
[`docs/long-horizon-benchmark-contract.md`](../docs/long-horizon-benchmark-contract.md).

## Canonical glossary

**episode**

One replayable persistent-task trajectory generated from fixed semantic and
trajectory seeds. It is the unit of statistical inference.

**session / handoff**

One isolated interaction window and the boundary after which working context is
cleared. Handoff count is a temporal-horizon variable, not a synonym for token
length.

**workspace**

Agent-visible task artifacts at a checkpoint. Required state is labelled
`explicit`, `derivable`, or `absent` evaluator-side; those labels are never
shown to the policy.

**State-Conditioned Evaluation Unit (SCEU)**

One continuation decision bound to current focal state, dependency closure,
workspace recoverability, an action catalog, and a programmatic checker.

**memory-reliant state**

Current required state that is absent from the workspace. This is the primary
denominator for the memory-to-behavior attribution funnel. Derivable state is a
separate sensitivity track.

**stored**

A current native memory object is deterministically aligned to a required state
with recorded lifecycle provenance. Ambiguous alignment earns no positive
coverage.

**backend-retrieved**

Objects returned by the memory backend before benchmark-owned reranking,
truncation, or prompt-budget filtering. An explicitly empty result is distinct
from a legacy record in which the field was not recorded.

**model-visible**

Objects actually serialized into the continuation policy's input.

**causally used**

A conservative label supported by a stable targeted intervention. Visibility,
retrieval, attention claims, or model self-report are insufficient.

**controlled track**

Systems share the benchmark-owned embedding/reranking configuration so memory
architecture and policy differences can be studied under a common readout.

**native track**

Each memory system uses its official or declared native configuration. It
measures the complete deployed system and is reported separately from the
controlled track.

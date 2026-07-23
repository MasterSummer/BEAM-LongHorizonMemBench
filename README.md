# BEAM: Diagnosing Memory-Supported State Control in Long-Horizon Tasks

BEAM is a counterfactual diagnostic benchmark that evaluates whether a memory
system maintains the right evolving task state
and causally supports the correct next action when an agent resumes a persistent
project. It measures memory-supported state maintenance and behavioral control
at critical continuation decisions through versioned task state and executable
behavior.

Long-horizon means one persistent goal with a replayable pre-decision causal
chain across context resets. The target setting is an
agent executing hundreds to thousands of mutually dependent steps in one
bounded task environment. Reports separately audit how much of each trajectory
was policy-evaluated, frozen replay, or environment-generated, so a sparse
critical-decision experiment cannot be mislabeled as a full online rollout.
Difficulty is measured through effective transitions, session handoffs, state
age, dependency depth, state evolution, authority/scope conflicts, and
workspace recoverability. Each system writes memory in its native
representation. At the same decision, BEAM reconstructs:

```text
stored -> backend-retrieved -> model-visible -> intervention evidence -> behavior
```

The current GPT-only qualification compares Workspace-only, Full-context,
Oracle-current-state, Flat retrieval, Mem0, A-MEM, and MemOS under a fixed
continuation policy. The canonical server workflow uses native Python virtual
environments, native services, and Slurm; Docker and other containers are not
part of the project or formal experiments.

## Core contributions

| Contribution | What is measured |
| --- | --- |
| **Counterfactual identification of state control under competing persistent channels** | State-first versioned goals, constraints, plans, revocation/invalidation, scoped authority, and executable continuations; v0.11 fixes the terminal decision while varying static/evolution/conflict history, then subtracts the matched workspace-only penalty to identify memory-channel value. Full-context and oracle controls must solve every matched history variant for each policy before a memory-channel claim is allowed. |
| **Goal-relative behavioral drift protocol** | Constraint loss, plan deviation, stale-state use, and local-over-global behavior. Single-point violations are separated from same-state-lineage adherence-anchored onset, persistence, drift-free survival, and reminder/update recovery; oracle/full-context contamination blocks a memory-specific claim. The contribution is the state-grounded measurement protocol, not the discovery of behavioral decay. |
| **Decision-aligned causal fault localization** | The earliest supported storage, retrieval, exposure, or utilization failure, with native/exact and inventory-inferred provenance separated and an explicit unobservable category. An outcome-equivalent diagnostic measures when the same checked action hides different observed failure profiles across memory systems. |

A supplementary horizon-dose diagnostic compares the same terminal state,
workspace semantics, opaque options, correct action, package, and hidden checker
after 4, 8, or 16 sessions. Its effective spans are 65, 129, and 257 transitions
with 3, 7, and 15 handoffs. The analysis asks whether the workspace-adjusted
state-evolution or authority-conflict penalty is larger at the long dose than
at the short control. Because transitions and handoffs change together, this
is a joint horizon-dose test rather than a pure handoff effect.

The independent v0.13 longitudinal release operationalizes C2 rather than
inferring drift from isolated errors. Each 16-session episode contains 13
registered continuation decisions over the same state lineages, including a
final reminder after ordinary plan, stale-state, constraint, and
local-over-global challenges. Its pre-call audit requires an earlier adherence
anchor, a later non-control challenge, and a still-later reminder/update
recovery opportunity for every drift category. It also requires current,
action-discriminative intervention targets for every memory-reliant decision.
The episode—not its 13 repeated decisions—is the statistical unit.

Behavior is checked programmatically. Identified unique causal influence is a
conservative lower bound from repeat-stable, state-targeted memory-object
interventions; visibility or model self-report never counts as use. A
visible-state error is subtyped as no detected unique causal effect, causal
influence with wrong behavior, or incomplete probe evidence. No detected
effect does not exclude redundant or compensated use. Memory-object count is evaluated with
matched within-opportunity interventions rather than token length or unrelated
checkpoints.

The v0.11 mechanism release records at least 256 effective task transitions per
16-session member before a final-session decision, after 15 handoffs, and
verifies their dependency/effect digest chain. Every counted step produces a
unique semantic task effect, consumes the effects of its declared predecessors,
and lies in the causal ancestry of a scored continuation; a digest-only chain
or a trailing step cannot satisfy the long-horizon threshold. It separately reports
policy-evaluated, frozen-replay, and environment-generated steps; it does not
relabel every frozen transition as an online model decision. Across matched
groups, current-v1, current-v2, and valid scoped-exception gold actions and
their opaque option positions are balanced, so a default-safe action cannot
solve the mechanism split.

Each task-span row additionally declares one of
`no_policy_evaluation`, `replay_backed_critical_decision`,
`sparse_closed_loop`, or `online_long_horizon_agent_execution`, together with
policy-conditioned downstream-step and later-decision counts. The current
matched/horizon/longitudinal releases are intentionally in the replay-backed critical-
decision tier. Their contribution is controlled identification of delayed
task-state control, not a claim that the tested policy performed all prefix
steps online.

Its primary mechanism statistic is not the raw accuracy drop from static to
evolving history. It is the difference-in-differences between that drop for a
memory condition and the drop for workspace-only on the same counterfactual
group. This prevents a changed file/log surface from being misreported as a
memory-system effect. Oracle and full-context controls separately test terminal
solvability and history availability. These controls are enforced per policy
over all matched static/evolution/conflict cells; a failed control blocks the
memory-channel interpretation rather than serving only as a loose bound.

BEAM also reports **outcome-equivalent fault-profile divergence**. It pairs two
memory conditions only at the identical policy, readout, episode, SCEU,
checkpoint, required-state set, and selected action. A positive value shows
that end-task accuracy would have hidden a storage/retrieval/exposure/use
difference; the last term means a difference in registered intervention
evidence, not direct observation of internal use. The pairs are dependent descriptive diagnostics, not additional
independent samples, and a zero value is not evidence of equivalence.

The precise claim boundary, related-work positioning, metrics, and wording for
the paper are in [the long-horizon benchmark contract](docs/long-horizon-benchmark-contract.md)
and the [Chinese paper-contribution positioning note](docs/paper-contribution-positioning-zh.md).
Every qualification report also emits `contribution_evidence.json` and
`contribution_evidence.md`. These artifacts bind C1--C3 to their required
controls, estimands, readiness gates, and claim boundaries. An evidence status
of `ready` means that the measurement contract is complete; it does not mean
that the estimated effect is positive, statistically significant, or
confirmatory.

For a run completed under an older report schema,
`python -m lhmsb.qualification audit-completed-report` creates a separate,
hashed, read-only contribution audit. It records source integrity and analysis
timing, inventories C1--C3 evidence gaps, and identifies possible zero-API
reaggregation when given the exact frozen dataset through `--dataset`, without
overwriting the canonical report or backdating a new claim.

For a verified candidate,
`python -m lhmsb.qualification reanalyze-completed-report` reconstructs the
same-decision storage → retrieval → model-exposure → intervention-evidence →
behavior funnel and state-lineage drift trajectories into a separate, hashed,
explicitly post-hoc directory. It makes no model/backend calls and rejects any
frozen dataset whose manifest does not match the source run.

Before any writer or policy-model call, `plan-systems` now emits and binds
`experiment_design_audit.json` into the immutable run identity. A matched run
is rejected if its triplets, gold actions/options, workspace recoverability,
terminal archetypes, drift-checker controls, memory-reliant decisions, or
effective-step causal chains violate the declared design. One-group runs remain
explicitly `diagnostic_only`; a balanced three-or-more-group release may become
`ready_for_calibration`.

For v0.13, the same audit fails before API calls if any canonical drift category
lacks a same-lineage anchor/challenge/recovery window, if a memory-reliant SCEU
lacks a current action-relevant intervention target, or if its 256-step
pre-decision causal chain fails the semantic-effect anti-padding audit.

The same pre-call artifact freezes contribution-specific analysis contracts for
C1--C3. C1's two primary estimands are the state-evolution and
hierarchical-conflict penalties after subtracting the matched workspace-only
penalty. C2's scope is frozen as either endpoint violation for a matched/horizon
release or lineage-backed longitudinal drift when repeated-checkpoint gates are
available; an endpoint cannot be relabelled as onset. C3 freezes the
same-decision trace order, exact/inferred provenance tracks, and repeat-stable
neutral-replacement intervention used to identify a unique observable causal
effect. The report validator checks these roles independently of artifact
hashes. This is an internal, content-addressed pre-specification, not a claim of
external public preregistration.

The operator must also declare `analysis_phase` as `development`,
`diagnostic`, `calibration`, or `confirmatory`. The phase is hashed into the
run identity and rechecked by every worker. A matched calibration requires at
least 3 counterfactual groups, while a matched confirmatory run requires at
least 30 groups; the corresponding standard and longitudinal-release minima are
5 and 50 independent episodes. The report validator independently reapplies the same eligibility
contract, so bypassing the planner cannot produce a valid undersized report.
These are scale and labeling guards, not substitutes for a disjoint
freeze, preregistration, measurement readiness, or statistical evidence.

## Legacy v1 adapter coverage (not active)

The original v1 harness contains six leaderboard adapters plus two
calibration-only sensitivity oracles. This table documents preserved interfaces;
it is not the current Mem0 qualification matrix.

| Condition | Type | Description |
|-----------|------|-------------|
| `no_memory` | Control | Stateless baseline -- no memory persists across sessions. |
| `chroma` | Leaderboard | In-browser/embedded vector database (ChromaDB) with deterministic offline embeddings. |
| `mem0` | Leaderboard | Hybrid memory system (vector + graph + key-value) with internal LLM extraction. |
| `letta` | Leaderboard | Self-editing block memory with sleeptime consolidation. |
| `graphiti` | Leaderboard | Temporal knowledge graph with entity/edge extraction. Requires a graph database. |
| `cognee` | Leaderboard | Self-reorganizing triple-store with graph enrichment. |
| `fake_perfect` | Sensitivity only | Calibration oracle that returns only current, non-retracted facts. Establishes the upper bound. |
| `fake_bad` | Sensitivity only | Adversarial oracle that returns retracted/superseded facts. Establishes the lower bound. |

`fake_perfect` and `fake_bad` are metric-validation tools. They are excluded from the leaderboard but reported in a separate calibration section.

### Wide Research first slice

The first external-data experiment uses exactly three conditions: `no_mem`, `mem`, and `wrong_mem`. Trace construction has three sealed phases: export question-only records, retrieve and freeze multi-session observations without evaluator gold, then join gold labels after the trace hash is fixed. The official bundle contains 400 Wide rows; one exact duplicate is removed, leaving 399 unique questions. The formal qualification rule keeps only episodes with at least one observed gold paper; it selects among already-frozen traces and cannot alter retrieval.

```bash
# Phase 1: no answer/arxiv_id fields are exported.
python -m lhmsb.datasets wide-questions \
  --input runs/external/AutoResearchBench.jsonl \
  --out runs/wide_formal/questions

# Phase 2: build a local index from all ten Parquet shards of the pinned
# librarian-bots/arxiv-metadata-snapshot revision, then freeze three sessions.
python -m lhmsb.datasets build-arxiv-index \
  --input runs/external/arxiv-metadata-snapshot/data/train-*.parquet \
  --out runs/wide_formal/arxiv-metadata.sqlite

python -m lhmsb.datasets wide-traces \
  --questions runs/wide_formal/questions/questions.jsonl \
  --search-backend local --index runs/wide_formal/arxiv-metadata.sqlite \
  --sessions 3 --top-k 30 --max-workers 8 \
  --out runs/wide_formal/traces

# Phase 3: evaluator-only gold join and declared qualification.
python -m lhmsb.datasets attach-wide-traces \
  --input runs/external/AutoResearchBench.jsonl \
  --traces runs/wide_formal/traces/traces.jsonl \
  --min-gold-observed 1 \
  --out runs/wide_formal/wide-qualified.jsonl

# Real agent run; configs/wide_research.yaml pins the provider and generation profile.
python -m lhmsb.pilot \
  --config configs/wide_research.yaml \
  --wide-input runs/wide_formal/wide-qualified.jsonl \
  --out runs/wide_research
```

The metadata snapshot used for the formal run is pinned to commit `0a7bddb5ae22d0358560d09e55627d4f86f3743a`; the generated index manifest records every shard's SHA-256. The run freezes the imported data, verifies checksums, replays all three conditions on identical episodes, and writes the scorecard under `runs/wide_research/native/`. The online arXiv backend remains available for small pilots, but is not the formal trace source.

## Installation

Requires Python 3.11 or later. Clone the repository and install in editable mode:

```bash
pip install -e ".[dev]"
```

Legacy per-adapter extras (install only when maintaining those adapters):

```bash
pip install -e ".[chroma]"       # ChromaDB vector baseline
pip install -e ".[mem0]"         # Mem0 hybrid memory
pip install -e ".[graphiti]"     # Graphiti temporal KG (needs a graph DB)
pip install -e ".[letta]"        # Letta self-editing blocks
pip install -e ".[cognee]"       # Cognee triple-store
pip install -e ".[tokenizers]"   # tiktoken for accurate token counting
pip install -e ".[metadata]"     # DuckDB for one-time Parquet metadata indexing
```

Chain extras: `pip install -e ".[dev,chroma,tokenizers]"`.

### Legacy Graphiti prerequisites

The preserved legacy Graphiti adapter requires an externally managed Neo4j or
FalkorDB service. It is not part of the active matrix or canonical server
workflow. Historical container recipes, if present in old revisions, are not
supported by the current project.

## Quickstart

All commands assume the repository root as working directory and an activated virtual environment with the package installed (`pip install -e ".[dev,chroma,tokenizers]"`).

### Native multisystem qualification

The first repaired long-horizon qualification slice is frozen separately from
the legacy v1 pilot. The native server entry points default to
`configs/experiments/systems_controlled_gpt_only_aaai.yaml`, which runs GPT-5.6 Sol
over workspace-only, full-context, oracle-current-state, flat retrieval, Mem0,
A-MEM, and MemOS. Native memory writers may use the fixed DeepSeek profile, but
it is not a continuation/policy comparison model.
The historical three-policy Mem0 run remains documented at
`configs/experiments/mem0_controlled_zen.yaml` and is excluded from this
GPT-only pilot (`workspace_only`, `oracle_current_state`, and
`mem0_controlled`; it excludes `mem0_native`).
The qualification records the full
`stored → retrieved → visible → causal use → behavior` chain and emits
programmatic state-evolution and behavioral-drift metrics.

On a clean Linux server with native Python virtual environments, host services,
Slurm, and at least two visible NVIDIA GPUs, run the live preflight, smoke, and
qualification. These live tests run on the server, not this workstation:

```bash
sudo install -d -m 0750 -o "$(id -un)" -g "$(id -gn)" /data/lhmsb
cp .env.example .env
chmod 600 .env
# Fill SHENGSUANYUN_API_KEY and DEEPSEEK_API_KEY.

scripts/bootstrap_systems_server.sh --data-root /data/lhmsb --env-file .env
scripts/preflight_systems.sh --data-root /data/lhmsb --env-file .env
scripts/run_systems_smoke.sh --data-root /data/lhmsb --env-file .env
scripts/run_systems_qualification.sh \
  --data-root /data/lhmsb --env-file .env \
  --run-name "gpt-only-q1-$(git rev-parse --short HEAD)"
```

See [the native systems server workflow](docs/systems-server-workflow.md) for
the pinned matrix, directory contract, Slurm commands, resume procedure,
outputs, and metric mapping.

### Smoke Run (offline, ~30 seconds)

Runs 4 offline conditions (`no_memory`, `chroma`, `fake_perfect`, `fake_bad`) over 1 episode per family with a deterministic stub agent. No network, no paid APIs, no live backends.

```bash
python -m lhmsb.pilot --smoke --config configs/pilot.yaml --out runs/smoke
```

Outputs land under `runs/smoke/native/`:

| File | Description |
|------|-------------|
| `scorecard.md` | Human-readable scorecard with ROI, task score, drift index, and storage/retrieval efficiency breakdown. |
| `scorecard.json` | Machine-readable scorecard. |
| `pareto_*.png` | Pareto-frontier plots: native overall, per-family. |
| `run_manifest.json` | Full reproducibility manifest: git SHA, config hash, dataset checksums, environment snapshot. |

### Legacy v1 full pilot (deferred)

This older command can run all six v1 adapters, but it is not part of the
current experimental plan and must not be used for the Mem0 qualification.
Cross-system execution remains deferred pending a new system-selection decision.

```bash
# Native track (each system's own defaults)
python -m lhmsb.pilot --config configs/pilot.yaml --out runs/pilot

# Controlled track (pinned internal model across systems, where supported)
python -m lhmsb.pilot --config configs/pilot.yaml --out runs/pilot --track controlled
```

Native and controlled outputs are written to separate subdirectories (`runs/pilot/native/` and `runs/pilot/controlled/`) and never merged.

### Dataset Verification

Verify that frozen datasets have not been tampered with:

```bash
python -m lhmsb.datasets verify --frozen runs/smoke/native/datasets/research
```

Generate the supplementary matched-construct release independently of the
v0.10 ranking data:

```bash
python -m lhmsb.datasets generate-mem0-stateful \
  --seeds 101 102 103 --n-episodes 1 --n-sessions 16 \
  --construct-mode matched_triplets --steps-per-session 16 \
  --out runs/matched_v011_stage
python -m lhmsb.datasets freeze-mem0-stateful \
  --src runs/matched_v011_stage --out runs/matched_v011
python -m lhmsb.datasets verify-mem0-stateful \
  --frozen runs/matched_v011
python -m lhmsb.datasets regen-check-mem0-stateful \
  --frozen runs/matched_v011
```

Each requested episode in `matched_triplets` mode denotes one counterfactual
group and produces three physical episodes. These outputs support the mechanism
analysis and must not be pooled with v0.10 backend-ranking estimates.
The server run uses
`configs/experiments/systems_controlled_gpt_only_matched_v011.yaml`; planning
rejects release mismatches and incomplete triplets, and records the
counterfactual group, rather than the physical member, as the primary analysis unit.
Use `LHMSB_ANALYSIS_PHASE=calibration` for this three-group release; a
30-group disjoint release must use `LHMSB_ANALYSIS_PHASE=confirmatory`.

Generate the supplementary same-decision horizon-dose release separately:

```bash
python -m lhmsb.datasets generate-mem0-stateful \
  --seeds 201 202 203 --n-episodes 1 --n-sessions 16 \
  --construct-mode horizon_panels --horizon-sessions 4 8 16 \
  --steps-per-session 16 --out runs/horizon_v012_stage
python -m lhmsb.datasets freeze-mem0-stateful \
  --src runs/horizon_v012_stage --out runs/horizon_v012
python -m lhmsb.datasets verify-mem0-stateful \
  --frozen runs/horizon_v012
python -m lhmsb.datasets regen-check-mem0-stateful \
  --frozen runs/horizon_v012
```

Each requested `horizon_panels` episode is one statistical panel containing
short/medium/long × static/evolution/conflict (9 dependent physical members).
The release is `software-matched-horizon-panels-v0.12.0` and uses
`configs/experiments/systems_controlled_gpt_only_horizon_v012.yaml`. Planning
rejects any episode limit that splits a nine-member panel. Reports suppress
generic episode inference and within-panel triplet inference; the only
inferential artifact for the horizon diagnostic is
`horizon_panel_statistics.json`, which resamples complete panels. Three panels
are the calibration floor; the current panel changes effective transitions,
dependency depth, and session handoffs jointly and must not be described as a
pure handoff experiment.

## Reproducibility

Every run records a `run_manifest.json` with the full configuration hash, git SHA, dataset checksums, and environment snapshot. Frozen datasets are checksummed per file and regeneratable from their declared seeds. The sparse judge is pinned as `lordx64/Qwable-v1` by revision hash in `configs/pilot.yaml`. The smoke run uses a deterministic stub agent and a zero-clock so two runs produce byte-identical `scorecard.json`.

## Documentation

| Document | Purpose |
|----------|---------|
| `spec/01-overview.md` | Formal scope, contributions, and glossary. |
| `spec/02-metrics.md` | Legacy v1 Memory ROI metrics (not the active paper scorecard). |
| `spec/03-protocol.md` | Legacy v1 factual-probe replay protocol. |
| `spec/04-datasets.md` | Legacy v1 task-family contract. |
| `spec/05-systems.md` | Preserved adapter interface and graceful degradation. |
| `docs/extending.md` | How to add new adapters, task families, and deferred dimensions. |
| `docs/long-horizon-benchmark-contract.md` | Operational definition, contribution/evidence contract, claim boundary, and related-work positioning. |
| `docs/systems-server-workflow.md` | Native-venv/service/Slurm execution, resume, validation, and result collection. |

## License

MIT.

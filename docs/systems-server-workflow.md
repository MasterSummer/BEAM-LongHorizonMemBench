# Multisystem native server workflow

This is the only current server workflow for the schema-v2 controlled track.
It uses host Python 3.11 virtual environments, loopback Qdrant/Neo4j/TEI
processes, and Slurm. The benchmark never substitutes another process
isolation runtime.

The run contains Workspace-only, Full-context, Oracle-current-state, Flat
retrieval, Mem0, official A-MEM, and MemOS-Tree. The current repaired pilot uses
GPT-5.6 Sol as the only continuation/policy model through ShengSuanYun; native
memory systems may use DeepSeek for the fixed writer. Provider keys
are read only by the policy process and never enter service environments,
manifests, or result hashes.

The tracked policy profile pins `https://router.shengsuanyun.com/api/v1` and
model ID `openai/gpt-5.6-sol`; the operator file supplies only
`SHENGSUANYUN_API_KEY`. Before a new run, the live preflight must confirm that
exact model. If it is unavailable, select the exact 5.5 ID returned by
`GET /models` in a new tracked profile and generate a new run identity—never
fall back between models inside one run.

## Prepare the server

Install native Qdrant, Neo4j Community, Java 17, and the CUDA TEI binary on the
host. Download the BGE-M3 and BGE-reranker-v2-m3 snapshots into the configured
model directories. The canonical deployment needs two visible, distinct NVIDIA
GPUs selected in the operator environment; on the current zyd host, GPU 0 is
assigned to embeddings and GPU 1 to reranking. Set `LHMSB_REQUIRE_A100=1`
only when reproducing a legacy A100-only deployment. Copy the repository and
create a mode-0600 operator file:

```bash
cp .env.example /data/lhmsb/env/operator.env
chmod 600 /data/lhmsb/env/operator.env
export LHMSB_DATA_ROOT=/data/lhmsb
export LHMSB_ENV_FILE=/data/lhmsb/env/operator.env
scripts/bootstrap_systems_server.sh --allow-dirty
scripts/verify_system_runtime.sh
```

Bootstrap checks out the pinned A-MEM and MemOS sources, creates four isolated
Python environments, generates hash-locked requirements and wheelhouses, and
writes runtime and source manifests. It does not read provider keys during
dependency or runtime installation.

`manifests/system-sources.json` records each upstream origin, commit, Git tree,
and the complete non-cache source file set. Verification also imports the live
A-MEM and MemOS modules and requires their `__file__` paths to reside in those
pristine checkouts. An exported directory, a manually injected commit marker,
or an editable install pointing at another user's checkout is rejected.

## Validate and run

Freeze a development calibration candidate first. It contains 50 episodes so
the factorial dataset audit runs, but only seeds 0--4 may be evaluated during
calibration; seeds 5--49 in this directory are never used as confirmatory data:

```bash
SEEDS=$(seq 0 49)
python -m lhmsb.datasets generate-mem0-stateful \
  --seeds ${SEEDS} --n-episodes 1 --n-sessions 16 \
  --out /data/lhmsb/datasets/software_v10_calibration.stage
python -m lhmsb.datasets freeze-mem0-stateful \
  --src /data/lhmsb/datasets/software_v10_calibration.stage \
  --out /data/lhmsb/datasets/software_v10_calibration
python -m lhmsb.datasets verify-mem0-stateful \
  --frozen /data/lhmsb/datasets/software_v10_calibration
python -m lhmsb.datasets regen-check-mem0-stateful \
  --frozen /data/lhmsb/datasets/software_v10_calibration
```

The manifest should report 50 episodes, 16 sessions, and release
`software-vertical-mem0-v0.10.0`. Do not run the pilot from a dirty checkout.
Frozen v0.8/v0.9 datasets and their reports remain immutable calibration
artifacts. They may be re-analysed with compatible stored traces, but must not be
silently relabelled as v0.10 or combined with a v0.10 confirmatory run.

Every wrapper supports a side-effect-free dry run:

```bash
scripts/bootstrap_systems_server.sh --dry-run --data-root /tmp/lhmsb
scripts/preflight_systems.sh --dry-run --data-root /tmp/lhmsb
scripts/run_systems_smoke.sh --dry-run --data-root /tmp/lhmsb
scripts/run_systems_qualification.sh --dry-run --data-root /tmp/lhmsb
```

Run the five-scenario calibration before the confirmatory matrix:

```bash
export LHMSB_DATA_ROOT="${LHMSB_DATA_ROOT:-/home/zyd/lhmsb-native-data}"
export LHMSB_ENV_FILE="${LHMSB_ENV_FILE:-${LHMSB_DATA_ROOT}/env/operator.env}"
set -a
source "${LHMSB_ENV_FILE}"
set +a
export LHMSB_SYSTEM_DATASET="${LHMSB_DATA_ROOT}/datasets/software_v10_calibration"
scripts/run_systems_qualification.sh \
  --data-root "${LHMSB_DATA_ROOT}" \
  --env-file "${LHMSB_ENV_FILE}" \
  --run-name systems-gpt56-shengsuanyun-calibration-v10 \
  --episode-limit 5 --analysis-phase calibration --keep-going
```

Only proceed to the unbounded 50-episode command after the calibration report
passes all measurement-readiness gates. Then freeze the disjoint confirmatory
release from seeds 5--54. Do not modify the generator, checker, prompts, gates,
or analysis after inspecting any confirmatory result:

```bash
unset LHMSB_SYSTEM_DATASET
SEEDS=$(seq 5 54)
python -m lhmsb.datasets generate-mem0-stateful \
  --seeds ${SEEDS} --n-episodes 1 --n-sessions 16 \
  --out /data/lhmsb/datasets/software_v10.stage
python -m lhmsb.datasets freeze-mem0-stateful \
  --src /data/lhmsb/datasets/software_v10.stage \
  --out /data/lhmsb/datasets/software_v10
python -m lhmsb.datasets verify-mem0-stateful \
  --frozen /data/lhmsb/datasets/software_v10
python -m lhmsb.datasets regen-check-mem0-stateful \
  --frozen /data/lhmsb/datasets/software_v10
export LHMSB_ANALYSIS_PHASE=confirmatory
```

Run the repository/runtime/service gate first. Services receive a unique job
instance, bind to `127.0.0.1`, use per-run Qdrant and Neo4j state, and record
PID start times for safe cleanup:

```bash
scripts/preflight_systems.sh
sbatch --export=ALL,LHMSB_SLURM_MODE=smoke \
  deploy/slurm/systems_qualification.sbatch
```

The one-episode smoke must produce four prefix artifacts and a valid report.
After inspection, submit the 16-session qualification:

```bash
PREP_JOB=$(sbatch --parsable \
  --output="${LHMSB_DATA_ROOT}/logs/slurm-prepare-%j.out" \
  --error="${LHMSB_DATA_ROOT}/logs/slurm-prepare-%j.err" \
  --export=ALL,LHMSB_SLURM_MODE=prepare,LHMSB_RUN_NAME=gpt-only-shengsuanyun-v10,LHMSB_SYSTEM_DATASET="${LHMSB_DATA_ROOT}/datasets/software_v10" \
  deploy/slurm/systems_qualification.sbatch)
sbatch --array=0-349%16 --dependency="afterok:${PREP_JOB}" \
  --output="${LHMSB_DATA_ROOT}/logs/slurm-eval-%A_%a.out" \
  --error="${LHMSB_DATA_ROOT}/logs/slurm-eval-%A_%a.err" \
  --export=ALL,LHMSB_RUN_NAME=gpt-only-shengsuanyun-v10 \
  deploy/slurm/systems_evaluate_task.sbatch
"${LHMSB_DATA_ROOT}/venvs/core/bin/python" -m lhmsb.qualification \
  aggregate-systems --run-dir "${LHMSB_DATA_ROOT}/runs/systems/gpt-only-shengsuanyun-v10" \
  --out "${LHMSB_DATA_ROOT}/runs/systems/gpt-only-shengsuanyun-v10/report"
"${LHMSB_DATA_ROOT}/venvs/core/bin/python" -m lhmsb.qualification \
  validate-systems --report "${LHMSB_DATA_ROOT}/runs/systems/gpt-only-shengsuanyun-v10/report" \
  --json "${LHMSB_DATA_ROOT}/runs/systems/gpt-only-shengsuanyun-v10/validation.json"
```

The preparation job requests two generic NVIDIA GPUs, assigns one to embedding
and one to reranking, serializes run names with a filesystem lock, starts native
services, and tears them down after prefixes are frozen. The preparation stage
creates 200 immutable prefix artifacts (four memory backends for each of 50
episodes). The 350 read-only GPT evaluation tasks run as a Slurm array and do
not start memory services. Prefixes
and task results remain on disk so a failed array cell can be retried with the
same run identity.

## Run the v0.11 matched mechanism experiment

The v0.10 commands above estimate descriptive system performance over ordinary
trajectory episodes. They do not identify the effect of state evolution or
hierarchical conflict. That mechanism claim uses the independent v0.11 release
and the tracked configuration
`configs/experiments/systems_controlled_gpt_only_matched_v011.yaml`.

First freeze three calibration groups. Each group produces exactly three
physical members (`static`, `evolution`, and `hierarchical_conflict`), so this
directory contains 9 physical episodes but only 3 statistical units:

```bash
python -m lhmsb.datasets generate-mem0-stateful \
  --seeds 101 102 103 --n-episodes 1 --n-sessions 16 \
  --construct-mode matched_triplets --steps-per-session 16 \
  --out "${LHMSB_DATA_ROOT}/datasets/software_matched_v011_calibration.stage"
python -m lhmsb.datasets freeze-mem0-stateful \
  --src "${LHMSB_DATA_ROOT}/datasets/software_matched_v011_calibration.stage" \
  --out "${LHMSB_DATA_ROOT}/datasets/software_matched_v011_calibration"
python -m lhmsb.datasets verify-mem0-stateful \
  --frozen "${LHMSB_DATA_ROOT}/datasets/software_matched_v011_calibration"
python -m lhmsb.datasets regen-check-mem0-stateful \
  --frozen "${LHMSB_DATA_ROOT}/datasets/software_matched_v011_calibration"
```

Select the matched dataset and config explicitly. `plan-systems` verifies that
the manifest release is `software-matched-constructs-v0.11.0` before any native
write or API call. It also rejects a physical-episode limit that splits a
triplet. For a one-group diagnostic use `--episode-limit 3`, never
`--episode-limit 1`.

Planning also writes `experiment_design_audit.json` and includes its canonical
hash in `run_identity`. The three-group calibration must report
`audit_status=ready_for_calibration` and
`balanced_mechanism_design_ready=true`. A one-group diagnostic is allowed to
report `diagnostic_only`, but it cannot support C1's balanced matched-mechanism
claim. Workers deterministically recompute the audit before loading the
immutable contract. The audit also freezes separate C1--C3 analysis contracts:
C1's workspace-adjusted matched estimands, C2's release-appropriate temporal
scope, and C3's ordered trace plus repeat-stable neutral-replacement
intervention. A report is not confirmatory merely because a later aggregator
can compute a metric that was absent from these pre-call contracts.

```bash
export LHMSB_SYSTEM_CONFIG="${PWD}/configs/experiments/systems_controlled_gpt_only_matched_v011.yaml"
export LHMSB_SYSTEM_DATASET="${LHMSB_DATA_ROOT}/datasets/software_matched_v011_calibration"
export LHMSB_RUN_NAME="gpt-only-shengsuanyun-matched-v011-calibration"
export LHMSB_ANALYSIS_PHASE=calibration

scripts/preflight_systems.sh
sbatch --wait \
  --output="${LHMSB_DATA_ROOT}/logs/slurm-matched-prepare-%j.out" \
  --error="${LHMSB_DATA_ROOT}/logs/slurm-matched-prepare-%j.err" \
  --export=ALL,LHMSB_SLURM_MODE=prepare,LHMSB_RUN_NAME="${LHMSB_RUN_NAME}",LHMSB_SYSTEM_DATASET="${LHMSB_SYSTEM_DATASET}",LHMSB_SYSTEM_CONFIG="${LHMSB_SYSTEM_CONFIG}" \
  deploy/slurm/systems_qualification.sbatch

RUN_DIR="${LHMSB_DATA_ROOT}/runs/systems/${LHMSB_RUN_NAME}"
TASKS=$(jq -r '.evaluation_task_count' "${RUN_DIR}/run_manifest.json")
test "${TASKS}" -eq 63
LAST_TASK=$((TASKS - 1))
EVAL_JOB=$(sbatch --parsable --array="0-${LAST_TASK}%16" \
  --output="${LHMSB_DATA_ROOT}/logs/slurm-matched-eval-%A_%a.out" \
  --error="${LHMSB_DATA_ROOT}/logs/slurm-matched-eval-%A_%a.err" \
  --export=ALL,LHMSB_RUN_NAME="${LHMSB_RUN_NAME}" \
  deploy/slurm/systems_evaluate_task.sbatch)
sbatch --wait --dependency="afterok:${EVAL_JOB}" \
  --output="${LHMSB_DATA_ROOT}/logs/slurm-matched-report-%j.out" \
  --error="${LHMSB_DATA_ROOT}/logs/slurm-matched-report-%j.err" \
  --export=ALL,LHMSB_RUN_NAME="${LHMSB_RUN_NAME}" \
  deploy/slurm/systems_report.sbatch
```

The calibration is an engineering and measurement gate, not an inferential
result. Confirm all of the following before freezing the confirmatory split:

- `validation.json` has `ok: true` and every planned task is complete;
- `run_manifest.json` records `primary_analysis_unit=counterfactual_group`,
  `physical_episode_count=9`, `n_statistical_units=3`, and
  `analysis_phase=calibration`;
- matched structural-invariance, gold-action/option balance, outcome
  completeness, workspace-recoverability balance, matched workspace/oracle
  action separation, workspace adjustment, per-policy full-context/oracle
  terminal-contract solvability, lifecycle provenance, attribution coverage,
  oracle, and sham stability gates pass;
- `matched_construct_statistics.json` reports group-level estimates and never
  treats the three physical members as independent samples.
- the primary mechanism rows
  `state_evolution_penalty_excess_over_workspace` and
  `hierarchical_conflict_penalty_excess_over_workspace` are present; they are
  matched difference-in-differences and must not be replaced by raw construct
  penalties in the paper's headline analysis.
- `contribution_evidence.json` marks C1--C3 only at the scope supported by the
  artifacts. In particular, the v0.11 C2 scope must remain
  `endpoint_violation_only`; it cannot be promoted to longitudinal drift onset.
- `fault_profile_divergence.json` contains at least one aligned
  outcome-equivalent pair for C3, and its pair rows remain descriptive rather
  than being counted as independent samples.

The standard per-episode workspace/action-separation and drift-separation gates
are intentionally not applicable to the one-decision physical members in the
matched release. Their matched counterparts operate on the complete
counterfactual group. A not-applicable standard gate is therefore not a waiver:
the group-level replacement gate must pass.

After the calibration passes, freeze a disjoint 30-group confirmatory release.
It contains 90 physical episodes and 630 evaluation tasks. Do not change the
generator, prompt, checker, config, gates, or estimands after inspecting these
results:

```bash
SEEDS=$(seq 1001 1030)
python -m lhmsb.datasets generate-mem0-stateful \
  --seeds ${SEEDS} --n-episodes 1 --n-sessions 16 \
  --construct-mode matched_triplets --steps-per-session 16 \
  --out "${LHMSB_DATA_ROOT}/datasets/software_matched_v011.stage"
python -m lhmsb.datasets freeze-mem0-stateful \
  --src "${LHMSB_DATA_ROOT}/datasets/software_matched_v011.stage" \
  --out "${LHMSB_DATA_ROOT}/datasets/software_matched_v011"
python -m lhmsb.datasets verify-mem0-stateful \
  --frozen "${LHMSB_DATA_ROOT}/datasets/software_matched_v011"
python -m lhmsb.datasets regen-check-mem0-stateful \
  --frozen "${LHMSB_DATA_ROOT}/datasets/software_matched_v011"

export LHMSB_SYSTEM_DATASET="${LHMSB_DATA_ROOT}/datasets/software_matched_v011"
export LHMSB_RUN_NAME="gpt-only-shengsuanyun-matched-v011-confirmatory"
export LHMSB_ANALYSIS_PHASE=confirmatory
```

Repeat the dynamic prepare/array/report sequence above and require
`evaluation_task_count=630`. The primary statistics are the paired evolution
and hierarchical-conflict penalties relative to static **after subtracting the
matched workspace-only penalty**. Raw paired penalties are secondary. Endpoint
drift columns are drift-compatible violation excesses, not longitudinal onset.
v0.10 and v0.11 estimates are separate evidence streams and must never be
pooled.

`plan-systems` freezes these roles in
`experiment_design_audit.analysis_contract` before any native writer or policy
call. Workers recompute the audit, and report validation rejects changed
primary estimands, analysis units, workspace adjustment, multiplicity scope, or
drift scope even when a file hash is manually updated.

The planner rejects a phase label that is inconsistent with the selected
statistical-unit count: matched calibration/confirmatory runs require at least
3/30 groups, and standard calibration/confirmatory runs require at least 5/50
episodes. Passing this check only grants phase-label and scale eligibility;
confirmatory interpretation still requires a disjoint frozen split,
pre-specified estimands, passing measurement gates, and the reported
uncertainty analysis. The report validator reapplies the same shared contract
and rejects an undersized phase even if report generation bypassed the planner.

The v0.11 config retains state-targeted leave-one-out and replacement probes
needed for causal-use localization, but disables the unrelated +1/+5/+20
memory-count load experiment. A blank v0.11 memory-count scorecard is therefore
expected rather than a missing measurement failure.

## Run the v0.12 same-decision horizon diagnostic

v0.12 is a supplementary construct-validity experiment, not a replacement for
the v0.10 system ranking or v0.11 mechanism estimate. One panel contains nine
dependent members: three history constructs at 4, 8, and 16 sessions. Only the
16-session members must cross the 200-effective-transition floor. A panel is
the analysis and uncertainty unit; neither its nine physical members nor its
three within-dose triplets may be used as independent samples.

Freeze three calibration panels (27 physical episodes, 189 evaluation tasks):

```bash
python -m lhmsb.datasets generate-mem0-stateful \
  --seeds 201 202 203 --n-episodes 1 --n-sessions 16 \
  --construct-mode horizon_panels --horizon-sessions 4 8 16 \
  --steps-per-session 16 \
  --out "${LHMSB_DATA_ROOT}/datasets/software_horizon_v012_calibration.stage"
python -m lhmsb.datasets freeze-mem0-stateful \
  --src "${LHMSB_DATA_ROOT}/datasets/software_horizon_v012_calibration.stage" \
  --out "${LHMSB_DATA_ROOT}/datasets/software_horizon_v012_calibration"
python -m lhmsb.datasets verify-mem0-stateful \
  --frozen "${LHMSB_DATA_ROOT}/datasets/software_horizon_v012_calibration"
python -m lhmsb.datasets regen-check-mem0-stateful \
  --frozen "${LHMSB_DATA_ROOT}/datasets/software_horizon_v012_calibration"

export LHMSB_SYSTEM_CONFIG="${PWD}/configs/experiments/systems_controlled_gpt_only_horizon_v012.yaml"
export LHMSB_SYSTEM_DATASET="${LHMSB_DATA_ROOT}/datasets/software_horizon_v012_calibration"
export LHMSB_RUN_NAME="gpt-only-shengsuanyun-horizon-v012-calibration"
export LHMSB_ANALYSIS_PHASE=calibration
```

Run `scripts/preflight_systems.sh`, then the same prepare → dynamic Slurm array
→ report sequence used above. Read `evaluation_task_count` from the immutable
run manifest; it must be 189 for this calibration. Planning fails before API
work if a limit splits a nine-member panel or if the release/config identities
differ.

The report must satisfy all of the following:

- `primary_analysis_unit=horizon_panel`, `physical_episode_count=27`, and
  `n_statistical_units=3`;
- `experiment_design_audit.analysis_contract` fixes short as reference, long as
  target, workspace-only as the matched control, and the two workspace-adjusted
  horizon-amplification estimands;
- `horizon_panel_statistics.json` reports `n_panels=3` per complete cell and
  uses panel bootstrap/sign flips;
- generic `statistics.json` has
  `status=suppressed_dependent_physical_members`, and
  `matched_construct_statistics.json` has
  `status=suppressed_within_panel_triplets`;
- `validation.json` is `ok: true`, all 189 tasks are complete, and dataset
  verify/regen-check pass.

The panel identifies a joint effective-transition/dependency/handoff dose. Do
not label it a pure session-handoff effect. A positive primary estimate means
the construct penalty grows from short to long more for the memory condition
than for workspace-only; a null or negative result narrows the long-horizon-
specific claim and must be retained.

## Run the v0.14 longitudinal drift and localization experiment

v0.14 is the independent multi-checkpoint release for C2 and C3. It does not
replace v0.11's matched C1 mechanism estimate or v0.12's horizon-dose
diagnostic. Each 16-session episode has 18 dependent continuation decisions,
including a final same-lineage recovery reminder. The episode is the
statistical unit; the 18 SCEUs are repeated observations and must never inflate
the sample size.

Freeze five independent calibration episodes:

```bash
python -m lhmsb.datasets generate-mem0-stateful \
  --seeds 301 302 303 304 305 --n-episodes 1 --n-sessions 16 \
  --construct-mode longitudinal_trajectories --steps-per-session 16 \
  --out "${LHMSB_DATA_ROOT}/datasets/software_longitudinal_v014_calibration.stage"
python -m lhmsb.datasets freeze-mem0-stateful \
  --src "${LHMSB_DATA_ROOT}/datasets/software_longitudinal_v014_calibration.stage" \
  --out "${LHMSB_DATA_ROOT}/datasets/software_longitudinal_v014_calibration"
python -m lhmsb.datasets verify-mem0-stateful \
  --frozen "${LHMSB_DATA_ROOT}/datasets/software_longitudinal_v014_calibration"
python -m lhmsb.datasets regen-check-mem0-stateful \
  --frozen "${LHMSB_DATA_ROOT}/datasets/software_longitudinal_v014_calibration"

export LHMSB_SYSTEM_CONFIG="${PWD}/configs/experiments/systems_controlled_gpt_only_longitudinal_v014_shengsuanyun_writer.yaml"
export LHMSB_SYSTEM_DATASET="${LHMSB_DATA_ROOT}/datasets/software_longitudinal_v014_calibration"
export LHMSB_RUN_NAME="gpt-only-shengsuanyun-longitudinal-v014-calibration"
export LHMSB_ANALYSIS_PHASE=calibration
```

Run the same preflight → prepare → dynamic Slurm array → report sequence. The
calibration manifest must record
`construct_mode=longitudinal_trajectories`,
`primary_analysis_unit=episode`, `physical_episode_count=5`,
`n_statistical_units=5`, 20 preparation tasks, and 35 evaluation tasks.
Planning must fail before API calls unless all of the following pass:

- every episode crosses the 200-step effective causal-span floor and passes
  semantic-effect anti-padding verification;
- every canonical drift category has a same-lineage adherence-capable anchor,
  a later ordinary challenge, and a still-later update/reminder recovery point;
- checker-positive and checker-negative actions exist for every drift category;
- every memory-reliant SCEU has a current action-relevant intervention target;
- no single fixed action solves more than 60% of frozen opportunities;
- the release/config identity is
  `software-longitudinal-trajectories-v0.14.0`.

After measurement gates pass, freeze a disjoint 50-episode confirmatory split:

```bash
SEEDS=$(seq 3001 3050)
python -m lhmsb.datasets generate-mem0-stateful \
  --seeds ${SEEDS} --n-episodes 1 --n-sessions 16 \
  --construct-mode longitudinal_trajectories --steps-per-session 16 \
  --out "${LHMSB_DATA_ROOT}/datasets/software_longitudinal_v014.stage"
python -m lhmsb.datasets freeze-mem0-stateful \
  --src "${LHMSB_DATA_ROOT}/datasets/software_longitudinal_v014.stage" \
  --out "${LHMSB_DATA_ROOT}/datasets/software_longitudinal_v014"
python -m lhmsb.datasets verify-mem0-stateful \
  --frozen "${LHMSB_DATA_ROOT}/datasets/software_longitudinal_v014"
python -m lhmsb.datasets regen-check-mem0-stateful \
  --frozen "${LHMSB_DATA_ROOT}/datasets/software_longitudinal_v014"

export LHMSB_SYSTEM_DATASET="${LHMSB_DATA_ROOT}/datasets/software_longitudinal_v014"
export LHMSB_RUN_NAME="gpt-only-shengsuanyun-longitudinal-v014-confirmatory"
export LHMSB_ANALYSIS_PHASE=confirmatory
```

The confirmatory plan has 50 statistical units, 200 preparation tasks, and 350
evaluation tasks. Primary C2 analysis is adherence-anchored onset, drift-free
survival, persistence, and recovery by state lineage with oracle/full-context
contamination gates. C3 retains the same-decision storage → retrieval → visible
→ intervention → behavior contract. Design readiness alone is not a positive
drift result, and v0.14 does not identify C1's matched state-evolution penalty
or short-to-long amplification.

## Expected outputs

```text
/data/lhmsb/
  manifests/{build,host,native-runtime,model-bundle,system-sources}.json
  sources/{amem,memos}/
  venvs/{core,mem0,amem,memos}/
  services/<run-instance>/
  runs/systems/<run-name>/
    run_manifest.json
    prepare_tasks.jsonl
    evaluation_task_templates.jsonl
    experiment_design_audit.json
    tasks.jsonl
    prefixes/
    results/
    report/{metrics.json,metrics_by_cell.json,scorecard.csv,storage_scorecard.csv,
            memory_count_scorecard.csv,failure_attribution_scorecard.csv,
            decision_attribution.jsonl,long_horizon_scorecard.csv,
            long_horizon_control_contrasts.csv,long_horizon_constructs.jsonl,
            matched_construct_contrasts.jsonl,matched_construct_scorecard.csv,
            matched_construct_statistics.json,drift_trajectories.json,
            fault_profile_divergence.json,fault_profile_divergence.md,
            statistics.json,measurement_gates.json,
            experiment_design_audit.json,experiment_design_audit.md,
            contribution_evidence.json,
            contribution_evidence.md}
    report/episodes/<episode-id>/{metrics.json,scorecard.csv,
            decision_attribution.jsonl,long_horizon_scorecard.csv,
            drift_trajectories.json,fault_profile_divergence.json,
            fault_profile_divergence.md,summary.json}
    validation.json
```

The evaluator reconstructs `stored → candidate → retrieved → visible →
behavior`. The causal scaling variable is the evaluator-controlled number of
model-visible memory objects (+1/+5/+20 within the same SCEU); native live
object count is reported separately as an observational diagnostic. Tokens,
characters, bytes, calls, and latency remain auxiliary diagnostics. Native and
common-rerank readouts are kept separate.

## Recovery rules

- A failed host/runtime gate stops the run; repair the host manifest first.
- A failed service is cleaned up by PID identity and its logs remain in the
  service instance directory.
- A partial A-MEM or MemOS preparation is discarded and rerun from session
  zero because those upstream stores are not resumable in this benchmark.
- Provider failures leave prefix artifacts untouched. Retry only the missing
  evaluation task after the endpoint/key is repaired.
- Never aggregate a run whose `validation.json` is not `ok: true`.

## Completed-run reaggregation and analysis timing

Policy calls and report aggregation are separate. A completed run keeps its
immutable `results/`, `cells/`, frozen dataset, run manifest, config, and
pre-call `experiment_design_audit.json`, so report bugs can normally be fixed
without another paid policy call. This does **not** authorize changing the
pre-specified scientific claim after seeing results.

Use two outputs:

```text
<run>/report-canonical/        original planned clean commit
<run>/report-posthoc-<commit>/ newer analysis code, explicitly exploratory
```

Canonical procedure:

1. read `code_commit`, `code_dirty`, dataset hash, run identity, analysis phase,
   and design-audit hash from the run manifest;
2. check out the exact clean evaluation commit in a separate checkout;
3. aggregate and validate there without changing the run directory;
4. retain the original design audit byte-for-byte;
5. only then run newer analysis code into a distinct post-hoc directory.

A newer aggregation may be promoted to the primary report only when its
estimand, controls, analysis unit, multiplicity scope, and claim boundary are
already identical to the immutable pre-call contract. Adding conservative
interaction-tier metadata is allowed as a post-hoc scope restriction, but it
may not be backdated as preregistered. Static/evolution/conflict members,
horizon-dose members, prior-adherence checkpoints, lifecycle traces, or causal
interventions missing from the original run cannot be synthesized by report
reaggregation.

Every completed experiment must also receive a separate contribution audit:

```bash
AUDIT_COMMIT=$(git rev-parse --short HEAD)
python -m lhmsb.qualification audit-completed-report \
  --report "<run>/report-canonical" \
  --dataset "<exact-frozen-dataset>" \
  --out "<run>/contribution-audit-${AUDIT_COMMIT}" \
  --analysis-timing post_hoc_scope_audit
```

The command is read-only with respect to the source report and rejects an
output path inside it. It verifies declared source hashes, preserves missing or
inconsistent source timing as such, inventories the current C1--C3 artifacts,
lists failed measurement gates, and writes:

```text
contribution-audit-<commit>/
  completed_report_audit.json
  completed_report_audit.md
  audit_manifest.json
```

The audit may identify a zero-API reaggregation candidate only when raw task,
SCEU, lifecycle, retrieval, and intervention traces are complete **and** the
provided frozen evaluator dataset has the exact manifest hash declared by the
source run. Omitting `--dataset`, supplying a different release, or missing
evaluator SCEUs keeps the candidate false. The audit never establishes effect
direction, rewrites the canonical report, or promotes a new post-hoc estimand
to confirmatory evidence.

When that candidate is true, materialize the C2 state-lineage trajectories and
C3 funnel into another immutable sibling directory:

```bash
python -m lhmsb.qualification reanalyze-completed-report \
  --report "<run>/report-canonical" \
  --dataset "<exact-frozen-dataset>" \
  --out "<run>/c2c3-posthoc-${AUDIT_COMMIT}"
```

This command makes no API or memory-backend calls. It joins evaluator-required
state, checkpoint inventories, native retrieval IDs, final model-visible IDs,
targeted interventions, and checked actions at the same SCEU. It also derives
state lineages from the frozen evaluator graph and requires earlier adherence
on the same lineage before calling a later violation observed drift. It writes
hashed `drift_trajectories.json/.md`, `decision_attribution.jsonl`, failure
scorecards, fault-profile divergence artifacts, and a summary whose analysis
timing is always `post_hoc_exploratory`. Oracle/full-context drift is reported
as control contamination rather than silently attributed to memory. No-effect
interventions are labelled as no detected unique causal effect, not proof of
non-use. The command refuses a mismatched dataset manifest and never rewrites
the canonical report.

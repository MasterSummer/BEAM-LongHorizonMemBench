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
  --out /data/lhmsb/datasets/software_v9_calibration.stage
python -m lhmsb.datasets freeze-mem0-stateful \
  --src /data/lhmsb/datasets/software_v9_calibration.stage \
  --out /data/lhmsb/datasets/software_v9_calibration
python -m lhmsb.datasets verify-mem0-stateful \
  --frozen /data/lhmsb/datasets/software_v9_calibration
python -m lhmsb.datasets regen-check-mem0-stateful \
  --frozen /data/lhmsb/datasets/software_v9_calibration
```

The manifest should report 50 episodes, 16 sessions, and release
`software-vertical-mem0-v0.9.0`. Do not run the pilot from a dirty checkout.

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
export LHMSB_SYSTEM_DATASET="${LHMSB_DATA_ROOT}/datasets/software_v9_calibration"
scripts/run_systems_qualification.sh \
  --data-root "${LHMSB_DATA_ROOT}" \
  --env-file "${LHMSB_ENV_FILE}" \
  --run-name systems-gpt56-shengsuanyun-calibration-v10 \
  --episode-limit 5 --keep-going
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
  --out /data/lhmsb/datasets/software_v9.stage
python -m lhmsb.datasets freeze-mem0-stateful \
  --src /data/lhmsb/datasets/software_v9.stage \
  --out /data/lhmsb/datasets/software_v9
python -m lhmsb.datasets verify-mem0-stateful \
  --frozen /data/lhmsb/datasets/software_v9
python -m lhmsb.datasets regen-check-mem0-stateful \
  --frozen /data/lhmsb/datasets/software_v9
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
  --export=ALL,LHMSB_SLURM_MODE=prepare,LHMSB_RUN_NAME=gpt-only-shengsuanyun-v10,LHMSB_SYSTEM_DATASET="${LHMSB_DATA_ROOT}/datasets/software_v9" \
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
    tasks.jsonl
    prefixes/
    results/
    report/{metrics.json,metrics_by_cell.json,scorecard.csv,storage_scorecard.csv,
            memory_count_scorecard.csv,statistics.json,measurement_gates.json}
    report/episodes/<episode-id>/{metrics.json,scorecard.csv,summary.json}
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

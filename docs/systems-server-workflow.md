# Multisystem native server workflow

This is the only current server workflow for the schema-v2 controlled track.
It uses host Python 3.11 virtual environments, loopback Qdrant/Neo4j/TEI
processes, and Slurm. The benchmark never substitutes another process
isolation runtime.

The run contains Workspace-only, Full-context, Oracle-current-state, Flat
retrieval, Mem0, official A-MEM, and MemOS-Tree. The current repaired pilot uses
GPT-5.6 Sol as the only continuation/policy model through OpenCode Zen; native
memory systems may use DeepSeek for the fixed writer. Provider keys
are read only by the policy process and never enter service environments,
manifests, or result hashes.

## Prepare the server

Install native Qdrant, Neo4j Community, Java 17, and the CUDA TEI binary on the
host. Download the BGE-M3 and BGE-reranker-v2-m3 snapshots into the configured
model directories. The canonical deployment needs two visible, distinct NVIDIA
GPUs; the current server profile assigns GPU 0 to embeddings and GPU 1 to
reranking. Set `LHMSB_REQUIRE_A100=1` only when reproducing a legacy A100-only
deployment. Copy the repository and create a mode-0600 operator file:

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

## Validate and run

Create the repaired GPT-only release before the smoke (the smoke may use a
four-session fixture with the same schema):

```bash
SEEDS=$(seq 0 29)
python -m lhmsb.datasets generate-mem0-stateful \
  --seeds ${SEEDS} --n-episodes 1 --n-sessions 16 \
  --out /data/lhmsb/datasets/software_v3.stage
python -m lhmsb.datasets freeze-mem0-stateful \
  --src /data/lhmsb/datasets/software_v3.stage \
  --out /data/lhmsb/datasets/software_v3
python -m lhmsb.datasets verify-mem0-stateful \
  --frozen /data/lhmsb/datasets/software_v3
python -m lhmsb.datasets regen-check-mem0-stateful \
  --frozen /data/lhmsb/datasets/software_v3
```

The manifest should report 30 episodes, 16 sessions, and release
`software-vertical-mem0-v0.3.0`. Do not run the pilot from a dirty checkout.

Every wrapper supports a side-effect-free dry run:

```bash
scripts/bootstrap_systems_server.sh --dry-run --data-root /tmp/lhmsb
scripts/preflight_systems.sh --dry-run --data-root /tmp/lhmsb
scripts/run_systems_smoke.sh --dry-run --data-root /tmp/lhmsb
scripts/run_systems_qualification.sh --dry-run --data-root /tmp/lhmsb
```

Run the repository/runtime/service gate first. Services receive a unique job
instance, bind to `127.0.0.1`, use per-run Qdrant and Neo4j state, and record
PID start times for safe cleanup:

```bash
scripts/preflight_systems.sh
sbatch --export=ALL,LHMSB_SLURM_MODE=smoke \
  deploy/slurm/systems_qualification.sbatch
```

The four-session smoke must produce four prefix artifacts and a valid report.
After inspection, submit the 16-session qualification:

```bash
scripts/run_systems_qualification.sh --prepare-only \
  --data-root /data/lhmsb --env-file /data/lhmsb/env/operator.env \
  --run-name gpt-only-v3
sbatch --array=0-209%16 \
  --export=ALL,LHMSB_RUN_NAME=gpt-only-v3 \
  deploy/slurm/systems_evaluate_task.sbatch
"${LHMSB_DATA_ROOT}/venvs/core/bin/python" -m lhmsb.qualification \
  aggregate-systems --run-dir /data/lhmsb/runs/systems/gpt-only-v3 \
  --out /data/lhmsb/runs/systems/gpt-only-v3/report
"${LHMSB_DATA_ROOT}/venvs/core/bin/python" -m lhmsb.qualification \
  validate-systems --report /data/lhmsb/runs/systems/gpt-only-v3/report \
  --json /data/lhmsb/runs/systems/gpt-only-v3/validation.json
```

The preparation job requests two generic NVIDIA GPUs, assigns one to embedding
and one to reranking, serializes run names with a filesystem lock, starts native
services, and tears them down after prefixes are frozen. The 210 read-only GPT
evaluation tasks run as a Slurm array and do not start memory services. Prefixes
and task results remain on disk so a failed array cell can be retried with the
same run identity.

## Expected outputs

```text
/data/lhmsb/
  manifests/{build,host,native-runtime}.json
  sources/{amem,memos}/
  venvs/{core,mem0,amem,memos}/
  services/<run-instance>/
  runs/systems/<run-name>/
    run_manifest.json
    preparation_tasks.jsonl
    evaluation_task_templates.jsonl
    tasks.jsonl
    prefixes/
    results/
    report/{metrics.json,metrics_by_cell.json,scorecard.csv}
    validation.json
```

The evaluator reconstructs `stored → candidate → retrieved → visible →
behavior`. Memory-object count is the primary scale variable; tokens,
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

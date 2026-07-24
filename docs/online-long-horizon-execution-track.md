# Online long-horizon execution track

The frozen qualification artifacts are replay-backed diagnostic evaluations;
they are not relabelled as online execution. The online track is a separate but
coupled artifact over the same frozen latent episodes.

## Claim contract

An episode is called `online_long_horizon_agent_execution` only when all of the
following hold:

1. the evaluated policy makes at least 200 decisions (the canonical run uses
   256: 16 sessions × 16 decisions);
2. all non-initial decisions consume the actual output digest of an earlier
   policy-selected work unit;
3. the 256 work units have unique paths and outputs and form complete
   inspect–implement–validate–record cycles;
4. selected JSON artifacts parse, selected Python artifacts compile and
   execute their dependency contract, and a hidden executable checker runs at
   the terminal state;
5. opaque option positions are permuted at each decision, so a fixed option
   position cannot stand for one fixed semantic action;
6. every decision has exactly one evaluator-valid action, no semantic action
   is valid for more than 60% of decisions, and no opaque option position is
   correct more than 40% of the time;
7. the workspace persists while working context is cleared at all 15 session
   boundaries; and
8. the exact requests, responses, memory IDs, workspace hashes, artifact
   hashes, content-dependency edges, code revision, and terminal result are
   retained.

Environment-generated executor events never count toward the 200-policy-call
threshold. A 16-call session-level run is labelled `sparse_closed_loop`.

The model sees the current session surface, the persistent workspace digest,
and a bounded slice containing the current implementation and immediate
dependency artifact. It does not receive evaluator state IDs, validity labels,
future events, or the entire accumulated workspace dump.

The online action set covers four genuinely different continuations: continue
the still-current v1 branch, quarantine v1 during the leakage-to-replacement
transition, continue the authoritative offline v2 branch, or use the explicitly
authorized hosted accelerator for an isolated profiler. A scoped profiler
request applies to its complete inspect–implement–validate–record session
cycle; the owner's authorization must still be recovered from task state. The
temporary quarantine action exists only in the online action surface and does
not mutate the frozen diagnostic release.

## Final matrix

The frozen population contains 50 confirmatory episodes. Before API calls, the
planner deterministically chooses a scenario × workspace-recoverability
balanced 20-episode online subset. Every selected episode is run under:

- workspace-only;
- Flat retrieval;
- Mem0;
- A-MEM;
- MemOS.

For one fixed policy model this is 100 trajectories and 25,600 policy calls.
All four memory conditions retain their native candidate generation but use
the same pinned local BGE reranker and `visible_k` before model exposure. The
trace separately records backend-retrieved IDs and model-visible reranked IDs;
a controlled memory trajectory fails integrity validation if this boundary is
missing.
At each handoff, every backend receives the same public session material and
the model's installed implementation summaries. Flat stores these unchanged as
raw history units; Mem0, A-MEM, and MemOS process them through their own write
lifecycle. Thus Flat is not handicapped by omitting online progress, and native
systems are not given evaluator-only state labels.

Before each retrieval, the runner records the backend's live native
memory-object count. Reports retain checked-score sum, live-count denominator,
mean live count, and `score / cumulative live-object exposure`. This efficiency
number is descriptive across systems because native object granularity differs;
raw score and raw denominator are never hidden.
Full-context and oracle-current-state remain registered diagnostic controls;
they are not multiplied into the expensive 256-decision online matrix.

The matrix planner refuses a dirty checkout, a release/config mismatch, a
non-longitudinal release, fewer than 50 frozen episodes, or episodes other than
16 sessions. It also enumerates all 5,120 decisions in the selected 20
episodes before any API call and refuses an unsolvable, multiply labelled, or
default-dominated design. `MATRIX.json` records the resulting action and opaque
option distributions. Each result identity includes episode, condition,
backend, and policy profile, preventing systems from overwriting one another.

## Native server workflow

The track uses native venvs, loopback Qdrant/Neo4j/TEI services, and Slurm. It
does not use Docker or another container substitute.

First verify the exact routed policy ID. The tracked configuration currently
pins `openai/gpt-5.6-sol`. If `GET /models` does not return that ID, add a new
tracked 5.5 profile using the exact catalog ID and generate a new matrix
identity; never silently fall back within a run.

```bash
scripts/run_online_longhorizon.sh models
```

This command prints only matching model IDs; it does not print the API key.

From a clean server checkout:

```bash
export LHMSB_DATA_ROOT=/home/zyd/lhmsb-native-data
export LHMSB_ENV_FILE="${LHMSB_DATA_ROOT}/env/operator.env"
export LHMSB_SYSTEM_DATASET="${LHMSB_DATA_ROOT}/datasets/software_longitudinal_v014"
export LHMSB_SYSTEM_CONFIG="${PWD}/configs/experiments/systems_controlled_gpt_only_longitudinal_v014_shengsuanyun_writer.yaml"
export LHMSB_ONLINE_RUN_NAME="memhorizon-online-confirmatory-v1"

scripts/run_online_longhorizon.sh plan

ARRAY_JOB=$(sbatch --parsable --array=0-19%1 \
  --export=ALL,LHMSB_ONLINE_RUN_NAME="${LHMSB_ONLINE_RUN_NAME}" \
  deploy/slurm/online_longhorizon_array.sbatch)

sbatch --dependency="afterok:${ARRAY_JOB}" \
  --export=ALL,LHMSB_ONLINE_RUN_NAME="${LHMSB_ONLINE_RUN_NAME}" \
  deploy/slurm/online_longhorizon_report.sbatch
```

`%1` is the canonical setting for one host dedicating GPU 0 to embeddings and
GPU 1 to reranking. Increase array concurrency only when Slurm can allocate an
independent pair of GPUs and sufficient RAM to every simultaneous task.

Every array task starts one isolated native service set and runs the five
conditions for one episode sequentially. Verified clean results are resumed
without another policy call. Different or dirty code revisions are not
silently reused. Attempt-local SQLite, A-MEM, and MemOS state is written under
`runtime/<service-instance>` so a scheduler retry starts empty without changing
the deterministic scientific result identity.

## Outputs and interpretation

```text
${LHMSB_DATA_ROOT}/runs/online/<run-name>/
  MATRIX.json
  tasks.jsonl
  results/
    task-0000/
      <episode>__<condition>__<backend>__<policy>.json
      MANIFEST.json
      report.json
      report.md
      DONE
    ...
    report.json
    report.md
  validation.json
```

Aggregation requires all 100 registered identities, the frozen code commit,
clean result revisions, valid content chains, substantive-work gates, and the
online horizon gate. Low task scores, observed drift, and failed terminal
behavior are scientific outcomes and never cause completed trajectories to be
deleted.

The online track establishes task execution and system-level behavior. Storage
provenance, right/no/wrong memory contrasts, and same-decision
stored→retrieved→visible→behavior localization are produced by the registered
diagnostic track over the same frozen latent episode definitions. Until a clean
real-policy matrix exists, the paper may describe this protocol but may not
state that the evaluated model has completed the online task.

# Multisystem A100 server workflow

This document is the handoff contract for the schema-v2 controlled track. It
covers the four controls/baselines and managed systems:

- Workspace-only, Full-context, Oracle-current-state, and Flat retrieval;
- Mem0 2.0.12, official A-MEM at
  `ceffb860f0712bbae97b184d440df62bc910ca8d`, and MemOS 2.0.23
  TreeTextMemory at `583b07b998afc4debb6c5078439b0b3896f5b097`.

A run uses the three configured policy profiles. Policy requests go through
OpenCode Zen for Opus 4.8 and GPT-5.6 Sol; the fixed writer and DeepSeek policy
route use the official DeepSeek endpoint. Only `OPENCODE_ZEN_API_KEY` and
`DEEPSEEK_API_KEY` are accepted. Do not add OpenAI or Anthropic credentials.

## Local handoff

Use a clean, committed checkout and verify the frozen dataset before copying
anything to the server:

```bash
git fetch origin
git checkout --detach <benchmark-commit>
uv sync --locked
uv run pytest tests/qualification tests/adapters -q
uv run ruff check src tests
uv run mypy src/lhmsb
bash -n scripts/lib/systems_common.sh scripts/bootstrap_systems_server.sh \
  scripts/preflight_systems.sh scripts/run_systems_smoke.sh \
  scripts/run_systems_qualification.sh scripts/verify_system_images.sh \
  deploy/slurm/systems_qualification.sbatch
```

Create the frozen 4-session and 16-session releases using the repository
dataset pipeline. The server must receive the resulting dataset directories,
the source checkout, model snapshots, and the image/wheelhouse bundle; provider
credentials are set only on the server.

## Server bootstrap

The operator creates `/data/lhmsb/.env` from `.env.example`, sets the two
provider keys, the local model revisions, and the resolved image/base digests.
The file must be mode 0600 and is never committed or copied into an image.

```bash
cp .env.example /data/lhmsb/.env
chmod 600 /data/lhmsb/.env
export LHMSB_DATA_ROOT=/data/lhmsb
export LHMSB_ENV_FILE=/data/lhmsb/.env
scripts/bootstrap_systems_server.sh
scripts/verify_system_images.sh
scripts/preflight_systems.sh
```

Bootstrap performs the following in order:

1. creates the persistent data tree;
2. checks the benchmark lock and checks out the exact A-MEM/MemOS commits;
3. exports transitive requirements with hashes and fills the per-system
   wheelhouse;
4. builds the core, Mem0, A-MEM, and MemOS worker images without online
   resolution;
5. pulls Qdrant, Neo4j Community, and TEI only to resolve immutable digests,
   archives them under `/data/lhmsb/images`, and writes
   `/data/lhmsb/manifests/images.json`;
6. records the benchmark commit and source pins before repository-only
   preflight.

If an OCI registry is unavailable after the bundle is copied, use the image
archives and run `scripts/verify_system_images.sh`; never change
`pull_policy: never` or substitute an unpinned image.

## Repository-only and dry-run checks

All wrappers support `--dry-run`. The mode prints commands and performs no
Docker, network, GPU, filesystem, or secret access:

```bash
scripts/bootstrap_systems_server.sh --dry-run --data-root /tmp/lhmsb
scripts/preflight_systems.sh --dry-run --data-root /tmp/lhmsb
scripts/run_systems_smoke.sh --dry-run --data-root /tmp/lhmsb
scripts/run_systems_qualification.sh --dry-run --data-root /tmp/lhmsb
scripts/verify_system_images.sh --dry-run --data-root /tmp/lhmsb
```

## Slurm execution

The batch recipe requests two distinct A100s. The first allocated device hosts
embedding TEI and the second hosts reranker TEI. The Compose project and
Neo4j/Qdrant namespace include the Slurm job ID, so concurrent jobs cannot share
a graph or vector collection.

Run the smoke first:

```bash
sbatch --export=ALL,LHMSB_SLURM_MODE=smoke \
  deploy/slurm/systems_qualification.sbatch
```

The smoke must produce four prefix artifacts, a finalized task table with 21
policy tasks and 30 scored cells, and `validation.json` with `ok: true`.
Inspect the report before the long run:

```bash
jq '.ok,.errors' /data/lhmsb/runs/systems/systems-<job-id>/validation.json
find /data/lhmsb/runs/systems/systems-<job-id> -maxdepth 2 -type f | sort
```

Then submit the 16-session run:

```bash
sbatch --export=ALL,LHMSB_SLURM_MODE=qualification \
  deploy/slurm/systems_qualification.sbatch
```

The qualification is resumable. A completed cell is reused only when its input
and prefix hashes match. Re-running with the same run name after a preemption
continues missing tasks; a changed commit, config, dataset, model, image, or
prefix requires a new run name. Never pass `--force` to a formal result unless
the old run has been archived.

## Expected result tree

```text
/data/lhmsb/
  manifests/
    build.json
    images.json
    host.json
    image-verification.json
  runs/systems/<run-name>/
    run_manifest.json
    preparation_tasks.jsonl
    evaluation_task_templates.jsonl
    tasks.jsonl
    prefixes/
    results/
    report/
      metrics.json
      metrics_by_cell.json
      scorecard.csv
    validation.json
```

Each managed prefix retains the normalized
`stored -> candidate -> retrieved -> visible -> behavior` chain. Native and
common-rerank readouts remain separate. Memory-object count is the primary
scale variable; token/character counts are auxiliary.

## Failure recovery

- **Host/GPU gate:** verify `nvidia-smi` shows at least two A100 devices and
  that `SLURM_JOB_GPUS` contains distinct IDs. No task should be rerun until
  the host manifest passes.
- **Image gate:** restore the archived OCI files, run
  `verify_system_images.sh`, and compare every `sha256:` ID with
  `manifests/images.json`. Do not pull a replacement by tag.
- **Qdrant/Neo4j contamination:** stop the Compose project and assign a new
  `LHMSB_SYSTEM_NAMESPACE`; MemOS preparation requires a fresh Community
  volume.
- **Provider failure:** keep the prefix artifacts, leave the failed cell in
  place, and rerun the same matrix with the same run identity after the
  endpoint/key is repaired. Provider errors must not trigger a memory rewrite.
- **Interrupted A-MEM/MemOS preparation:** discard the incomplete artifact and
  rerun that preparation from session zero. A partial prefix is not executable.
- **Validation failure:** do not aggregate into paper results. Preserve the
  run directory, inspect the first foreign-key/hash error, and rerun only after
  the source of the mismatch is corrected.

Never place keys in command arguments, manifests, logs, Docker build arguments,
or result hashes.


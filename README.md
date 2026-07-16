# LongHorizonMemSysBench v1

A benchmark for AI memory-management systems over long, multi-session agentic tasks.

Modern AI agents operate across days or weeks: a research agent updates a growing body of evidence, a coding agent maintains a codebase through evolving requirements, a personal assistant tracks user preferences across months. Memory management systems promise to give these agents persistent, evolvable recall beyond their context window. **LongHorizonMemSysBench (LHMSB) v1 answers a single deployment question**: does the memory system improve task performance, store selectively, and retrieve the right evidence across sessions?

The headline metric is **Memory ROI**, defined here as normalized task gain per recorded memory item. It is paired with Storage Efficiency and Retrieval Efficiency, measured on the same counterfactual replay episodes.

## What v1 Measures

v1 implements four dimensions with Memory ROI as the cross-cutting headline:

| Dimension | Status | Description |
|-----------|--------|-------------|
| **Dim 2: Goal-Directed Utilization** | Implemented | Does the agent use its memory to improve task completion? Scored via programmatic rubrics plus a sparse LLM judge. |
| **Dim 3: Goal Drift & Behavioral Stability** | Implemented | Does the agent's behavior remain stable across sessions? Measured via programmatic invariants over aligned probes. |
| **Dim 4: Retrieval Quality** | Supporting | Endogenous (agent-initiated) and oracle (fixed-query) retrieval precision and recall. Supports the other dimensions but is not the headline. |
| **Storage Efficiency** | Implemented | Whether required facts are written and explicitly forbidden facts are avoided. |
| **Retrieval Efficiency** | Implemented | Whether relevant facts are returned, distractors are avoided, and useful memories are used in time. |

**Memory ROI** is `mean(normalized_gain) / mean(recorded_memory_count)`, reported with bootstrap confidence intervals. `no_mem` is the counterfactual baseline and zero-memory systems are `N/A`, never infinite.

## Systems Under Test

Six leaderboard conditions plus two calibration-only sensitivity oracles:

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

Per-adapter extras (install only what you need):

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

### Graphiti Prerequisites

Graphiti requires a running graph database (Neo4j or FalkorDB). A Docker Compose file is provided:

```bash
docker compose -f docker/graphiti-compose.yml up -d
```

This starts a Neo4j container on `localhost:7687`. Set `GRAPHITI_NEO4J_URI=bolt://localhost:7687` (or the matching env var) before running with the `graphiti` condition.

## Quickstart

All commands assume the repository root as working directory and an activated virtual environment with the package installed (`pip install -e ".[dev,chroma,tokenizers]"`).

### Mem0 A100 qualification

The first real long-horizon qualification slice is frozen separately from the
legacy v1 pilot. It runs three policy models over workspace, oracle, Controlled
Mem0, and Native Mem0 conditions; records the full
`stored → retrieved → visible → causal use → behavior` chain; and emits
programmatic state-evolution and behavioral-drift metrics.

On a clean Linux server with Docker, the NVIDIA container runtime, and at least
two visible A100 GPUs:

```bash
cp .env.example .env
# Fill ANTHROPIC_API_KEY, DEEPSEEK_API_KEY, and OPENAI_API_KEY.

scripts/bootstrap_server.sh --data-root /data/lhmsb --env-file .env
scripts/preflight_mem0.sh --data-root /data/lhmsb --env-file .env
scripts/run_mem0_smoke.sh --data-root /data/lhmsb --env-file .env
scripts/run_mem0_qualification.sh \
  --data-root /data/lhmsb --env-file .env \
  --run-name "mem0-q1-$(git rev-parse --short HEAD)"
```

See [the Mem0 server workflow](docs/mem0-server-workflow.md) for the pinned
matrix, directory contract, Slurm commands, resume procedure, outputs, and
metric mapping.

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

### Full Pilot Run

Runs all 6 leaderboard conditions at pilot scale (3 seeds, 20 episodes per family). Requires live backends and API keys for `mem0`, `letta`, `graphiti`, and `cognee`.

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

## Reproducibility

Every run records a `run_manifest.json` with the full configuration hash, git SHA, dataset checksums, and environment snapshot. Frozen datasets are checksummed per file and regeneratable from their declared seeds. The sparse judge is pinned as `lordx64/Qwable-v1` by revision hash in `configs/pilot.yaml`. The smoke run uses a deterministic stub agent and a zero-clock so two runs produce byte-identical `scorecard.json`.

## Documentation

| Document | Purpose |
|----------|---------|
| `spec/01-overview.md` | Formal scope, contributions, and glossary. |
| `spec/02-metrics.md` | Metric formulas, edge-case policies, and worked examples. |
| `spec/03-protocol.md` | Counterfactual replay protocol and track separation. |
| `spec/04-datasets.md` | Dataset generation, freezing, and verification contract. |
| `spec/05-systems.md` | Memory system adapter interface and graceful degradation. |
| `docs/extending.md` | How to add new adapters, task families, and deferred dimensions. |
| `docs/mem0-server-workflow.md` | A100 migration, Docker/Slurm execution, resume, validation, and result collection. |

## License

MIT.

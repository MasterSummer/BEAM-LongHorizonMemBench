# 04 — Datasets & Task Families

> **Status**: canonical for v1. Tasks 7, 10, 11, and 20 implement against these
> specifications.

---

## 1. Simulator Design

The shared simulator core (`src/lhmsb/sim/core.py`, Task 7) produces episodes
through a family-agnostic pipeline. Each episode is a fixed, seed-derived
schedule of world events and aligned probes — the agent does NOT mutate the
world in v1 (interactive worlds deferred to v2).

### 1.1 Core Concepts

**Episode**: a self-contained task spanning multiple sessions. Identified by
`episode_id`, tagged with `family` and `seed`. Consists of an ordered list of
`WorldEvent`s and an aligned list of `Probe`s.

**WorldEvent**: a change to the exogenous evidence world at a specific step:
- `inject`: a new fact enters the valid set.
- `change`: an existing fact is refined or replaced.
- `retract`: a previously-valid fact is withdrawn (no longer true).

**Probe**: a question or task the agent must answer or perform at a specific
step. Carries:
- `probe_id`: unique within the episode.
- `step`: the step at which the probe is issued.
- `kind`: factual (deterministic gold), synthesis (open-ended, judge-assisted),
  or behavioral (tests constraint adherence).
- `gold`: the correct answer given the world state at the probe step (revealed
  minus retracted facts).
- `cross_session`: whether the probe requires facts from a session prior to its
  own session.

**World state**: the set of currently-valid facts at step `t`, defined as
`{injected facts up to t} \ {retracted facts up to t}`. Probe gold is derived
from this state.

**world_event_hash**: a stable hash over the ordered exogenous event and probe
schedule. Identical for the same `(episode_id, seed)` across all memory
conditions, enforcing the counterfactual invariant.

### 1.2 Rendering

The simulator renders structured world events and probes into natural-language
text via an LLM (`SurfaceRenderer`), producing the surface text the agent reads.
Rendered text is **frozen-cached** keyed by `(episode_id, seed, step)`, so
rendering is reproducible and its cost is excluded from system cost accounting.
A `validate_render(episode)` guard asserts rendered text does not contradict
or leak facts beyond the ground-truth world state at that step.

### 1.3 Checker Protocol

Each family module implements a `Checker` that programmatically grades an
agent's answer against the probe gold. The checker returns a `[0,1]` score
plus structured metadata (which facts were used, drift flags, pass/fail per
criterion). Only probes that cannot be programmatically checked are deferred
to the sparse judge — and the judge share is bounded and reported separately.

---

## 2. Family Scope Caps (Anti-Explosion)

These caps prevent the benchmark from requiring real-world infrastructure,
network access, or unbounded computation during an episode. Both families
must be fully runnable offline after dataset generation.

### 2.1 Software-Development Family

**Concept**: the agent receives an evolving software specification across
multiple sessions. Requirements and conventions are injected, changed, and
retracted over time. Probes test whether the agent recalls and correctly
applies the CURRENT set of requirements — not stale ones.

**Scope caps**:

| Constraint | Limit |
|---|---|
| Package size | ≤ 6 files (`.py` + test files) |
| File length | ≤ 200 lines per file |
| Dependencies | Python stdlib only (no `pip install` during an episode) |
| Network access | **None** — sandboxed subprocess with `network=none` |
| Test framework | `pytest` (deterministic, offline) |
| Code execution | Sandboxed subprocess with time (≤ 30s), memory (≤ 256 MB), and output (≤ 10 KB) limits |
| Episode sessions | 2–5 sessions, modeling specification evolution over days/weeks |
| Requirements per episode | 5–15 total events (inject/change/retract mix) |

**Probe types**:
- **Implementation**: write/modify code to satisfy the current spec.
- **Convention adherence**: apply a still-active naming/style/API convention.
- **Deprecation awareness**: use the new API after an old one is retracted.
- **Test-driven**: code must pass the hidden `T_t` (pytest suite) for the
  current `R_t`.

**Grading**: the `SoftwareChecker` runs `T_t` in a sandboxed subprocess and
parses the pytest results. Static checks verify convention adherence. Drift
detection checks for stale-API use and constraint violations.

### 2.2 Research-Project Family

**Concept**: the agent conducts an autonomous investigation over a synthetic
evidence world. Findings are progressively injected, some are refined, others
are retracted (debunked). Probes test whether the agent synthesizes the current
evidence correctly and does not cite retracted findings.

**Scope caps**:

| Constraint | Limit |
|---|---|
| Entity names | Synthetic only — generated from a seeded name list (e.g., "Project Chimera", "Entity-42", "Study Gamma-7"). NO real paper titles, authors, DOIs. |
| Evidence IDs | Generated `fact_id`s (e.g., `ev-001`, `ev-002`). All probe golds map to a known synthetic fact. |
| Fact count | 15–40 facts per episode |
| Retraction rate | 20%–40% of injected facts are later retracted or superseded |
| Episode sessions | 3–6 sessions, modeling months-long investigation |
| Evidence graph | Frozen DAG — dependencies between facts are seeded and fixed. A retracted parent cascades to dependent facts. |
| Agent actions | Do NOT mutate the evidence graph (v1 fixed world). |

**Probe types**:
- **Factual recall**: "What did Study X conclude about Y?" (gold = current
  non-retracted synthesis).
- **Update correctness**: "Based on the latest evidence, what is the current
  understanding of Z?" (must NOT cite retracted findings).
- **Synthesis**: "Summarize the current state of evidence on topic T."
  Open-ended — deferred to sparse judge with rubric.
- **Objective adherence**: "Is this line of investigation still consistent
  with the original research question?" (constraint check).

**Grading**: the `ResearchChecker` maps every factual claim in the answer to a
known synthetic `fact_id`. Deterministic for factual/update probes; synthesis
probes deferred to judge. Drift detection checks for stale-fact citation and
objective-constraint violations.

**Leakage guard**: `lint_no_real_entities()` scans generated text for known
real-world identifiers (paper titles, author names, DOI patterns). Any match
is rejected at generation time.

---

## 3. Reproducibility

### 3.1 Frozen + Checksummed

Pilot datasets are generated once, validated, and frozen to versioned files
under `datasets/<family>_pilot/`. Each frozen set includes:

- `episodes.jsonl`: one JSON object per episode (events, probes, gold,
  world_event_hash, episode_hash).
- `rendered/`: per-step rendered text, keyed by `(episode_id, seed, step)`.
- `MANIFEST.json`: generator version, git SHA, config hash, seeds, scale
  params, per-file SHA-256 checksums, generation timestamp.
- `dataset_card.md`: human-readable summary (see Section 4).

The `verify` command recomputes checksums and asserts they match the manifest.
A tampered file (one byte flipped) causes `verify` to exit non-zero with a
mismatch report.

### 3.2 Seeded Regeneration

The `regen-check` command regenerates episodes from the stored seeds and
asserts **identical** `world_event_hash` and `episode_hash` values to the
frozen set. This proves the generation recipe is deterministic and reproducible
from seeds alone — even without the frozen files.

Seeds, scale params, and the generator version are all stored in the manifest.
The agent and judge models are NOT pinned in the dataset itself (they are
run-config), but the rendering model's config is recorded so rendered text
can be reproduced.

### 3.3 Run Manifest

Each experiment run produces a `run_manifest.json` pinning:
- Git SHA of the benchmark code.
- Config hash (model names, budget, track, weights).
- Dataset checksums (from the frozen manifest).
- Agent model revision, judge model revision (`lordx64/Qwable-v1` by commit
  hash).
- Run timestamp and environment.

This enables exact replication of any reported result.

---

## 4. Dataset Cards

Each family's `dataset_card.md` is a structured, human-readable summary
following a standard template:

### Template

```markdown
# Dataset: <family_name> Pilot (v1)

## Overview
- **Family**: Research / Software-Dev
- **Episodes**: <n>
- **Total sessions**: <n>
- **Seeds**: <list>
- **Frozen date**: <ISO 8601>
- **Generator version**: <git SHA>

## Content Description
<one-paragraph summary of what the episodes contain>

## Probe Composition
| Type | Count | Grading method |
|---|---|---|
| Factual recall | <n> | Programmatic (ResearchChecker) |
| Implementation | <n> | Programmatic (pytest via SoftwareChecker) |
| Synthesis (open-ended) | <n> | Judge (`lordx64/Qwable-v1`) |
| Constraint/behavioral | <n> | Programmatic (drift invariants) |

## Scope Compliance
- [ ] All entity names synthetic (no real papers/authors)
- [ ] No network access required
- [ ] File count ≤ 6 (SW-Dev) / N/A (Research)
- [ ] `validate_render` passed on all episodes
- [ ] `lint_no_real_entities` passed on all episodes

## Reproducibility
- **SHA-256** (episodes.jsonl): `<hash>`
- **SHA-256** (rendered/): `<hash>`
- **Regeneration verified**: <date> — identical hashes
- **Run**: `python -m lhmsb.datasets regen-check --frozen datasets/<name>`

## Intended Use
Benchmarking long-horizon memory systems on <family> tasks.
Not for training or fine-tuning.

## Limitations
- Synthetic content only — does not reflect real-world <domain> complexity.
- Fixed evidence graph (v1) — agent actions do not affect the world.
- Pilot scale (<n> episodes) — not powered for fine-grained system ranking.
```

---

## 5. Dataset Lifecycle

```
generate  ───→  validate  ───→  freeze  ───→  verify  ───→  regen-check
  │               │               │              │               │
  │  (Task 20)    │  (sim core    │  (write      │  (integrity   │  (prove
  │               │   + families) │   manifest)  │   check)      │   reproducible)
  ▼               ▼               ▼              ▼               ▼
episodes       render ok?      datasets/       checksums       hashes match
+ rendered     no leakage      <name>/         match           frozen set
text           no real ents    MANIFEST.json
                               dataset_card.md
```

All steps are automated via `python -m lhmsb.datasets`. No manual curation,
no hand-editing of generated episodes.

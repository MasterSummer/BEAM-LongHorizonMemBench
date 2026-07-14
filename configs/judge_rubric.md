<!-- rubric-version: 1.0.0 -->
<!-- judge-model: lordx64/Qwable-v1 -->
<!-- judge-revision: 0000000000000000000000000000000000000000 -->
<!-- judge-max-weight: 0.20 -->

# LHMSB Judge Rubric (v1.0.0)

> **Fixed, versioned rubric.** This file is the single source of truth for the LLM
> judge's scoring criteria AND its model pin. It is read at runtime by
> `lhmsb.judge.load_judge_config()` and `lhmsb.judge.load_rubric()`. Changing the
> criteria below REQUIRES bumping `rubric-version` in the metadata block at the top
> of this file, so every recorded `JudgeScore` is traceable to an exact rubric revision.

## Model Pin (config-driven, never hard-coded in source)

| Field | Value | Notes |
|-------|-------|-------|
| `judge-model` | `lordx64/Qwable-v1` | User-specified judge model. Loaded ONLY via config; source code never hard-codes this id. |
| `judge-revision` | *(placeholder — see below)* | The judge MUST be pinned by a revision/commit hash. The 40-hex placeholder above MUST be replaced with the real pinned commit SHA before any live run (`LHMSB_LIVE_JUDGE=1`). |
| `judge-max-weight` | `0.20` | Maximum fraction of any composite score the judge may contribute. Hard cap; the judge never dominates the headline metric. |

> **Pinning discipline.** `JudgeConfig` rejects an empty revision. The placeholder
> `0000…0000` is intentionally obvious; replace it with the exact commit hash the
> evaluation was run against and record it in `run_manifest.json`.

## Scope & Discipline (read before using the judge)

The judge is used **only** where programmatic grading is impossible — open-ended
**synthesis** quality and free-text **rationale** checks — and **only at episode
boundaries**, never per step and never inside the agent loop. Judge tokens are an
auditing cost; they are **excluded** from the system `CostVector` and from Memory ROI.
Every judge prompt and output is written to an audit trace log.

## Scoring Scale

The judge returns a single continuous **score in `[0, 1]`** plus a short rationale.
Anchor the score to the bands below; interpolate between anchors as needed.

| Score band | Label | Meaning |
|------------|-------|---------|
| `1.00` | Fully correct | Captures all required points in the gold reference; no contradictions; no use of retracted/superseded facts. |
| `0.75` | Mostly correct | Captures the main required points; at most a minor omission; no contradictions with the gold reference. |
| `0.50` | Partially correct | Captures roughly half of the required points, OR is correct but materially incomplete. |
| `0.25` | Mostly incorrect | Misses most required points, OR contains a significant error/contradiction. |
| `0.00` | Incorrect / empty | Wrong, empty, off-topic, or relies on a fact that was retracted or superseded at the probe step. |

## Scoring Criteria (in priority order)

1. **Faithfulness to gold.** The answer must agree with the gold reference derived
   from the world state at the probe step (revealed-minus-retracted facts). Using a
   fact that was retracted or superseded at or before the probe step is a hard error
   and caps the score at `0.25`.
2. **Coverage.** Of the required points present in the gold reference, what fraction
   does the answer correctly include? Reward completeness, penalize omissions.
3. **No fabrication.** Claims not supported by the evidence world reduce the score.
   A single confident fabrication that contradicts gold caps the score at `0.50`.
4. **Internal consistency.** The answer must not contradict itself.
5. **Relevance & concision.** The answer should address the probe query directly.
   Irrelevant padding does not raise the score but does not penalize a correct core.

## Output Contract

The judge backend is prompted to return a strict JSON object:

```json
{"score": 0.0, "rationale": "<one or two sentences citing the deciding criterion>"}
```

`score` is clamped to `[0, 1]` by the harness. The rationale must name the criterion
that decided the score (e.g., "missed 2 of 4 required points → coverage 0.5").

## Calibration & Consistency

Before reporting any judge-assisted metric, the harness runs:

- **Calibration** (`lhmsb.judge.calibrate`) on a small gold set of `(probe, answer,
  expected_score)` triples, reporting an **agreement** rate (fraction within a
  tolerance) and mean absolute error against human-assigned gold scores.
- **Consistency** (`lhmsb.judge.judge_consistency`) by re-scoring the same input
  repeatedly and reporting score spread/standard deviation (repeat-stability).

If the judge's calibration agreement falls below the declared threshold, or its
consistency spread exceeds the declared bound, the judge-assisted probes are flagged
for manual audit rather than silently trusted.

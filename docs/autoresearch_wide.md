# AutoResearchBench Wide Research

This repository can import the public AutoResearchBench Wide Research JSONL as an
external `research_wide` dataset. The importer preserves the original question and
gold arXiv ID set. It only creates history events when the input record contains an
explicit, separately frozen `history` trace; it never invents a trace or puts gold
papers into memory implicitly.

## Smoke import

The official repository publishes a small example file at:

`https://raw.githubusercontent.com/CherYou/AutoResearchBench/main/input_data/academic_widesearch_example.jsonl`

```bash
curl -L \
  https://raw.githubusercontent.com/CherYou/AutoResearchBench/main/input_data/academic_widesearch_example.jsonl \
  -o /tmp/academic_widesearch_example.jsonl

python -m lhmsb.datasets import-wide \
  --input /tmp/academic_widesearch_example.jsonl \
  --limit 5 \
  --seed 7 \
  --out /tmp/lhmsb-wide-stage

python -m lhmsb.datasets freeze \
  --src /tmp/lhmsb-wide-stage \
  --out /tmp/lhmsb-wide-frozen

python -m lhmsb.datasets verify \
  --frozen /tmp/lhmsb-wide-frozen
```

## Scoring

Each imported episode has one `wide_set` probe. The `WideResearchChecker` extracts
modern arXiv IDs from JSON or free-form answers and reports:

- `iou`: primary set-overlap score;
- `recall`: fraction of gold papers returned;
- `precision`: fraction of returned papers that are in the gold set.

The official benchmark bundle is obfuscated and must be decrypted with the release
repository's `decrypt_benchmark.py` before importing. The original bundle contains
target questions and gold paper IDs, but not a multi-session research trace. A
memory benchmark track must therefore add a separately frozen search trace before
claiming long-horizon memory results. The importer accepts that extension as a
record-level `history` list with `session`, `step`, `text`, `arxiv_ids`, and
`memory_policy` fields.

## Formal leakage-safe construction

The decrypted bundle contains 400 Wide rows. Two rows are exact copies of the same
question, answer, and gold-ID set, so construction removes that duplicate and exports
399 unique question records. A repeated question with conflicting gold IDs is an
integrity error rather than an implicit merge.

Formal traces use all ten Parquet shards from
`librarian-bots/arxiv-metadata-snapshot` pinned at commit
`0a7bddb5ae22d0358560d09e55627d4f86f3743a`. The local SQLite FTS5 index manifest
records each shard's byte count and SHA-256, and the trace manifest records the
index hash as its search revision.

Construction is deliberately split into three irreversible stages:

1. `wide-questions` exports only the question, stable ID, and question hash. Gold
   fields are recursively forbidden in this artifact.
2. `wide-traces` uses only that question artifact and the frozen metadata index to
   produce three retrieval sessions. It seals the trace file and hash before any
   evaluator labels are available.
3. `attach-wide-traces` verifies the sealed hash, joins evaluator gold, labels
   observed gold papers `must_store` and other observed papers `must_not_store`, and
   appends a context-only final synthesis session.

The declared formal subset uses `--min-gold-observed 1`. Qualification therefore
happens only after every trace is frozen and cannot change queries, rankings, or
observations. Both the all-question audit and the qualified audit must be retained so
coverage and exclusion rates remain visible.

## Three-condition replay

The first controlled experiment uses exactly three conditions:

| Condition | Meaning |
| --- | --- |
| `no_mem` | Cross-session writes are discarded and retrieval is empty. |
| `mem` | Relevant recorded history can be returned. This is the controlled upper-bound memory condition. |
| `wrong_mem` | The same write lifecycle runs, but retrieval returns deterministic distractor content. |

Use `MEMORY_ABLATION_CONDITIONS` with `run_matrix` to keep the condition order fixed:

```python
from lhmsb.runner import MEMORY_ABLATION_CONDITIONS, run_matrix

table = run_matrix(
    episodes,
    run_config,
    agent_model=agent_model,
    conditions=MEMORY_ABLATION_CONDITIONS,
    paper_search=frozen_paper_search,
)
```

## Count-based efficiency

The scorecard no longer uses token/resource cost as its efficiency denominator.
It reports `Memory ROI = mean(normalized task gain) / mean(recorded memory count)`.
`no_mem` has ROI `N/A`; a system that records zero memories is also `N/A`, never
infinite. The legacy token-cost API remains importable only for old result files and
is not used by the current scorecard.

Each run additionally records:

| Metric | Definition |
| --- | --- |
| `stored_memory_count` | Number of write IDs returned by the memory adapter, excluding `no_mem`. |
| `storage_precision` | Required writes divided by required plus explicitly forbidden writes. |
| `storage_recall` | Required writes actually issued divided by all `must_store` records. |
| `retrieval_precision` | Relevant unique facts/papers divided by all unique items returned at a probe. |
| `retrieval_recall` | Relevant unique facts/papers returned divided by the probe's required set. |
| `retrieval_false_positive_rate` | `1 - retrieval_precision`. |
| `retrieval_timeliness` | Fraction of relevant retrieved items that are used in the answer to that same probe. |

`memory_policy` is evaluator metadata. It does not claim to reveal a backend's
private storage layout. The benchmark measures observable write decisions and
downstream use, which stays comparable when different memory systems merge,
summarize, or graph the same input differently.

## Required history shape

Raw Wide Research questions without `history` are valid for set-retrieval scoring,
but they are not sufficient evidence for long-horizon memory. A memory experiment
record should contain at least one earlier session with `must_store` evidence, one
later-session distractor or update, and a target probe whose answer depends on the
earlier evidence. Gold arXiv IDs must never be inserted into history unless they were
actually present in the frozen research trace.

"""Counterfactual experiment runner / orchestrator (task 21).

Executes the counterfactual matrix — every ``(episode, condition, seed)`` cell —
over the frozen datasets, grades the harness's raw probe answers with the family
checkers + sparse judge, applies the per-episode metrics (task / drift / retrieval),
and persists a tidy results table (jsonl + parquet) keyed by (episode_id, condition,
seed). Native and controlled tracks are never mixed.

Submodules:
  * :mod:`lhmsb.runner.adapters` — condition -> adapter factory.
  * :mod:`lhmsb.runner.grading` — raw harness answers -> graded ProbeResults.
  * :mod:`lhmsb.runner.results` — tidy RunRow + jsonl/parquet persistence.
  * :mod:`lhmsb.runner.orchestrator` — the matrix loop + counterfactual invariant.
"""

from __future__ import annotations

from lhmsb.runner.adapters import (
    ALL_CONDITIONS,
    LEADERBOARD_CONDITIONS,
    MEM,
    MEMORY_ABLATION_CONDITIONS,
    NO_MEM,
    SENSITIVITY_CONDITIONS,
    WRONG_MEM,
    AdapterFactory,
    UnknownConditionError,
    build_adapter,
    default_adapter_factory,
    ground_truth_facts,
)
from lhmsb.runner.grading import (
    DEFAULT_JUDGE_CORRECT_THRESHOLD,
    build_checker,
    drift_category_for,
    grade_episode,
    grade_probe,
)
from lhmsb.runner.orchestrator import (
    DEFAULT_MAX_ATTEMPTS,
    CounterfactualError,
    load_frozen_dataset,
    merge_costs,
    run_matrix,
)
from lhmsb.runner.results import (
    COST_FIELDS,
    ResultsTable,
    RunRow,
    write_jsonl,
    write_parquet,
    write_results,
)

__all__ = [
    "ALL_CONDITIONS",
    "COST_FIELDS",
    "DEFAULT_JUDGE_CORRECT_THRESHOLD",
    "DEFAULT_MAX_ATTEMPTS",
    "LEADERBOARD_CONDITIONS",
    "MEMORY_ABLATION_CONDITIONS",
    "MEM",
    "NO_MEM",
    "WRONG_MEM",
    "SENSITIVITY_CONDITIONS",
    "AdapterFactory",
    "CounterfactualError",
    "ResultsTable",
    "RunRow",
    "UnknownConditionError",
    "build_adapter",
    "build_checker",
    "default_adapter_factory",
    "drift_category_for",
    "grade_episode",
    "grade_probe",
    "ground_truth_facts",
    "load_frozen_dataset",
    "merge_costs",
    "run_matrix",
    "write_jsonl",
    "write_parquet",
    "write_results",
]

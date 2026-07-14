"""Set-valued metrics used by the AutoResearchBench Wide Research family."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class WideSetMetrics:
    """Exact set-comparison metrics for a Wide Research answer."""

    iou: float
    recall: float
    precision: float
    hit_ids: tuple[str, ...]
    missed_ids: tuple[str, ...]
    extra_ids: tuple[str, ...]


def compute_wide_set_metrics(
    gold_ids: Iterable[str], predicted_ids: Iterable[str]
) -> WideSetMetrics:
    """Compute IoU, recall, precision, and the three ID partitions."""
    gold = {str(item) for item in gold_ids if str(item)}
    predicted = {str(item) for item in predicted_ids if str(item)}
    hits = gold & predicted
    missed = gold - predicted
    extra = predicted - gold
    union = gold | predicted
    iou = 1.0 if not gold and not predicted else len(hits) / len(union)
    recall = len(hits) / len(gold) if gold else (1.0 if not predicted else 0.0)
    precision = len(hits) / len(predicted) if predicted else (1.0 if not gold else 0.0)
    return WideSetMetrics(
        iou=iou,
        recall=recall,
        precision=precision,
        hit_ids=tuple(sorted(hits)),
        missed_ids=tuple(sorted(missed)),
        extra_ids=tuple(sorted(extra)),
    )

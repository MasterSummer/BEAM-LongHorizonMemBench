"""Zero-API conformance suite for decision-aligned failure localization.

The suite starts from backend-shaped raw lifecycle events, derives the stored,
retrieved, and model-visible state sets, and only then calls the production
attribution function.  It therefore tests the event-to-stage bridge instead of
handing final stage sets directly to the classifier.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from lhmsb.longhorizon.failure_attribution import (
    DecisionMemoryAttribution,
    StorageEvidenceMode,
    attribute_decision_memory,
)

BackendTrack = Literal["flat_retrieval", "mem0", "amem", "memos"]
FaultClass = Literal[
    "no_fault",
    "storage_missing",
    "retrieval_missing",
    "exposure_missing",
    "visible_but_wrong",
]
ObservedClass = Literal[
    "no_fault",
    "storage_missing",
    "retrieval_missing",
    "exposure_missing",
    "visible_but_wrong",
    "unresolved",
]

KNOWN_FAULT_SCHEMA_VERSION = "memhorizon-known-fault-conformance-v1"
DEFAULT_EPISODE_IDS = tuple(f"software-longitudinal-{seed}" for seed in range(301, 306))
BACKEND_TRACKS: tuple[BackendTrack, ...] = (
    "flat_retrieval",
    "mem0",
    "amem",
    "memos",
)
FAULT_CLASSES: tuple[FaultClass, ...] = (
    "no_fault",
    "storage_missing",
    "retrieval_missing",
    "exposure_missing",
    "visible_but_wrong",
)

_EVENT_KIND_BY_BACKEND: dict[BackendTrack, dict[str, str]] = {
    "flat_retrieval": {
        "storage": "flat_index_write",
        "retrieval": "flat_candidate_returned",
        "exposure": "common_rerank_visible",
        "probe": "neutral_replacement_effect",
    },
    "mem0": {
        "storage": "mem0_native_write",
        "retrieval": "mem0_native_search_hit",
        "exposure": "common_rerank_visible",
        "probe": "neutral_replacement_effect",
    },
    "amem": {
        "storage": "amem_native_write",
        "retrieval": "amem_native_search_hit",
        "exposure": "common_rerank_visible",
        "probe": "neutral_replacement_effect",
    },
    "memos": {
        "storage": "inventory_state_present",
        "retrieval": "memos_search_hit",
        "exposure": "common_rerank_visible",
        "probe": "neutral_replacement_effect",
    },
}


@dataclass(frozen=True)
class RawLifecycleEvent:
    sequence: int
    event_kind: str
    state_id: str
    object_id: str

    def __post_init__(self) -> None:
        if self.sequence < 0:
            raise ValueError("event sequence must be non-negative")
        if not self.event_kind or not self.state_id or not self.object_id:
            raise ValueError("lifecycle event fields must be non-empty")


@dataclass(frozen=True)
class RawDecisionTrace:
    case_id: str
    episode_id: str
    backend: BackendTrack
    required_state_ids: tuple[str, ...]
    events: tuple[RawLifecycleEvent, ...]
    behavior_correct: bool

    def __post_init__(self) -> None:
        if not self.case_id or not self.episode_id:
            raise ValueError("trace case and episode IDs must be non-empty")
        if not self.required_state_ids:
            raise ValueError("known-fault trace requires target state")
        sequences = tuple(event.sequence for event in self.events)
        if sequences != tuple(sorted(sequences)) or len(sequences) != len(set(sequences)):
            raise ValueError("lifecycle event sequence must be unique and ordered")


@dataclass(frozen=True)
class KnownFaultCaseResult:
    case_id: str
    episode_id: str
    backend: BackendTrack
    provenance_track: StorageEvidenceMode
    injected_fault: FaultClass
    expected_class: ObservedClass
    observed_class: ObservedClass
    passed: bool
    raw_events: tuple[RawLifecycleEvent, ...]
    attribution: DecisionMemoryAttribution

    def to_dict(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "episode_id": self.episode_id,
            "backend": self.backend,
            "provenance_track": self.provenance_track,
            "injected_fault": self.injected_fault,
            "expected_class": self.expected_class,
            "observed_class": self.observed_class,
            "passed": self.passed,
            "raw_events": [asdict(event) for event in self.raw_events],
            "attribution": self.attribution.to_dict(),
        }


def _provenance_track(backend: BackendTrack) -> StorageEvidenceMode:
    return "inferred" if backend == "memos" else "native/exact"


def _events_for_fault(
    backend: BackendTrack,
    fault: FaultClass,
    *,
    state_id: str,
) -> tuple[RawLifecycleEvent, ...]:
    kinds = _EVENT_KIND_BY_BACKEND[backend]
    stages = {
        "no_fault": ("storage", "retrieval", "exposure", "probe"),
        "storage_missing": (),
        "retrieval_missing": ("storage",),
        "exposure_missing": ("storage", "retrieval"),
        "visible_but_wrong": ("storage", "retrieval", "exposure", "probe"),
    }[fault]
    return tuple(
        RawLifecycleEvent(
            sequence=index,
            event_kind=kinds[stage],
            state_id=state_id,
            object_id=f"{backend}:{state_id}:object-0",
        )
        for index, stage in enumerate(stages)
    )


def build_known_fault_trace(
    episode_id: str,
    backend: BackendTrack,
    fault: FaultClass,
) -> RawDecisionTrace:
    """Build one isolated raw-event fixture with a known earliest break."""
    target = "P2"
    return RawDecisionTrace(
        case_id=f"{episode_id}__{backend}__{fault}",
        episode_id=episode_id,
        backend=backend,
        required_state_ids=(target,),
        events=_events_for_fault(backend, fault, state_id=target),
        behavior_correct=fault == "no_fault",
    )


def attribute_raw_trace(trace: RawDecisionTrace) -> DecisionMemoryAttribution:
    """Normalize backend-shaped events and call the production classifier."""
    kinds = _EVENT_KIND_BY_BACKEND[trace.backend]
    inverse = {value: stage for stage, value in kinds.items()}
    by_stage: dict[str, set[str]] = {
        "storage": set(),
        "retrieval": set(),
        "exposure": set(),
        "probe": set(),
    }
    for event in trace.events:
        try:
            stage = inverse[event.event_kind]
        except KeyError as exc:
            raise ValueError(
                f"unknown {trace.backend} lifecycle event: {event.event_kind}"
            ) from exc
        by_stage[stage].add(event.state_id)

    if not by_stage["retrieval"].issubset(by_stage["storage"]):
        raise ValueError("retrieval event lacks upstream storage evidence")
    if not by_stage["exposure"].issubset(by_stage["retrieval"]):
        raise ValueError("exposure event lacks upstream retrieval evidence")
    if not by_stage["probe"].issubset(by_stage["exposure"]):
        raise ValueError("probe event lacks upstream exposure evidence")

    return attribute_decision_memory(
        memory_reliant_state_ids=trace.required_state_ids,
        stored_state_ids=tuple(sorted(by_stage["storage"])),
        retrieved_state_ids=tuple(sorted(by_stage["retrieval"])),
        visible_state_ids=tuple(sorted(by_stage["exposure"])),
        probed_state_ids=tuple(sorted(by_stage["probe"])),
        causally_used_state_ids=tuple(sorted(by_stage["probe"])),
        behavior_correct=trace.behavior_correct,
        has_memory_channel=True,
        storage_evidence_mode=_provenance_track(trace.backend),
    )


def _observed_class(attribution: DecisionMemoryAttribution) -> ObservedClass:
    return {
        "storage_failure": "storage_missing",
        "retrieval_failure": "retrieval_missing",
        "exposure_failure": "exposure_missing",
        "utilization_failure": "visible_but_wrong",
        "behavior_success_causal": "no_fault",
    }.get(attribution.stage, "unresolved")  # type: ignore[return-value]


def run_known_fault_suite(
    episode_ids: tuple[str, ...] = DEFAULT_EPISODE_IDS,
) -> dict[str, object]:
    """Run the deterministic 5 × 4 × 5 conformance matrix."""
    if not episode_ids or len(episode_ids) != len(set(episode_ids)):
        raise ValueError("episode IDs must be a non-empty unique sequence")
    cases: list[KnownFaultCaseResult] = []
    for episode_id in episode_ids:
        for backend in BACKEND_TRACKS:
            for fault in FAULT_CLASSES:
                trace = build_known_fault_trace(episode_id, backend, fault)
                attribution = attribute_raw_trace(trace)
                observed = _observed_class(attribution)
                cases.append(
                    KnownFaultCaseResult(
                        case_id=trace.case_id,
                        episode_id=episode_id,
                        backend=backend,
                        provenance_track=_provenance_track(backend),
                        injected_fault=fault,
                        expected_class=fault,
                        observed_class=observed,
                        passed=observed == fault,
                        raw_events=trace.events,
                        attribution=attribution,
                    )
                )

    labels: tuple[ObservedClass, ...] = (*FAULT_CLASSES, "unresolved")
    confusion = {
        expected: {
            observed: sum(
                case.expected_class == expected and case.observed_class == observed
                for case in cases
            )
            for observed in labels
        }
        for expected in FAULT_CLASSES
    }
    exact = [case for case in cases if case.provenance_track == "native/exact"]
    inferred = [case for case in cases if case.provenance_track == "inferred"]
    failed = [case.case_id for case in cases if not case.passed]
    return {
        "schema_version": KNOWN_FAULT_SCHEMA_VERSION,
        "episode_ids": list(episode_ids),
        "backend_tracks": list(BACKEND_TRACKS),
        "fault_classes": list(FAULT_CLASSES),
        "case_count": len(cases),
        "expected_case_count": len(episode_ids) * len(BACKEND_TRACKS) * len(FAULT_CLASSES),
        "passed_case_count": len(cases) - len(failed),
        "failed_case_ids": failed,
        "exact_case_count": len(exact),
        "exact_passed_case_count": sum(case.passed for case in exact),
        "inferred_case_count": len(inferred),
        "inferred_passed_case_count": sum(case.passed for case in inferred),
        "no_fault_false_positive_count": sum(
            case.injected_fault == "no_fault" and case.observed_class != "no_fault"
            for case in cases
        ),
        "confusion": confusion,
        "ok": not failed,
        "cases": [case.to_dict() for case in cases],
    }


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def write_known_fault_report(report: dict[str, object], out_dir: Path) -> None:
    """Write case ledger, summary, and a rectangular confusion CSV."""
    cases = report["cases"]
    assert isinstance(cases, list)
    summary = {key: value for key, value in report.items() if key != "cases"}
    _atomic_write_text(
        out_dir / "known_fault_summary.json",
        json.dumps(summary, sort_keys=True, ensure_ascii=False, indent=2) + "\n",
    )
    _atomic_write_text(
        out_dir / "known_fault_cases.jsonl",
        "".join(json.dumps(case, sort_keys=True, ensure_ascii=False) + "\n" for case in cases),
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "known_fault_confusion.csv"
    descriptor, temporary = tempfile.mkstemp(prefix=f".{csv_path.name}.", dir=out_dir)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            labels = [*FAULT_CLASSES, "unresolved"]
            writer = csv.DictWriter(handle, fieldnames=["expected", *labels])
            writer.writeheader()
            confusion = report["confusion"]
            assert isinstance(confusion, dict)
            for expected in FAULT_CLASSES:
                row = confusion[expected]
                assert isinstance(row, dict)
                writer.writerow({"expected": expected, **row})
        os.replace(temporary, csv_path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--episode-id", action="append", dest="episode_ids")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    episode_ids = tuple(args.episode_ids or DEFAULT_EPISODE_IDS)
    report = run_known_fault_suite(episode_ids)
    write_known_fault_report(report, args.out_dir)
    summary = {key: value for key, value in report.items() if key != "cases"}
    print(json.dumps(summary, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "BACKEND_TRACKS",
    "DEFAULT_EPISODE_IDS",
    "FAULT_CLASSES",
    "KnownFaultCaseResult",
    "RawDecisionTrace",
    "RawLifecycleEvent",
    "attribute_raw_trace",
    "build_known_fault_trace",
    "main",
    "run_known_fault_suite",
    "write_known_fault_report",
]

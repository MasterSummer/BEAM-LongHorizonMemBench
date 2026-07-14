"""Programmatic grader for the Software-Dev family (spec/04-datasets.md §2.1).

``SoftwareChecker`` grades an agent's code answer for a probe by:
  1. interpreting the ACTIVE requirements ``R_t`` at the probe step (revealed-minus-
     retracted world state, reusing :class:`~lhmsb.sim.core.WorldState`);
  2. generating the hidden test suite ``T_t`` from those active requirements (so
     ``T_t`` always encodes the CURRENT spec) and running it in the offline,
     resource-bounded :func:`~lhmsb.families.software.sandbox.run_tests_sandboxed`;
  3. running static AST checks for deprecated-API use;
  4. scoring utilization (used the recalled active decision?), update-correctness
     (used the NEW value after a change?), and constraint adherence (did not violate
     a still-active constraint), emitting ``drift_flags`` (``stale-api``,
     ``stale-decision``, ``stale-value``, ``constraint-violation``).

Deterministic: the same answer yields the same ``CheckResult`` (sandbox pass/fail
and the static scan are both deterministic).
"""

from __future__ import annotations

import ast
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from lhmsb.families.software.generator import SoftwareSpec
from lhmsb.families.software.sandbox import (
    DEFAULT_MEM_LIMIT_MB,
    DEFAULT_OUTPUT_LIMIT_BYTES,
    DEFAULT_TIME_LIMIT_S,
    TestResult,
    run_tests_sandboxed,
)
from lhmsb.sim.core import CheckResult, WorldState
from lhmsb.types import Probe, WorldEvent

SandboxRunner = Callable[..., TestResult]
_DRIFT_NEGATES_UTILIZATION = ("stale-api", "stale-decision")
_SAMPLE_NAME = "Sample Name"


@dataclass(frozen=True)
class RuleSet:
    """Active requirements ``R_t`` at a probe step, interpreted from world events."""

    active_fn: str | None
    api_fact_id: str | None
    deprecated_fns: list[str] = field(default_factory=list)
    require_snake_case: bool = False
    snake_rule_id: str | None = None
    snake_fact_id: str | None = None
    active_prefix: str | None = None
    prefix_fact_id: str | None = None
    retracted_prefixes: list[tuple[str, str, str]] = field(default_factory=list)
    status_value: str | None = None
    status_fact_id: str | None = None
    status_changed: bool = False


def _str_field(payload: dict[str, object], key: str) -> str | None:
    """Narrow ``payload[key]`` to ``str`` or ``None``."""
    value = payload.get(key)
    return value if isinstance(value, str) else None


def _str_list_field(payload: dict[str, object], key: str) -> list[str]:
    """Narrow ``payload[key]`` to a ``list[str]`` (dropping non-str members)."""
    value = payload.get(key)
    if isinstance(value, list):
        return [item for item in value if isinstance(item, str)]
    return []


def _find_retracted_prefixes(
    events: list[WorldEvent], step: int, valid: Mapping[str, object]
) -> list[tuple[str, str, str]]:
    """Prefix conventions injected by ``step`` but no longer valid (decision reversed)."""
    seen: dict[str, tuple[str | None, str | None]] = {}
    for event in sorted(events, key=lambda e: e.step):
        if event.step > step:
            break
        if event.kind in ("inject", "change") and _str_field(event.payload, "conv_kind") == (
            "prefix"
        ):
            seen[event.fact_id] = (
                _str_field(event.payload, "rule_id"),
                _str_field(event.payload, "prefix"),
            )
    out: list[tuple[str, str, str]] = []
    for fact_id, (rule_id, prefix) in seen.items():
        if fact_id not in valid and rule_id is not None and prefix is not None:
            out.append((rule_id, fact_id, prefix))
    return out


def _was_changed(events: list[WorldEvent], fact_id: str, step: int) -> bool:
    """True if a ``change`` event hit ``fact_id`` at or before ``step``."""
    return any(
        e.kind == "change" and e.fact_id == fact_id and e.step <= step for e in events
    )


def interpret_rules(events: list[WorldEvent], step: int) -> RuleSet:
    """Build the active :class:`RuleSet` from the world state at ``step``."""
    valid = WorldState(events).valid_facts_at(step)
    active_fn = api_fact_id = snake_rule_id = snake_fact_id = None
    active_prefix = prefix_fact_id = status_value = status_fact_id = None
    deprecated_fns: list[str] = []
    require_snake_case = False
    for fact_id, fact in valid.items():
        payload = fact.payload
        rule_kind = _str_field(payload, "rule_kind")
        if rule_kind == "api":
            active_fn = _str_field(payload, "active_fn")
            api_fact_id = fact_id
            deprecated_fns = _str_list_field(payload, "deprecated_fns")
        elif rule_kind == "convention":
            conv_kind = _str_field(payload, "conv_kind")
            if conv_kind == "snake_case":
                require_snake_case = True
                snake_rule_id = _str_field(payload, "rule_id")
                snake_fact_id = fact_id
            elif conv_kind == "prefix":
                active_prefix = _str_field(payload, "prefix")
                prefix_fact_id = fact_id
        elif rule_kind == "default" and _str_field(payload, "field") == "status":
            status_value = _str_field(payload, "value")
            status_fact_id = fact_id
    return RuleSet(
        active_fn=active_fn,
        api_fact_id=api_fact_id,
        deprecated_fns=deprecated_fns,
        require_snake_case=require_snake_case,
        snake_rule_id=snake_rule_id,
        snake_fact_id=snake_fact_id,
        active_prefix=active_prefix,
        prefix_fact_id=prefix_fact_id,
        retracted_prefixes=_find_retracted_prefixes(events, step, valid),
        status_value=status_value,
        status_fact_id=status_fact_id,
        status_changed=_was_changed(events, "req-status", step),
    )


def generate_tests(rules: RuleSet) -> str:
    """Render the hidden pytest suite ``T_t`` encoding the active requirements."""
    fn = rules.active_fn or "make_widget"
    lines = [
        "import widgetlib.core as core",
        "from widgetlib.conventions import is_snake_case",
        "",
        f"_NAME = {_SAMPLE_NAME!r}",
        "",
        "",
        "def test_uses_active_api():",
        f"    assert hasattr(core, {fn!r}), {f'active API {fn} must be defined'!r}",
        "",
        "",
        "def test_basic_creation():",
        f"    widget = core.{fn}(_NAME)",
        "    assert isinstance(widget, dict)",
        '    assert "id" in widget and "status" in widget',
    ]
    if rules.require_snake_case:
        lines += [
            "",
            "",
            "def test_snake_case_id():",
            f"    widget = core.{fn}(_NAME)",
            '    assert is_snake_case(widget["id"])',
        ]
    if rules.active_prefix is not None:
        lines += [
            "",
            "",
            "def test_prefix_applied():",
            f"    widget = core.{fn}(_NAME)",
            f'    assert widget["id"].startswith({rules.active_prefix!r})',
        ]
    for rule_id, _fact_id, prefix in rules.retracted_prefixes:
        lines += [
            "",
            "",
            f"def test_no_retracted_prefix_{rule_id}():",
            f"    widget = core.{fn}(_NAME)",
            f'    assert not widget["id"].startswith({prefix!r})',
        ]
    if rules.status_value is not None:
        lines += [
            "",
            "",
            "def test_default_status():",
            f"    widget = core.{fn}(_NAME)",
            f'    assert widget["status"] == {rules.status_value!r}',
        ]
    return "\n".join(lines) + "\n"


def static_scan(answer: str) -> tuple[set[str], set[str]]:
    """Return ``(defined_function_names, used_names)`` from the answer source."""
    try:
        tree = ast.parse(answer)
    except SyntaxError:
        return set(), set()
    defined = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    return defined, used


class SoftwareChecker:
    """Runs ``T_t`` via the sandbox + static checks; scores utilization/drift."""

    def __init__(
        self,
        spec: SoftwareSpec,
        *,
        sandbox_runner: SandboxRunner = run_tests_sandboxed,
        time_limit_s: int = DEFAULT_TIME_LIMIT_S,
        mem_limit_mb: int = DEFAULT_MEM_LIMIT_MB,
        output_limit_bytes: int = DEFAULT_OUTPUT_LIMIT_BYTES,
    ) -> None:
        self._spec = spec
        self._sandbox_runner = sandbox_runner
        self._time_limit_s = time_limit_s
        self._mem_limit_mb = mem_limit_mb
        self._output_limit_bytes = output_limit_bytes

    def check(self, probe: Probe, answer: str) -> CheckResult:
        """Grade ``answer`` (source for the answer file) against ``R_t``/``T_t``."""
        rules = interpret_rules(self._spec.events, probe.step)
        result = self._run_suite(answer, generate_tests(rules))
        defined, used = static_scan(answer)
        failed = set(result.failed_tests)
        active_present = bool(rules.active_fn) and rules.active_fn in defined

        drift_flags = self._drift_flags(rules, defined, used, failed, active_present)
        test_score = result.passed / result.total if result.total > 0 else 0.0
        score = min(test_score, 0.25) if drift_flags else test_score
        is_correct = result.all_passed and not drift_flags

        utilization = active_present and not any(
            f.startswith(_DRIFT_NEGATES_UTILIZATION) for f in drift_flags
        )
        update_correctness = (
            (active_present and "test_default_status" not in failed)
            if rules.status_changed
            else True
        )
        constraint_adherence = not any(
            f.startswith("constraint-violation") for f in drift_flags
        )
        metadata: dict[str, object] = {
            "subtype": self._subtype(probe.probe_id),
            "step": probe.step,
            "test_score": round(test_score, 4),
            "passed": result.passed,
            "failed": result.failed,
            "errors": result.errors,
            "total": result.total,
            "failed_tests": list(result.failed_tests),
            "utilization": utilization,
            "update_correctness": update_correctness,
            "update_applicable": rules.status_changed,
            "constraint_adherence": constraint_adherence,
            "active_api_fn": rules.active_fn,
            "default_status": rules.status_value,
            "sandbox_timed_out": result.timed_out,
            "returncode": result.returncode,
        }
        return CheckResult(
            score=round(score, 4),
            is_correct=is_correct,
            facts_used=self._facts_used(rules, active_present, failed),
            drift_flags=drift_flags,
            metadata=metadata,
        )

    @staticmethod
    def _drift_flags(
        rules: RuleSet,
        defined: set[str],
        used: set[str],
        failed: set[str],
        active_present: bool,
    ) -> list[str]:
        """Collect drift flags: stale-api (static) + behavioral (runtime, gated)."""
        flags: list[str] = []
        for fn in rules.deprecated_fns:
            if fn in defined or fn in used:
                flags.append(f"stale-api:{fn}")
        if not active_present:
            return flags  # behavioral checks are moot when the active API is missing
        if rules.snake_rule_id is not None and "test_snake_case_id" in failed:
            flags.append(f"constraint-violation:{rules.snake_rule_id}")
        for rule_id, _fact_id, _prefix in rules.retracted_prefixes:
            if f"test_no_retracted_prefix_{rule_id}" in failed:
                flags.append(f"stale-decision:{rule_id}")
        if rules.status_value is not None and "test_default_status" in failed:
            flags.append("stale-value:status" if rules.status_changed else (
                "constraint-violation:status"
            ))
        return flags

    @staticmethod
    def _facts_used(rules: RuleSet, active_present: bool, failed: set[str]) -> list[str]:
        """Active requirement fact_ids the answer honoured (feeds the utilization metric)."""
        if not active_present:
            return []
        facts: list[str] = []
        if rules.api_fact_id is not None:
            facts.append(rules.api_fact_id)
        if rules.snake_fact_id is not None and "test_snake_case_id" not in failed:
            facts.append(rules.snake_fact_id)
        if rules.status_fact_id is not None and "test_default_status" not in failed:
            facts.append(rules.status_fact_id)
        if rules.prefix_fact_id is not None and "test_prefix_applied" not in failed:
            facts.append(rules.prefix_fact_id)
        return facts

    @staticmethod
    def _subtype(probe_id: str) -> str:
        """Extract the SW-Dev probe subtype from the probe_id (e.g. ``impl``)."""
        parts = probe_id.split("-")
        return parts[1] if len(parts) > 1 else "impl"

    def _run_suite(self, answer: str, test_src: str) -> TestResult:
        """Assemble a temp package (base + answer + ``T_t``) and run the sandbox."""
        with tempfile.TemporaryDirectory(prefix="lhmsb-sw-") as tmp:
            work = Path(tmp)
            for rel, source in self._spec.package_files.items():
                path = work / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(source, encoding="utf-8")
            answer_path = work / self._spec.answer_path
            answer_path.parent.mkdir(parents=True, exist_ok=True)
            answer_path.write_text(answer, encoding="utf-8")
            (work / "test_core.py").write_text(test_src, encoding="utf-8")
            return self._sandbox_runner(
                str(work),
                ["test_core.py"],
                time_limit_s=self._time_limit_s,
                mem_limit_mb=self._mem_limit_mb,
                output_limit_bytes=self._output_limit_bytes,
            )

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lhmsb.qualification.preflight import (
    PreflightContext,
    PreflightError,
    PreflightGate,
    redact_secrets,
    require_live_gate,
    run_preflight,
)


def _context(tmp_path: Path) -> PreflightContext:
    return PreflightContext(
        repository_root=tmp_path,
        dataset_root=tmp_path / "dataset",
        config_path=tmp_path / "config.yaml",
        data_root=tmp_path / "data",
        allow_dirty=False,
        repository_only=True,
        environment={
            "OPENAI_API_KEY": "super-secret",
            "ANTHROPIC_API_KEY": "another-secret",
            "LHMSB_QDRANT_URL": "http://qdrant:6333",
        },
    )


def test_preflight_stops_at_first_failure_and_writes_redacted_json(
    tmp_path: Path,
) -> None:
    calls: list[str] = []

    def passed(_: PreflightContext) -> dict[str, object]:
        calls.append("first")
        return {"token": "not-a-secret-token"}

    def failed(_: PreflightContext) -> dict[str, object]:
        calls.append("second")
        raise PreflightError("preflight_failure", "second gate failed")

    def never(_: PreflightContext) -> dict[str, object]:
        calls.append("third")
        return {}

    report_path = tmp_path / "preflight.json"
    report = run_preflight(
        _context(tmp_path),
        gates=(
            PreflightGate("first", "repository", passed),
            PreflightGate("second", "repository", failed),
            PreflightGate("third", "repository", never),
        ),
        output_json=report_path,
    )

    assert report.ok is False
    assert report.stopped_at == "second"
    assert calls == ["first", "second"]
    payload = report_path.read_text(encoding="utf-8")
    assert "super-secret" not in payload
    assert "another-secret" not in payload
    parsed = json.loads(payload)
    assert parsed["checks"][0]["status"] == "pass"
    assert parsed["checks"][1]["error_class"] == "preflight_failure"


def test_repository_only_skips_live_gates(tmp_path: Path) -> None:
    calls: list[str] = []

    def repository(_: PreflightContext) -> dict[str, object]:
        calls.append("repository")
        return {}

    def live(_: PreflightContext) -> dict[str, object]:
        calls.append("live")
        return {}

    report = run_preflight(
        _context(tmp_path),
        gates=(
            PreflightGate("repository", "repository", repository),
            PreflightGate("live", "live", live),
        ),
    )

    assert report.ok is True
    assert calls == ["repository"]
    assert [check.status for check in report.checks] == ["pass", "skip"]


def test_live_gate_requires_explicit_exact_environment_value() -> None:
    with pytest.raises(PreflightError, match="LHMSB_LIVE_QUALIFICATION"):
        require_live_gate({})
    with pytest.raises(PreflightError):
        require_live_gate({"LHMSB_LIVE_QUALIFICATION": "true"})
    require_live_gate({"LHMSB_LIVE_QUALIFICATION": "1"})


def test_recursive_redaction_never_emits_secret_values() -> None:
    value = {
        "api_key": "secret-a",
        "nested": {
            "Authorization": "Bearer secret-b",
            "safe": "visible",
        },
        "required_secret_env": ["OPENAI_API_KEY"],
    }
    redacted = redact_secrets(value)
    rendered = json.dumps(redacted, sort_keys=True)

    assert "secret-a" not in rendered
    assert "secret-b" not in rendered
    assert "visible" in rendered
    assert "OPENAI_API_KEY" in rendered

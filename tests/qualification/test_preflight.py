from __future__ import annotations

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import lhmsb.qualification.preflight as preflight_module
from lhmsb.qualification.preflight import (
    PreflightContext,
    PreflightError,
    PreflightGate,
    _gate_controlled_mem0_lifecycle,
    _gate_host_runtime,
    _host_runtime_inventory,
    _selected_gpu,
    current_repository_snapshot,
    default_preflight_gates,
    redact_secrets,
    require_live_gate,
    run_preflight,
)

ROOT = Path(__file__).resolve().parents[2]


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


def test_repository_snapshot_uses_container_build_manifest_without_git(
    tmp_path: Path,
) -> None:
    (tmp_path / "BUILD.json").write_text(
        json.dumps(
            {
                "commit": "abc123",
                "dirty": False,
                "ref": "feat/mem0-qualification",
            }
        ),
        encoding="utf-8",
    )

    snapshot = current_repository_snapshot(tmp_path)

    assert snapshot.commit == "abc123"
    assert snapshot.dirty is False
    assert snapshot.ref == "feat/mem0-qualification"


def test_containerized_preflight_reads_host_runtime_manifest(
    tmp_path: Path,
) -> None:
    host_manifest = tmp_path / "data" / "manifests" / "host.json"
    host_manifest.parent.mkdir(parents=True)
    host_manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "docker": "Docker version 28.0.0",
                "compose": "Docker Compose version v2.35.0",
                "gpus": [
                    "0, NVIDIA A100-SXM4-80GB, 81920 MiB",
                    "1, NVIDIA A100-SXM4-80GB, 81920 MiB",
                ],
            }
        ),
        encoding="utf-8",
    )
    context = PreflightContext(
        repository_root=tmp_path,
        dataset_root=tmp_path / "dataset",
        config_path=tmp_path / "config.yaml",
        data_root=tmp_path / "data",
        allow_dirty=False,
        repository_only=False,
        environment={
            "LHMSB_CONTAINERIZED": "1",
            "LHMSB_HOST_MANIFEST": str(host_manifest),
        },
    )

    inventory = _host_runtime_inventory(context)

    assert inventory["docker"] == "Docker version 28.0.0"
    assert inventory["compose"] == "Docker Compose version v2.35.0"
    gpus = inventory["gpus"]
    assert isinstance(gpus, list)
    assert len(gpus) == 2


@pytest.mark.parametrize(
    ("gpu_lines", "environment", "match"),
    (
        (
            [
                "0, NVIDIA A100-SXM4-80GB, GPU-a, 81920 MiB",
                "1, NVIDIA H100 80GB HBM3, GPU-b, 81920 MiB",
            ],
            {"LHMSB_REQUIRE_A100": "1", "LHMSB_MIN_FREE_BYTES": "0"},
            "A100",
        ),
        (
            [
                "0, NVIDIA A100-SXM4-80GB, GPU-a, 81920 MiB",
                "1, NVIDIA A100-SXM4-80GB, GPU-b, 81920 MiB",
            ],
            {
                "LHMSB_EMBEDDING_GPU_ID": "0",
                "LHMSB_RERANKER_GPU_ID": "GPU-a",
            },
            "distinct",
        ),
    ),
)
def test_host_runtime_rejects_invalid_selected_a100_pair(
    gpu_lines: list[str],
    environment: dict[str, str],
    match: str,
    tmp_path: Path,
) -> None:
    host_manifest = tmp_path / "data" / "manifests" / "host.json"
    host_manifest.parent.mkdir(parents=True)
    host_manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "docker": "Docker version 28.0.0",
                "compose": "Docker Compose version v2.35.0",
                "gpus": gpu_lines,
            }
        ),
        encoding="utf-8",
    )
    context = PreflightContext(
        repository_root=tmp_path,
        dataset_root=tmp_path / "dataset",
        config_path=tmp_path / "config.yaml",
        data_root=tmp_path / "data",
        allow_dirty=False,
        repository_only=False,
        environment={
            "LHMSB_CONTAINERIZED": "1",
            "LHMSB_HOST_MANIFEST": str(host_manifest),
            "LHMSB_LIVE_PREFLIGHT": "1",
            **environment,
        },
    )

    with pytest.raises(PreflightError, match=match):
        _gate_host_runtime(context)


def test_live_preflight_includes_real_controlled_mem0_lifecycle_gate() -> None:
    names = [gate.name for gate in default_preflight_gates()]

    assert names.index("mem0_runtime_pin") < names.index(
        "controlled_mem0_lifecycle"
    )
    assert names.index("controlled_mem0_lifecycle") < names.index(
        "native_mem0_profile"
    )


def test_selected_gpu_accepts_rtx_4090_in_default_policy() -> None:
    lines = (
        "0, NVIDIA GeForce RTX 4090, GPU-a, 24564 MiB",
        "1, NVIDIA GeForce RTX 4090, GPU-b, 24564 MiB",
    )

    assert _selected_gpu(lines, "0") == lines[0]
    assert _selected_gpu(lines, "GPU-b") == lines[1]


def test_selected_gpu_can_require_a100_for_legacy_deployments() -> None:
    lines = ("0, NVIDIA GeForce RTX 4090, GPU-a, 24564 MiB",)

    with pytest.raises(PreflightError, match="A100"):
        _selected_gpu(lines, "0", require_a100=True)


def test_controlled_mem0_lifecycle_checks_all_policy_profiles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    closed_profiles = 0
    request_apis: list[str | None] = []

    class FakeQdrantClient:
        def __init__(self, **_: object) -> None:
            self.closed = False

        def collection_exists(self, _: str) -> bool:
            return False

        def delete_collection(self, _: str) -> None:
            raise AssertionError("no fake collection should need deletion")

        def close(self) -> None:
            self.closed = True

    class FakeAdapter:
        def close(self) -> None:
            nonlocal closed_profiles
            closed_profiles += 1

        def write_session(self, *_: object, **__: object) -> object:
            return SimpleNamespace(
                inventory=SimpleNamespace(
                    items=(SimpleNamespace(memory_id="memory-1"),),
                    n_live=1,
                ),
                n_write=1,
                usage_events=(
                    SimpleNamespace(component="memory_internal_llm"),
                    SimpleNamespace(component="embedding"),
                ),
            )

        def history_delta(self, *_: object, **__: object) -> tuple[object, ...]:
            return ({"event": "ADD"},)

        def search_candidates(self, *_: object, **__: object) -> object:
            return SimpleNamespace(
                candidates=(SimpleNamespace(memory_id="memory-1"),),
                usage_events=(SimpleNamespace(component="embedding"),),
            )

    def fake_create_live(*args: object, **kwargs: object) -> FakeAdapter:
        request_api = kwargs.get("internal_llm_request_api")
        assert request_api is None or isinstance(request_api, str)
        request_apis.append(request_api)
        return FakeAdapter()

    monkeypatch.setattr(
        preflight_module,
        "build_mem0_live_config",
        lambda *args, **kwargs: {},
    )
    monkeypatch.setattr(
        "lhmsb.qualification.preflight.Mem0QualificationAdapter.create_live",
        fake_create_live,
    )
    real_import = importlib.import_module
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name: (
            SimpleNamespace(QdrantClient=FakeQdrantClient)
            if name == "qdrant_client"
            else real_import(name)
        ),
    )
    context = PreflightContext(
        repository_root=ROOT,
        dataset_root=ROOT / "runs" / "vertical" / "software_mem0_v2",
        config_path=ROOT
        / "configs"
        / "experiments"
        / "mem0_qualification.yaml",
        data_root=tmp_path / "data",
        allow_dirty=False,
        repository_only=False,
        environment={
            "ANTHROPIC_API_KEY": "secret-a",
            "DEEPSEEK_API_KEY": "secret-d",
            "OPENAI_API_KEY": "secret-o",
        },
    )

    result = _gate_controlled_mem0_lifecycle(context)

    profiles = result["profiles"]
    assert isinstance(profiles, list)
    assert [item["profile_id"] for item in profiles] == [
        "opus_4_8",
        "deepseek_v4_pro",
        "gpt_5_6_sol",
    ]
    assert request_apis == ["messages", "chat_completions", "responses"]
    assert closed_profiles == 3

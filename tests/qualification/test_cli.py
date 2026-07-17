from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

import lhmsb.qualification.cli as cli_module
from lhmsb.families.software.mem0_vertical import (
    SoftwareMem0VerticalFamily,
    SoftwareMem0VerticalSpec,
)
from lhmsb.longhorizon.interventions import ContinuationOutcome
from lhmsb.qualification.cli import (
    _live_components,
    _qdrant_collection_count,
    _qdrant_collection_snapshot_size,
    _sqlite_store_size,
    main,
)
from lhmsb.qualification.config import load_qualification_config
from lhmsb.qualification.report import write_qualification_report
from lhmsb.qualification.runner import (
    ConditionRunResult,
    QualificationMatrixResult,
    QualificationTaskResult,
    SCEURunResult,
    TaskIsolation,
)
from lhmsb.qualification.schema import QualificationTask
from lhmsb.qualification.validate import validate_qualification_artifacts

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "experiments" / "mem0_qualification.yaml"
DATASET_RELEASE = (
    ROOT
    / "datasets"
    / "releases"
    / "software-vertical-mem0-v0.2.0"
    / "software_mem0_v2.tar.gz"
)


@pytest.mark.parametrize(
    ("condition", "expected_request_api"),
    (("mem0_controlled", "responses"), ("mem0_native", None)),
)
def test_live_components_routes_internal_llm_api_only_for_controlled_track(
    condition: str,
    expected_request_api: str | None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[str | None] = []

    def fake_create_live(*args: object, **kwargs: object) -> object:
        request_api = kwargs.get("internal_llm_request_api")
        assert request_api is None or isinstance(request_api, str)
        seen.append(request_api)
        return SimpleNamespace()

    monkeypatch.setattr(
        cli_module,
        "HttpPolicyClient",
        lambda *args, **kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(
        cli_module,
        "build_mem0_live_config",
        lambda *args, **kwargs: {},
    )
    monkeypatch.setattr(
        "lhmsb.qualification.cli.Mem0QualificationAdapter.create_live",
        fake_create_live,
    )
    monkeypatch.setattr(
        cli_module,
        "RerankerClient",
        lambda *args, **kwargs: SimpleNamespace(),
    )
    config = load_qualification_config(CONFIG)
    task = SimpleNamespace(
        policy_profile_id="gpt_5_6_sol",
        condition=condition,
    )
    isolation = SimpleNamespace(
        history_db_path=tmp_path / "history.sqlite",
        collection_name="collection",
        user_id="user",
        run_id="run",
    )
    spec = SimpleNamespace(
        plan=None,
        package_files=(),
        hidden_tests=(),
        actions=(),
        surface_hash="surface",
    )

    _live_components(
        cast(QualificationTask, task),
        cast(TaskIsolation, isolation),
        spec=cast(SoftwareMem0VerticalSpec, spec),
        config=config,
        environment={"OPENAI_API_KEY": "not-a-real-secret"},
    )

    assert seen == [expected_request_api]


@pytest.fixture(scope="session")
def qualification_dataset(tmp_path_factory: pytest.TempPathFactory) -> Path:
    release_root = tmp_path_factory.mktemp("mem0-qualification-dataset")
    with tarfile.open(DATASET_RELEASE, "r:gz") as archive:
        if hasattr(tarfile, "data_filter"):
            archive.extractall(release_root, filter="data")
        else:  # pragma: no cover - extraction filters predate supported CI images
            archive.extractall(release_root)
    return release_root / "software_mem0_v2"


def test_module_help_lists_all_qualification_commands() -> None:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT / "src")
    completed = subprocess.run(
        [sys.executable, "-m", "lhmsb.qualification", "--help"],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0
    for command in (
        "plan",
        "run-task",
        "run-matrix",
        "aggregate",
        "validate",
        "preflight",
        "smoke",
    ):
        assert command in completed.stdout


def test_plan_writes_redacted_identity_bound_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    qualification_dataset: Path,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-be-written")
    monkeypatch.setenv(
        "OPENAI_BASE_URL",
        "https://gateway.example/openai/v1",
    )
    data_root = tmp_path / "data"
    (data_root / "manifests").mkdir(parents=True)
    (data_root / "runs" / "preflight").mkdir(parents=True)
    image_manifest = data_root / "manifests" / "images.json"
    model_manifest = data_root / "manifests" / "models.json"
    image_manifest.write_text(
        '{"qdrant":"sha256:image"}\n',
        encoding="utf-8",
    )
    model_manifest.write_text(
        '{"files":{},"revisions":{}}\n',
        encoding="utf-8",
    )
    (data_root / "runs" / "preflight" / "latest.json").write_text(
        json.dumps(
            {
                "checks": [
                    {
                        "name": "host_and_gpu_runtime",
                        "status": "pass",
                        "details": {
                            "gpus": [
                                "0, NVIDIA A100-SXM4-80GB, 81920 MiB",
                                "1, NVIDIA A100-SXM4-80GB, 81920 MiB",
                            ]
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("LHMSB_DATA_ROOT", str(data_root))
    report_path = tmp_path / "plan.json"
    run_dir = tmp_path / "run"

    status = main(
        [
            "plan",
            "--dataset",
            str(qualification_dataset),
            "--config",
            str(CONFIG),
            "--out",
            str(run_dir),
            "--allow-dirty",
            "--json",
            str(report_path),
        ]
    )

    assert status == 0
    manifest_text = (run_dir / "run_manifest.json").read_text(encoding="utf-8")
    assert "must-not-be-written" not in manifest_text
    manifest = json.loads(manifest_text)
    assert manifest["task_count"] == 12
    assert manifest["image_digests_hash"] == hashlib.sha256(
        image_manifest.read_bytes()
    ).hexdigest()
    assert manifest["model_files_hash"] == hashlib.sha256(
        model_manifest.read_bytes()
    ).hexdigest()
    assert len(manifest["hardware_profile_hash"]) == 64
    assert len(manifest["effective_policy_profiles_hash"]) == 64
    assert manifest["required_secret_env"] == [
        "ANTHROPIC_API_KEY",
        "DEEPSEEK_API_KEY",
        "OPENAI_API_KEY",
    ]
    assert len((run_dir / "tasks.jsonl").read_text().splitlines()) == 12
    assert json.loads(report_path.read_text())["run_identity"] == manifest["run_identity"]


def test_run_task_dry_run_needs_no_credentials_but_live_execution_is_gated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    qualification_dataset: Path,
) -> None:
    run_dir = tmp_path / "run"
    assert (
        main(
            [
                "plan",
                "--dataset",
                str(qualification_dataset),
                "--config",
                str(CONFIG),
                "--out",
                str(run_dir),
                "--allow-dirty",
            ]
        )
        == 0
    )
    monkeypatch.delenv("LHMSB_LIVE_QUALIFICATION", raising=False)

    assert (
        main(
            [
                "run-task",
                "--run-dir",
                str(run_dir),
                "--task-index",
                "0",
                "--dry-run",
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "run-task",
                "--run-dir",
                str(run_dir),
                "--task-index",
                "0",
            ]
        )
        == 2
    )
    assert "LHMSB_LIVE_QUALIFICATION" in capsys.readouterr().err


def test_run_contract_identity_mismatch_is_rejected(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    qualification_dataset: Path,
) -> None:
    run_dir = tmp_path / "run"
    assert (
        main(
            [
                "plan",
                "--dataset",
                str(qualification_dataset),
                "--config",
                str(CONFIG),
                "--out",
                str(run_dir),
                "--allow-dirty",
            ]
        )
        == 0
    )
    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["run_identity"] = "tampered"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    status = main(
        [
            "run-task",
            "--run-dir",
            str(run_dir),
            "--task-index",
            "0",
            "--dry-run",
        ]
    )

    assert status == 2
    assert "identity" in capsys.readouterr().err.casefold()


def test_run_identity_changes_when_the_persistent_data_root_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    qualification_dataset: Path,
) -> None:
    identities: list[str] = []
    for index in range(2):
        monkeypatch.setenv(
            "LHMSB_DATA_ROOT",
            str(tmp_path / f"data-{index}"),
        )
        run_dir = tmp_path / f"run-{index}"
        assert (
            main(
                [
                    "plan",
                    "--dataset",
                    str(qualification_dataset),
                    "--config",
                    str(CONFIG),
                    "--out",
                    str(run_dir),
                    "--allow-dirty",
                ]
            )
            == 0
        )
        manifest = json.loads(
            (run_dir / "run_manifest.json").read_text(encoding="utf-8")
        )
        identities.append(manifest["run_identity"])

    assert identities[0] != identities[1]


def test_run_identity_changes_when_a_provider_endpoint_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    qualification_dataset: Path,
) -> None:
    identities: list[str] = []
    monkeypatch.setenv("LHMSB_DATA_ROOT", str(tmp_path / "data"))
    for index, endpoint in enumerate(
        (
            "https://api.openai.com",
            "https://gateway.example/openai/v1",
        )
    ):
        monkeypatch.setenv("OPENAI_BASE_URL", endpoint)
        run_dir = tmp_path / f"run-endpoint-{index}"
        assert (
            main(
                [
                    "plan",
                    "--dataset",
                    str(qualification_dataset),
                    "--config",
                    str(CONFIG),
                    "--out",
                    str(run_dir),
                    "--allow-dirty",
                ]
            )
            == 0
        )
        manifest = json.loads(
            (run_dir / "run_manifest.json").read_text(encoding="utf-8")
        )
        identities.append(manifest["run_identity"])

    assert identities[0] != identities[1]


def test_resume_rejects_a_changed_provider_endpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    qualification_dataset: Path,
) -> None:
    monkeypatch.setenv("LHMSB_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com")
    run_dir = tmp_path / "run"
    assert (
        main(
            [
                "plan",
                "--dataset",
                str(qualification_dataset),
                "--config",
                str(CONFIG),
                "--out",
                str(run_dir),
                "--allow-dirty",
            ]
        )
        == 0
    )

    monkeypatch.setenv(
        "OPENAI_BASE_URL",
        "https://gateway.example/openai/v1",
    )
    status = main(
        [
            "run-task",
            "--run-dir",
            str(run_dir),
            "--task-index",
            "0",
            "--dry-run",
        ]
    )

    assert status == 2
    assert "provider request profiles" in capsys.readouterr().err


def test_live_plan_requires_a_complete_successful_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    qualification_dataset: Path,
) -> None:
    data_root = tmp_path / "data"
    manifests = data_root / "manifests"
    preflight = data_root / "runs" / "preflight"
    manifests.mkdir(parents=True)
    preflight.mkdir(parents=True)
    (manifests / "images.json").write_text(
        '{"qdrant":"sha256:image"}\n',
        encoding="utf-8",
    )
    (manifests / "models.json").write_text(
        '{"files":{},"revisions":{}}\n',
        encoding="utf-8",
    )
    (preflight / "latest.json").write_text(
        json.dumps(
            {
                "ok": False,
                "stopped_at": "provider_structured_smoke",
                "repository_only": False,
                "checks": [
                    {
                        "name": "host_and_gpu_runtime",
                        "status": "pass",
                        "details": {
                            "gpus": [
                                "0, NVIDIA A100-SXM4-80GB, 81920 MiB",
                                "1, NVIDIA A100-SXM4-80GB, 81920 MiB",
                            ]
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("LHMSB_DATA_ROOT", str(data_root))
    monkeypatch.setenv("LHMSB_LIVE_QUALIFICATION", "1")

    status = main(
        [
            "plan",
            "--dataset",
            str(qualification_dataset),
            "--config",
            str(CONFIG),
            "--out",
            str(tmp_path / "run"),
            "--allow-dirty",
        ]
    )

    assert status == 2
    assert "successful full preflight" in capsys.readouterr().err


def _tiny_report(tmp_path: Path) -> Path:
    spec = SoftwareMem0VerticalFamily.generate(42, n_sessions=4)
    sceu = spec.plan.sceu_units[0]
    row = SCEURunResult(
        result_id="workspace-result",
        sceu_id=sceu.sceu_id,
        opportunity_id=sceu.opportunity_id,
        checkpoint_session=sceu.checkpoint_session,
        matched_group=sceu.matched_group,
        control_kind="workspace",
        workspace_hash="workspace-hash",
        candidate_memory_ids=(),
        retrieved_memory_ids=(),
        model_visible_memory_ids=(),
        selected_option_id="option-03",
        selected_action_id="safe_v2_offline",
        behavior=ContinuationOutcome(
            action_id="safe_v2_offline",
            behavior_score=1.0,
            is_correct=True,
        ),
        normalized_drift_flags=(),
        baseline_stable=True,
        baseline_evaluations=(),
        interventions=(),
        retrieval_trace_id=None,
    )
    condition = ConditionRunResult(
        result_id="workspace-result",
        condition="workspace_only",
        readout="none",
        status="complete",
        sceu_results=(row,),
    )
    task = QualificationTaskResult(
        task_id="task-001",
        episode_id=spec.plan.episode_id,
        policy_profile_id="policy-a",
        condition="workspace_only",
        status="complete",
        condition_results=(condition,),
        writes=(),
        alignments=(),
        retrieval_traces=(),
    )
    out = tmp_path / "report"
    write_qualification_report(
        QualificationMatrixResult("run-identity", (task,)),
        {spec.plan.episode_id: spec},
        out,
    )
    return out


def test_validate_command_checks_hashes_and_trace_ordering(
    tmp_path: Path,
) -> None:
    report = _tiny_report(tmp_path)
    valid = validate_qualification_artifacts(
        report,
        expected_run_identity="run-identity",
    )
    assert valid.ok is True
    assert main(["validate", "--report", str(report)]) == 0

    sceu_path = report / "sceu_results.jsonl"
    row = json.loads(sceu_path.read_text())
    row["candidate_memory_ids"] = ["memory-a"]
    row["retrieved_memory_ids"] = ["memory-b"]
    sceu_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    invalid = validate_qualification_artifacts(report)
    assert invalid.ok is False
    assert any("retrieved" in error for error in invalid.errors)
    assert main(["validate", "--report", str(report)]) == 1


def test_validation_requires_metrics_for_every_condition_cell(
    tmp_path: Path,
) -> None:
    report = _tiny_report(tmp_path)
    metrics_path = report / "metrics_by_cell.json"
    metrics_path.write_text(
        '{"groups":[],"schema_version":2}\n',
        encoding="utf-8",
    )
    manifest_path = report / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifact_hashes"]["metrics_by_cell.json"] = hashlib.sha256(
        metrics_path.read_bytes()
    ).hexdigest()
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    invalid = validate_qualification_artifacts(report)

    assert invalid.ok is False
    assert any("metrics_by_cell coverage" in error for error in invalid.errors)


def test_qdrant_collection_count_uses_exact_task_collection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class Response:
        def __enter__(self) -> Response:
            return self

        def __exit__(self, *_: object) -> None:
            return None

        def read(self) -> bytes:
            return b'{"result":{"count":7},"status":"ok"}'

    def urlopen(request: object, *, timeout: float) -> Response:
        captured["url"] = request.full_url  # type: ignore[attr-defined]
        captured["data"] = request.data  # type: ignore[attr-defined]
        captured["timeout"] = timeout
        return Response()

    monkeypatch.setattr("urllib.request.urlopen", urlopen)

    count = _qdrant_collection_count(
        "http://qdrant:6333",
        "run--task",
    )

    assert count == 7
    assert captured == {
        "url": "http://qdrant:6333/collections/run--task/points/count",
        "data": b'{"exact":true}',
        "timeout": 30.0,
    }


def test_qdrant_snapshot_size_is_measured_and_snapshot_is_deleted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, object]] = []

    class FakeQdrantClient:
        def __init__(self, *, url: str, timeout: float) -> None:
            calls.append(("init", (url, timeout)))

        def create_snapshot(
            self,
            *,
            collection_name: str,
            wait: bool,
        ) -> object:
            calls.append(("create", (collection_name, wait)))
            return SimpleNamespace(name="task.snapshot", size=8192)

        def delete_snapshot(
            self,
            *,
            collection_name: str,
            snapshot_name: str,
            wait: bool,
        ) -> None:
            calls.append(
                ("delete", (collection_name, snapshot_name, wait))
            )

        def close(self) -> None:
            calls.append(("close", None))

    monkeypatch.setitem(
        sys.modules,
        "qdrant_client",
        SimpleNamespace(QdrantClient=FakeQdrantClient),
    )

    size = _qdrant_collection_snapshot_size(
        "http://qdrant:6333",
        "run--task",
    )

    assert size == 8192
    assert calls == [
        ("init", ("http://qdrant:6333", 60.0)),
        ("create", ("run--task", True)),
        ("delete", ("run--task", "task.snapshot", True)),
        ("close", None),
    ]


def test_sqlite_store_size_includes_wal_and_shared_memory(
    tmp_path: Path,
) -> None:
    history = tmp_path / "history.sqlite"
    history.write_bytes(b"a" * 11)
    Path(f"{history}-wal").write_bytes(b"b" * 13)
    Path(f"{history}-shm").write_bytes(b"c" * 17)

    assert _sqlite_store_size(history) == 41
    assert _sqlite_store_size(tmp_path / "missing.sqlite") == 0

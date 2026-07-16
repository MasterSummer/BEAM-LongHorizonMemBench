"""Ordered, stop-on-first-failure qualification preflight gates."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Literal, cast

from lhmsb.datasets.mem0_stateful_pipeline import (
    Mem0StatefulDatasetError,
    regen_check_mem0_stateful,
    verify_mem0_stateful,
)
from lhmsb.families.software.mem0_vertical import SoftwareMem0VerticalSpec
from lhmsb.longhorizon.public_surface import (
    EvaluatorContinuation,
    PublicActionOption,
    PublicContinuation,
    SurfaceLeakPolicy,
    validate_public_payload,
)
from lhmsb.longhorizon.schema import ActionSpec, EpisodePlan
from lhmsb.qualification.config import load_qualification_config
from lhmsb.qualification.providers import (
    HttpPolicyClient,
    PolicyMessage,
    PolicyRequest,
)
from lhmsb.qualification.schema import PolicyProfile
from lhmsb.qualification.tei import (
    EmbeddingClient,
    RerankCandidate,
    RerankerClient,
)

PreflightScope = Literal["repository", "live"]
PreflightStatus = Literal["pass", "fail", "skip"]
PreflightProbe = Callable[["PreflightContext"], Mapping[str, object] | None]
_SECRET_MARKERS = (
    "api_key",
    "apikey",
    "authorization",
    "credential",
    "password",
    "secret",
    "token",
)


class PreflightError(RuntimeError):
    """Typed gate failure suitable for a machine-readable report."""

    def __init__(self, error_class: str, message: str) -> None:
        super().__init__(message)
        self.error_class = error_class


@dataclass(frozen=True)
class RepositorySnapshot:
    commit: str
    dirty: bool
    ref: str


@dataclass(frozen=True)
class PreflightContext:
    repository_root: Path
    dataset_root: Path
    config_path: Path
    data_root: Path
    allow_dirty: bool
    repository_only: bool
    environment: Mapping[str, str]


@dataclass(frozen=True)
class PreflightGate:
    name: str
    scope: PreflightScope
    probe: PreflightProbe


@dataclass(frozen=True)
class PreflightCheck:
    name: str
    scope: PreflightScope
    status: PreflightStatus
    error_class: str | None = None
    message: str | None = None
    details: Mapping[str, object] | None = None


@dataclass(frozen=True)
class PreflightReport:
    ok: bool
    stopped_at: str | None
    checks: tuple[PreflightCheck, ...]
    repository_only: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "stopped_at": self.stopped_at,
            "repository_only": self.repository_only,
            "checks": [asdict(check) for check in self.checks],
        }


def run_preflight(
    context: PreflightContext,
    *,
    gates: Sequence[PreflightGate] | None = None,
    output_json: Path | None = None,
) -> PreflightReport:
    """Execute gates in declaration order and stop after the first failure."""
    checks: list[PreflightCheck] = []
    stopped_at: str | None = None
    for gate in gates or default_preflight_gates():
        if context.repository_only and gate.scope == "live":
            checks.append(
                PreflightCheck(
                    name=gate.name,
                    scope=gate.scope,
                    status="skip",
                    message="repository-only preflight",
                )
            )
            continue
        try:
            details = gate.probe(context)
        except Exception as exc:
            stopped_at = gate.name
            checks.append(
                PreflightCheck(
                    name=gate.name,
                    scope=gate.scope,
                    status="fail",
                    error_class=(
                        exc.error_class
                        if isinstance(exc, PreflightError)
                        else "preflight_failure"
                    ),
                    message=_redact_text(str(exc), context.environment),
                )
            )
            break
        checks.append(
            PreflightCheck(
                name=gate.name,
                scope=gate.scope,
                status="pass",
                details=cast(
                    Mapping[str, object],
                    redact_secrets(dict(details or {})),
                ),
            )
        )
    report = PreflightReport(
        ok=stopped_at is None,
        stopped_at=stopped_at,
        checks=tuple(checks),
        repository_only=context.repository_only,
    )
    if output_json is not None:
        _atomic_json(output_json, redact_secrets(report.to_dict()))
    return report


def default_preflight_gates() -> tuple[PreflightGate, ...]:
    """Return the immutable repository-first qualification gate order."""
    return (
        PreflightGate("repository_and_config", "repository", _gate_repository_config),
        PreflightGate("legacy_release_v0_1", "repository", _gate_legacy_release),
        PreflightGate("mem0_v0_2_regeneration", "repository", _gate_mem0_regeneration),
        PreflightGate("public_surface_firewall", "repository", _gate_public_firewall),
        PreflightGate("mem0_v0_2_archive", "repository", _gate_mem0_release),
        PreflightGate("dependency_and_system_locks", "repository", _gate_dependency_locks),
        PreflightGate("host_and_gpu_runtime", "live", _gate_host_runtime),
        PreflightGate("oci_image_digests", "live", _gate_image_digests),
        PreflightGate("local_model_files", "live", _gate_model_files),
        PreflightGate("qdrant_isolation", "live", _gate_qdrant),
        PreflightGate("tei_embedding_dimension", "live", _gate_embedding),
        PreflightGate("tei_reranker_order", "live", _gate_reranker),
        PreflightGate("provider_credentials", "live", _gate_provider_credentials),
        PreflightGate("provider_structured_smoke", "live", _gate_provider_smoke),
        PreflightGate("mem0_runtime_pin", "live", _gate_mem0_runtime),
        PreflightGate("native_mem0_profile", "live", _gate_native_profile),
        PreflightGate("trace_and_prompt_contract", "live", _gate_trace_contract),
    )


def require_live_gate(
    environment: Mapping[str, str],
    *,
    variable: str = "LHMSB_LIVE_QUALIFICATION",
) -> None:
    """Require an exact opt-in value before any paid or mutating execution."""
    if environment.get(variable) != "1":
        raise PreflightError(
            "preflight_failure",
            f"set {variable}=1 to authorize live qualification execution",
        )


def current_repository_snapshot(root: Path) -> RepositorySnapshot:
    """Inspect Git without importing the legacy experiment runtime."""
    commit = _git_output(root, ("rev-parse", "HEAD"))
    status = _git_output(
        root,
        ("status", "--porcelain", "--untracked-files=normal"),
    )
    try:
        ref = _git_output(
            root,
            ("symbolic-ref", "--short", "-q", "HEAD"),
        )
    except PreflightError:
        ref = "detached"
    return RepositorySnapshot(
        commit=commit,
        dirty=bool(status),
        ref=ref,
    )


def redact_secrets(value: object) -> object:
    """Recursively redact values whose field names carry secret semantics."""
    if isinstance(value, Mapping):
        output: dict[str, object] = {}
        for raw_key, child in value.items():
            key = str(raw_key)
            lowered = key.casefold()
            if any(marker in lowered for marker in _SECRET_MARKERS):
                if lowered in {"required_secret_env", "required_secret_envs"}:
                    output[key] = redact_secrets(child)
                else:
                    output[key] = "<redacted>"
            else:
                output[key] = redact_secrets(child)
        return output
    if isinstance(value, (tuple, list)):
        return [redact_secrets(child) for child in value]
    return value


def load_mem0_specs(
    frozen: Path,
    *,
    verify: bool = True,
) -> tuple[SoftwareMem0VerticalSpec, ...]:
    """Load evaluator specs only after validating the frozen release."""
    if verify:
        report = verify_mem0_stateful(frozen)
        if not report.ok:
            raise Mem0StatefulDatasetError(
                f"frozen Mem0 dataset failed verification: "
                f"missing={report.missing}, mismatches={report.mismatches}"
            )
    records = _read_jsonl(frozen / "evaluator" / "episodes.jsonl")
    specs: list[SoftwareMem0VerticalSpec] = []
    for record in records:
        plan = EpisodePlan.from_dict(_mapping(record.get("plan"), "plan"))
        actions = tuple(
            ActionSpec.from_dict(_mapping(item, "action"))
            for item in _sequence(record.get("actions"), "actions")
        )
        public_root = frozen / "public" / plan.episode_id / "continuation"
        public = tuple(
            PublicContinuation.from_dict(_read_json(path))
            for path in sorted(public_root.glob("*.json"))
        )
        evaluator = tuple(
            _evaluator_continuation(item)
            for item in _sequence(
                record.get("evaluator_continuations"),
                "evaluator_continuations",
            )
        )
        specs.append(
            SoftwareMem0VerticalSpec(
                plan=plan,
                package_files=_pairs(record.get("package_files"), "package_files"),
                hidden_tests=_pairs(record.get("hidden_tests"), "hidden_tests"),
                actions=actions,
                public_continuations=public,
                evaluator_continuations=evaluator,
                surface_hash=str(record["surface_hash"]),
            )
        )
    return tuple(specs)


def _gate_repository_config(context: PreflightContext) -> dict[str, object]:
    snapshot = current_repository_snapshot(context.repository_root)
    if snapshot.dirty and not context.allow_dirty:
        raise PreflightError(
            "preflight_failure",
            "Git worktree is dirty; commit changes or pass --allow-dirty",
        )
    config = load_qualification_config(context.config_path)
    return {
        "code_commit": snapshot.commit,
        "code_dirty": snapshot.dirty,
        "code_ref": snapshot.ref,
        "config_hash": config.config_hash,
        "required_secret_env": list(config.required_secret_env),
    }


def _gate_legacy_release(context: PreflightContext) -> dict[str, object]:
    release = context.repository_root / "datasets/releases/software-vertical-v0.1.0"
    return _verify_release(release)


def _gate_mem0_regeneration(context: PreflightContext) -> dict[str, object]:
    verify = verify_mem0_stateful(context.dataset_root)
    if not verify.ok:
        raise PreflightError(
            "preflight_failure",
            f"Mem0 frozen dataset verification failed: {verify}",
        )
    regen = regen_check_mem0_stateful(context.dataset_root)
    if not regen.ok:
        raise PreflightError(
            "preflight_failure",
            f"Mem0 regeneration mismatch: {regen.mismatches}",
        )
    return {
        "checked_files": verify.n_checked,
        "checked_episodes": regen.checked,
    }


def _gate_public_firewall(context: PreflightContext) -> dict[str, object]:
    specs = load_mem0_specs(context.dataset_root)
    scanned = 0
    for spec in specs:
        policy = SurfaceLeakPolicy(
            forbidden_state_ids=tuple(
                state.state_id for state in spec.plan.state_units
            ),
            forbidden_action_ids=tuple(
                action.action_id for action in spec.actions
            ),
            answer_revealing_phrases=(
                "correct action",
                "globally correct",
                "accepted action",
            ),
        )
        report = validate_public_payload(
            {
                "sessions": spec.public_session_dicts,
                "continuations": [
                    item.to_dict() for item in spec.public_continuations
                ],
            },
            policy,
        )
        scanned += report.scanned_strings
        _audit_recoverability(spec)
    return {"episodes": len(specs), "scanned_strings": scanned}


def _gate_mem0_release(context: PreflightContext) -> dict[str, object]:
    release = (
        context.repository_root
        / "datasets/releases/software-vertical-mem0-v0.2.0"
    )
    details = _verify_release(release)
    release_json = _read_json(release / "RELEASE.json")
    manifest_path = context.dataset_root / "MANIFEST.json"
    manifest_sha = _sha256(manifest_path)
    if manifest_sha != str(release_json.get("dataset_manifest_sha256", "")):
        raise PreflightError(
            "preflight_failure",
            "unpacked Mem0 manifest does not match RELEASE.json",
        )
    return {**details, "dataset_manifest_sha256": manifest_sha}


def _gate_dependency_locks(context: PreflightContext) -> dict[str, object]:
    uv_lock = context.repository_root / "uv.lock"
    systems_lock = context.repository_root / "configs/systems.lock.yaml"
    if not uv_lock.is_file() or not systems_lock.is_file():
        raise PreflightError(
            "preflight_failure",
            "uv.lock and configs/systems.lock.yaml are required",
        )
    lock_text = uv_lock.read_text(encoding="utf-8")
    if 'name = "mem0ai"' not in lock_text or 'version = "2.0.12"' not in lock_text:
        raise PreflightError(
            "preflight_failure",
            "uv.lock does not contain the exact Mem0 2.0.12 pin",
        )
    systems = systems_lock.read_text(encoding="utf-8")
    required = (
        "version: 2.0.12",
        "42cf18c4e6adb448e981aa1c7b55c1602b0cb670",
        "6b7e1afa466f6e14dd34b5e9222c159a69fad38f8d787e73adbf91dbb29e73e2",
    )
    if not all(item in systems for item in required):
        raise PreflightError(
            "preflight_failure",
            "systems.lock.yaml does not match the declared Mem0 pin",
        )
    return {
        "uv_lock_sha256": _sha256(uv_lock),
        "systems_lock_sha256": _sha256(systems_lock),
    }


def _gate_host_runtime(context: PreflightContext) -> dict[str, object]:
    require_live_gate(
        context.environment,
        variable="LHMSB_LIVE_PREFLIGHT",
    )
    docker = _command_output(("docker", "--version"))
    compose = _command_output(("docker", "compose", "version"))
    gpu_lines = _command_output(
        (
            "nvidia-smi",
            "--query-gpu=index,name,memory.total",
            "--format=csv,noheader",
        )
    ).splitlines()
    if len(gpu_lines) < 2:
        raise PreflightError(
            "preflight_failure",
            f"at least two NVIDIA GPUs are required, found {len(gpu_lines)}",
        )
    context.data_root.mkdir(parents=True, exist_ok=True)
    probe = context.data_root / ".lhmsb-write-probe"
    probe.write_text("ok\n", encoding="utf-8")
    probe.unlink()
    free_bytes = shutil.disk_usage(context.data_root).free
    minimum = int(
        context.environment.get("LHMSB_MIN_FREE_BYTES", str(50 * 1024**3))
    )
    if free_bytes < minimum:
        raise PreflightError(
            "preflight_failure",
            f"insufficient free disk: {free_bytes} < {minimum}",
        )
    wheel = Path(
        context.environment.get(
            "LHMSB_MEM0_WHEEL",
            str(context.data_root / "wheelhouse/mem0ai-2.0.12-py3-none-any.whl"),
        )
    )
    if not wheel.is_file():
        raise PreflightError("preflight_failure", f"missing Mem0 wheel: {wheel}")
    expected = "6b7e1afa466f6e14dd34b5e9222c159a69fad38f8d787e73adbf91dbb29e73e2"
    if _sha256(wheel) != expected:
        raise PreflightError("preflight_failure", "Mem0 wheel hash mismatch")
    return {
        "docker": docker,
        "compose": compose,
        "gpus": gpu_lines,
        "free_bytes": free_bytes,
        "mem0_wheel_sha256": expected,
    }


def _gate_image_digests(context: PreflightContext) -> dict[str, object]:
    manifest = Path(
        context.environment.get(
            "LHMSB_IMAGE_DIGEST_MANIFEST",
            str(context.data_root / "manifests/images.json"),
        )
    )
    data = _read_json(manifest)
    if not data or not all(
        isinstance(value, str) and value.startswith("sha256:")
        for value in data.values()
    ):
        raise PreflightError(
            "preflight_failure",
            "image digest manifest must map every image to sha256:<digest>",
        )
    return {"manifest_sha256": _sha256(manifest), "image_count": len(data)}


def _gate_model_files(context: PreflightContext) -> dict[str, object]:
    manifest = Path(
        context.environment.get(
            "LHMSB_MODEL_FILE_MANIFEST",
            str(context.data_root / "manifests/models.json"),
        )
    )
    data = _read_json(manifest)
    files = _mapping(data.get("files"), "model files")
    for raw_path, expected in files.items():
        path = Path(raw_path)
        if not path.is_absolute():
            path = context.data_root / path
        if not path.is_file() or _sha256(path) != str(expected):
            raise PreflightError(
                "preflight_failure",
                f"model file hash mismatch: {path}",
            )
    config = load_qualification_config(context.config_path)
    revisions = json.dumps(data.get("revisions", {}), sort_keys=True)
    for revision in (
        config.retrieval.embedding_revision,
        config.retrieval.reranker_revision,
    ):
        if revision not in revisions:
            raise PreflightError(
                "preflight_failure",
                f"model manifest lacks revision {revision}",
            )
    return {"manifest_sha256": _sha256(manifest), "file_count": len(files)}


def _gate_qdrant(context: PreflightContext) -> dict[str, object]:
    try:
        qdrant_client = importlib.import_module("qdrant_client")
        models = importlib.import_module("qdrant_client.models")
    except ImportError as exc:
        raise PreflightError(
            "preflight_failure",
            "qdrant-client is not installed",
        ) from exc
    url = context.environment.get("LHMSB_QDRANT_URL", "http://qdrant:6333")
    client = qdrant_client.QdrantClient(url=url, timeout=30)
    collection = "lhmsb_preflight_isolation"
    try:
        if client.collection_exists(collection):
            client.delete_collection(collection)
        client.create_collection(
            collection,
            vectors_config=models.VectorParams(
                size=4,
                distance=models.Distance.COSINE,
            ),
        )
        client.upsert(
            collection,
            points=[
                models.PointStruct(
                    id=1,
                    vector=[1.0, 0.0, 0.0, 0.0],
                    payload={"scope": "preflight"},
                )
            ],
            wait=True,
        )
        rows = client.query_points(
            collection,
            query=[1.0, 0.0, 0.0, 0.0],
            limit=1,
        ).points
        if len(rows) != 1:
            raise PreflightError(
                "preflight_failure",
                "Qdrant write/search isolation probe returned no point",
            )
    finally:
        if client.collection_exists(collection):
            client.delete_collection(collection)
        client.close()
    return {"url": url, "isolation_probe": "pass"}


def _gate_embedding(context: PreflightContext) -> dict[str, object]:
    config = load_qualification_config(context.config_path)
    client = EmbeddingClient(
        context.environment.get(
            "LHMSB_EMBEDDING_URL",
            "http://embedding:80",
        ),
        model=config.retrieval.embedding_model,
        revision=config.retrieval.embedding_revision,
        expected_dimension=config.retrieval.embedding_dimension,
    )
    try:
        health = client.health()
        if not health.ok:
            raise PreflightError(
                "preflight_failure",
                f"embedding health failed: {health.status_code}",
            )
        batch = client.embed(("offline pipeline",))
    finally:
        client.close()
    return {
        "dimension": batch.dimension,
        "revision": batch.revision,
        "response_hash": batch.response_hash,
    }


def _gate_reranker(context: PreflightContext) -> dict[str, object]:
    config = load_qualification_config(context.config_path)
    client = RerankerClient(
        context.environment.get(
            "LHMSB_RERANKER_URL",
            "http://reranker:80",
        ),
        model=config.retrieval.reranker_model,
        revision=config.retrieval.reranker_revision,
    )
    candidates = (
        RerankCandidate("offline", "fully offline pipeline", 1),
        RerankCandidate("cloud", "cloud shortcut", 2),
    )
    try:
        health = client.health()
        if not health.ok:
            raise PreflightError(
                "preflight_failure",
                f"reranker health failed: {health.status_code}",
            )
        first = client.rerank("offline execution", candidates, top_k=2)
        second = client.rerank("offline execution", candidates, top_k=2)
    finally:
        client.close()
    if (
        first.ordered_memory_ids != second.ordered_memory_ids
        or first.scores != second.scores
    ):
        raise PreflightError(
            "preflight_failure",
            "reranker fixed fixture is nondeterministic",
        )
    return {
        "revision": first.revision,
        "ordered_memory_ids": list(first.ordered_memory_ids),
    }


def _gate_provider_credentials(context: PreflightContext) -> dict[str, object]:
    config = load_qualification_config(context.config_path)
    missing = [
        name
        for name in config.required_secret_env
        if not context.environment.get(name)
    ]
    if missing:
        raise PreflightError(
            "provider_auth_failure",
            f"missing required provider environment variables: {missing}",
        )
    return {
        "required_secret_env": list(config.required_secret_env),
        "present_count": len(config.required_secret_env),
    }


def _gate_provider_smoke(context: PreflightContext) -> dict[str, object]:
    config = load_qualification_config(context.config_path)
    responses: list[dict[str, object]] = []
    option = PublicActionOption(
        option_id="option-preflight",
        files=(("preflight.txt", "ok\n"),),
    )
    for profile in config.policy_profiles:
        effective = _effective_profile(profile, context.environment)
        client = HttpPolicyClient(
            effective,
            api_key=context.environment[profile.api_key_env],
        )
        try:
            response = client.submit_action(
                PolicyRequest(
                    request_id=f"preflight-{profile.profile_id}",
                    system_prompt=(
                        "Select the only supplied option using the required "
                        "structured action tool."
                    ),
                    messages=(
                        PolicyMessage(
                            role="user",
                            content="Return the available preflight option.",
                        ),
                    ),
                    options=(option,),
                    max_output_tokens=64,
                )
            )
        finally:
            client.close()
        if response.selected_option_id != option.option_id:
            raise PreflightError(
                "provider_model_unavailable",
                f"{profile.profile_id} failed the structured option smoke",
            )
        responses.append(
            {
                "profile_id": profile.profile_id,
                "model_id": response.model_id,
                "request_hash": response.request_hash,
                "response_hash": response.response_hash,
            }
        )
    return {"responses": responses}


def _gate_mem0_runtime(context: PreflightContext) -> dict[str, object]:
    try:
        from importlib.metadata import version

        installed = version("mem0ai")
    except Exception as exc:
        raise PreflightError(
            "preflight_failure",
            f"cannot inspect installed Mem0: {exc}",
        ) from exc
    config = load_qualification_config(context.config_path)
    if installed != "2.0.12":
        raise PreflightError(
            "preflight_failure",
            f"installed Mem0 version is {installed}, expected 2.0.12",
        )
    return {
        "installed_version": installed,
        "controlled_profile": config.controlled_mem0.profile_id,
    }


def _gate_native_profile(context: PreflightContext) -> dict[str, object]:
    profile = load_qualification_config(context.config_path).native_mem0
    expected = (
        profile.track == "native"
        and profile.internal_llm_model == "gpt-5-mini"
        and profile.embedding_model == "text-embedding-3-small"
        and profile.vector_store == "qdrant"
    )
    if not expected:
        raise PreflightError(
            "preflight_failure",
            "native Mem0 profile is not the declared explicit configuration",
        )
    return {
        "profile_id": profile.profile_id,
        "internal_llm_model": profile.internal_llm_model,
        "embedding_model": profile.embedding_model,
    }


def _gate_trace_contract(context: PreflightContext) -> dict[str, object]:
    config = load_qualification_config(context.config_path)
    if config.controlled_mem0.prompt_source != "mem0_builtin":
        raise PreflightError(
            "preflight_failure",
            "Controlled Mem0 prompt source must remain mem0_builtin",
        )
    return {
        "required_trace_layers": [
            "write",
            "inventory",
            "candidate",
            "retrieved",
            "visible",
            "causal_use",
            "behavior",
        ],
        "prompt_source": config.controlled_mem0.prompt_source,
        "telemetry_enabled": config.controlled_mem0.telemetry_enabled,
    }


def _audit_recoverability(spec: SoftwareMem0VerticalSpec) -> None:
    variant = spec.plan.metadata_dict["recoverability_variant"]
    for latent, public in zip(
        spec.plan.workspaces,
        spec.plan.sessions,
        strict=True,
    ):
        if latent.recoverability["C1"] != variant:
            raise PreflightError(
                "surface_leak",
                "workspace recoverability label does not match episode variant",
            )
        text = "\n".join(
            artifact.content for artifact in public.workspace.artifacts
        ).casefold()
        if variant == "explicit" and not (
            "offline" in text and "cloud services" in text
        ):
            raise PreflightError(
                "surface_leak",
                "explicit workspace lacks the offline constraint",
            )
        if variant == "derivable" and "network_access = false" not in text:
            raise PreflightError(
                "surface_leak",
                "derivable workspace lacks configured evidence",
            )
        if variant == "absent" and any(
            phrase in text
            for phrase in ("offline", "do not call cloud", "network_access")
        ):
            raise PreflightError(
                "surface_leak",
                "absent workspace exposes the hidden constraint",
            )


def _verify_release(release_directory: Path) -> dict[str, object]:
    release = _read_json(release_directory / "RELEASE.json")
    archive = release_directory / str(release["dataset_archive"])
    expected = str(release["dataset_archive_sha256"])
    actual = _sha256(archive)
    if actual != expected:
        raise PreflightError(
            "preflight_failure",
            f"release archive hash mismatch: {archive.name}",
        )
    sidecar = archive.with_suffix(archive.suffix + ".sha256")
    sidecar_value = sidecar.read_text(encoding="utf-8").split()[0]
    if sidecar_value != expected:
        raise PreflightError(
            "preflight_failure",
            f"release sidecar hash mismatch: {sidecar.name}",
        )
    return {
        "release": release["release"],
        "archive": archive.name,
        "archive_sha256": actual,
    }


def _effective_profile(
    profile: PolicyProfile,
    environment: Mapping[str, str],
) -> PolicyProfile:
    endpoint_override_env = profile.endpoint_override_env
    endpoint = profile.endpoint
    if endpoint_override_env and environment.get(endpoint_override_env):
        endpoint = environment[endpoint_override_env]
    return replace(profile, endpoint=endpoint)


def _command_output(command: Sequence[str]) -> str:
    try:
        completed = subprocess.run(
            list(command),
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise PreflightError(
            "preflight_failure",
            f"command failed: {' '.join(command)}: {exc}",
        ) from exc
    return completed.stdout.strip()


def _git_output(root: Path, arguments: Sequence[str]) -> str:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise PreflightError(
            "preflight_failure",
            f"Git command failed: {' '.join(arguments)}: {exc}",
        ) from exc
    return completed.stdout.strip()


def _redact_text(text: str, environment: Mapping[str, str]) -> str:
    output = text
    for key, value in environment.items():
        if value and any(marker in key.casefold() for marker in _SECRET_MARKERS):
            output = output.replace(value, "<redacted>")
    return output


def _evaluator_continuation(value: object) -> EvaluatorContinuation:
    data = _mapping(value, "evaluator continuation")
    return EvaluatorContinuation(
        opportunity_id=str(data["opportunity_id"]),
        option_to_action=_pairs(
            data.get("option_to_action"),
            "option_to_action",
        ),
    )


def _pairs(value: object, label: str) -> tuple[tuple[str, str], ...]:
    output: list[tuple[str, str]] = []
    for item in _sequence(value, label):
        pair = _sequence(item, label)
        if len(pair) != 2:
            raise PreflightError(
                "preflight_failure",
                f"{label} must contain two-item pairs",
            )
        output.append((str(pair[0]), str(pair[1])))
    return tuple(output)


def _mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise PreflightError(
            "preflight_failure",
            f"{label} must be an object",
        )
    return {str(key): child for key, child in value.items()}


def _sequence(value: object, label: str) -> tuple[object, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise PreflightError(
            "preflight_failure",
            f"{label} must be an array",
        )
    return tuple(value)


def _read_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PreflightError(
            "preflight_failure",
            f"cannot read JSON {path}: {exc}",
        ) from exc
    return _mapping(value, str(path))


def _read_jsonl(path: Path) -> tuple[dict[str, object], ...]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise PreflightError(
            "preflight_failure",
            f"cannot read JSONL {path}: {exc}",
        ) from exc
    output: list[dict[str, object]] = []
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            output.append(_mapping(json.loads(line), f"{path}:{number}"))
        except json.JSONDecodeError as exc:
            raise PreflightError(
                "preflight_failure",
                f"invalid JSONL {path}:{number}: {exc}",
            ) from exc
    return tuple(output)


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=True,
            separators=(",", ":"),
            default=str,
        )
        + "\n"
    )
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        handle.write(payload)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise PreflightError(
            "preflight_failure",
            f"cannot hash {path}: {exc}",
        ) from exc
    return digest.hexdigest()


__all__ = [
    "PreflightCheck",
    "PreflightContext",
    "PreflightError",
    "PreflightGate",
    "PreflightReport",
    "RepositorySnapshot",
    "current_repository_snapshot",
    "default_preflight_gates",
    "load_mem0_specs",
    "redact_secrets",
    "require_live_gate",
    "run_preflight",
]

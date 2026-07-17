# Zen-Controlled Qualification Readiness Implementation Plan

> **Status (2026-07-17):** Superseded after Task 1 by the narrower
> Controlled-only server-readiness scope. The implemented path uses OpenCode
> Zen only for Claude Opus 4.8 and GPT-5.6 Sol, keeps DeepSeek V4 Pro direct,
> and defers Native-track and result-schema/report expansion. The checklist
> below is retained as implementation history, not as the current runbook.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the frozen Mem0 Software qualification runnable on the A100 server with Claude Opus 4.8 and GPT-5.6 Sol through OpenCode Zen, DeepSeek V4 Pro through the official DeepSeek API, and no standard OpenAI API dependency in the first Controlled-only run.

**Architecture:** Keep the existing four-condition qualification contract backward-compatible, but make the ordered condition matrix configurable and hash-addressed. Carry a non-secret route identity from model profiles through HTTP responses, task results, run manifests, and report artifacts. Resolve Native Mem0 OpenAI credentials only when `mem0_native` is selected, make preflight gates condition-aware, and route every server wrapper through one validated repository-relative experiment config. Preserve the frozen dataset, evaluator, Mem0 adapter semantics, and existing full-track configuration.

**Tech Stack:** Python 3.11, frozen dataclasses, PyYAML, urllib-based provider clients, pytest, mypy, ruff, Bash, Docker Compose, Slurm, Mem0 OSS 2.0.12, Qdrant, Hugging Face TEI, BGE-M3, BGE-reranker-v2-m3.

---

## Scope and invariants

The implementation must preserve these invariants throughout all tasks:

- `configs/experiments/mem0_controlled_zen.yaml` is the default server experiment and contains exactly `workspace_only`, `oracle_current_state`, and `mem0_controlled`.
- One episode under that config expands to 9 tasks and 12 scored result cells.
- `configs/experiments/mem0_qualification.yaml` remains the explicit full matrix: 12 tasks and 15 scored result cells per episode.
- A Controlled-only plan neither requires nor reads `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `MEM0_NATIVE_OPENAI_API_KEY`, or `MEM0_NATIVE_OPENAI_BASE_URL`.
- Native Mem0 uses `MEM0_NATIVE_OPENAI_*` first and legacy `OPENAI_*` only as a documented fallback when `mem0_native` is active.
- Secrets never enter hashes, manifests, reports, exception text, command output, or test snapshots.
- The frozen Software dataset and evaluator gold remain byte-identical.
- No offline test contacts Zen, DeepSeek, OpenAI, Anthropic, Mem0, Qdrant, or TEI over the network.
- The user-owned research drafts outside this isolated worktree are not touched.

## Task 1: Make the condition matrix configurable and add the Zen profiles

**Files:**

- Modify: `src/lhmsb/qualification/schema.py`
- Modify: `src/lhmsb/qualification/config.py`
- Modify: `configs/models/claude-opus-4-8.yaml`
- Modify: `configs/models/deepseek-v4-pro.yaml`
- Modify: `configs/models/gpt-5.6-sol.yaml`
- Create: `configs/models/claude-opus-4-8-zen.yaml`
- Create: `configs/models/gpt-5.6-sol-zen.yaml`
- Modify: `configs/experiments/mem0_qualification.yaml`
- Create: `configs/experiments/mem0_controlled_zen.yaml`
- Modify: `tests/qualification/test_config.py`

- [ ] **Step 1: Write failing tests for explicit conditions, compatibility, route IDs, and task counts**

Add repository constants and assertions to `tests/qualification/test_config.py`:

```python
CONTROLLED_ZEN_CONFIG = (
    ROOT / "configs" / "experiments" / "mem0_controlled_zen.yaml"
)


def test_controlled_zen_config_has_three_routes_and_three_conditions() -> None:
    config = load_qualification_config(CONTROLLED_ZEN_CONFIG)

    assert config.conditions == (
        "workspace_only",
        "oracle_current_state",
        "mem0_controlled",
    )
    assert [profile.route_id for profile in config.policy_profiles] == [
        "opencode_zen",
        "deepseek_direct",
        "opencode_zen",
    ]
    assert config.required_secret_env == (
        "OPENCODE_ZEN_API_KEY",
        "DEEPSEEK_API_KEY",
    )


def test_controlled_zen_matrix_has_nine_tasks_and_twelve_result_cells() -> None:
    config = load_qualification_config(CONTROLLED_ZEN_CONFIG)
    tasks = build_qualification_tasks(
        config,
        episode_ids=("software-mem0-42",),
        run_identity="run-hash",
    )

    assert len(tasks) == 9
    assert len(
        [result for task in tasks for result in task.scored_conditions]
    ) == 12
    assert {task.condition for task in tasks} == {
        "workspace_only",
        "oracle_current_state",
        "mem0_controlled",
    }


def test_repository_full_config_declares_all_conditions_explicitly() -> None:
    config = load_qualification_config(CONFIG)
    assert config.conditions == (
        "workspace_only",
        "oracle_current_state",
        "mem0_controlled",
        "mem0_native",
    )


def test_schema_v1_without_conditions_uses_legacy_full_matrix(
    tmp_path: Path,
) -> None:
    copied_configs = tmp_path / "repo" / "configs"
    shutil.copytree(ROOT / "configs", copied_configs)
    compatibility = copied_configs / "experiments" / "mem0_qualification.yaml"
    source = compatibility.read_text(encoding="utf-8")
    compatibility.write_text(
        source.replace(
            "conditions:\n"
            "  - workspace_only\n"
            "  - oracle_current_state\n"
            "  - mem0_controlled\n"
            "  - mem0_native\n",
            "",
        ),
        encoding="utf-8",
    )

    assert load_qualification_config(compatibility).conditions == (
        "workspace_only",
        "oracle_current_state",
        "mem0_controlled",
        "mem0_native",
    )
```

Also add parametrized rejection tests for an empty list, a duplicate condition,
and an unsupported value. Build each bad config by copying the repository config
into a temporary config tree, then assert `QualificationConfigError` includes the
field name `conditions`.

- [ ] **Step 2: Run the focused test and confirm it fails**

Run:

```bash
uv run pytest tests/qualification/test_config.py -q
```

Expected: failures because `QualificationConfig.conditions`,
`PolicyProfile.route_id`, and the Zen-owned YAML profiles do not exist.

- [ ] **Step 3: Extend the immutable schemas**

Add `route_id` to `PolicyProfile` immediately after `model_id`:

```python
@dataclass(frozen=True)
class PolicyProfile:
    profile_id: str
    provider: PolicyProvider
    model_id: str
    route_id: str
    api_key_env: str
    endpoint: str
    endpoint_override_env: str | None
    request_api: str
    timeout_seconds: float
    max_retries: int
    format_repair_attempts: int
```

In `config.py`, rename `_CONDITIONS` to `_DEFAULT_CONDITIONS`, add a supported
set, and store the resolved tuple on `QualificationConfig`:

```python
_DEFAULT_CONDITIONS: tuple[QualificationCondition, ...] = (
    "workspace_only",
    "oracle_current_state",
    "mem0_controlled",
    "mem0_native",
)
_SUPPORTED_CONDITIONS = frozenset(_DEFAULT_CONDITIONS)


@dataclass(frozen=True)
class QualificationConfig:
    schema_version: int
    experiment_id: str
    dataset_release: str
    data_root_env: str
    policy_profiles: tuple[PolicyProfile, ...]
    conditions: tuple[QualificationCondition, ...]
    retrieval: RetrievalProfile
    controlled_mem0: Mem0Profile
    native_mem0: Mem0Profile
    required_secret_env: tuple[str, ...]
```

Validate that conditions are non-empty, unique, and supported. Do not sort
them: their declared order determines task indices and is part of the config
hash. Add `conditions` to `to_dict()`, parse it with the old full tuple as the
schema-v1 default, and change `build_qualification_tasks()` to iterate
`config.conditions`. Validate that `required_secret_env` equals the ordered
unique `api_key_env` sequence derived from the configured policy profiles. This
makes a profile/config credential mismatch a configuration error while allowing
two Zen profiles to share one key.

Load `route_id` as a required non-empty field in `_load_policy()`. Update every
direct `PolicyProfile(...)` construction under `tests/` to pass an explicit
route ID; do not introduce a dataclass default that could hide missing route
metadata.

- [ ] **Step 4: Add explicit direct and Zen route profiles**

Set these exact route contracts:

| Profile file | provider | model ID | route ID | API key env | endpoint | request API |
|---|---|---|---|---|---|---|
| `claude-opus-4-8.yaml` | `anthropic` | `claude-opus-4-8` | `anthropic_direct` | `ANTHROPIC_API_KEY` | existing Anthropic endpoint | `messages` |
| `deepseek-v4-pro.yaml` | `deepseek` | `deepseek-v4-pro` | `deepseek_direct` | `DEEPSEEK_API_KEY` | existing DeepSeek endpoint | existing chat-completions value |
| `gpt-5.6-sol.yaml` | `openai` | `gpt-5.6-sol` | `openai_direct` | `OPENAI_API_KEY` | existing OpenAI endpoint | `responses` |
| `claude-opus-4-8-zen.yaml` | `anthropic` | `claude-opus-4-8` | `opencode_zen` | `OPENCODE_ZEN_API_KEY` | `https://opencode.ai/zen` | `messages` |
| `gpt-5.6-sol-zen.yaml` | `openai` | `gpt-5.6-sol` | `opencode_zen` | `OPENCODE_ZEN_API_KEY` | `https://opencode.ai/zen` | `responses` |

Both Zen profiles use `OPENCODE_ZEN_BASE_URL` as the endpoint override. Retain
the existing timeout, retry, and format-repair values from the corresponding
direct profile.

Add all four conditions explicitly to `mem0_qualification.yaml`. Create
`mem0_controlled_zen.yaml` with the same frozen dataset, retrieval profile, and
Mem0 profile references, but use the two Zen policy profiles plus direct
DeepSeek, declare only the three Controlled conditions, and list only:

```yaml
required_secret_env:
  - OPENCODE_ZEN_API_KEY
  - DEEPSEEK_API_KEY
```

- [ ] **Step 5: Run config tests and static checks**

Run:

```bash
uv run pytest tests/qualification/test_config.py -q
uv run mypy src/lhmsb/qualification/schema.py src/lhmsb/qualification/config.py
uv run ruff check src/lhmsb/qualification/schema.py src/lhmsb/qualification/config.py tests/qualification/test_config.py
```

Expected: pass, with 9/12 for the Controlled-Zen matrix and unchanged 12/15 for
the full matrix.

- [ ] **Step 6: Commit**

```bash
git add src/lhmsb/qualification/schema.py src/lhmsb/qualification/config.py configs/models configs/experiments tests/qualification/test_config.py
git commit -m "feat: add controlled Zen qualification matrix"
```

## Task 2: Carry route identity through policy calls

**Files:**

- Modify: `src/lhmsb/qualification/schema.py`
- Modify: `src/lhmsb/qualification/providers.py`
- Modify: `src/lhmsb/qualification/runner.py`
- Modify: `tests/qualification/test_providers.py`
- Modify: `tests/qualification/test_runner.py`
- Modify: `tests/qualification/test_mem0_vertical_slice.py`

- [ ] **Step 1: Write failing endpoint/protocol and serialization tests**

Add a test for each Zen protocol to `tests/qualification/test_providers.py`.
Reuse the existing `_profile()`, `_request()`, `httpx.MockTransport`, and
`dataclasses.replace`; add this concrete response helper and assert the exact
URL, authorization headers, body protocol, and returned route:

```python
def _zen_success(provider: str, model_id: str) -> dict[str, object]:
    action = {
        "action_id": "option-01",
        "optional_patch": None,
        "concise_rationale": "Selected.",
    }
    if provider == "anthropic":
        return {
            "id": "msg_zen",
            "model": model_id,
            "content": [
                {
                    "type": "tool_use",
                    "name": "submit_action",
                    "input": action,
                }
            ],
            "usage": {"input_tokens": 10, "output_tokens": 3},
        }
    return {
        "id": "resp_zen",
        "model": model_id,
        "output": [
            {
                "type": "function_call",
                "name": "submit_action",
                "arguments": json.dumps(action),
            }
        ],
        "usage": {"input_tokens": 10, "output_tokens": 3},
    }


@pytest.mark.parametrize(
    ("provider", "expected_url", "request_api"),
    (
        (
            "anthropic",
            "https://opencode.ai/zen/v1/messages",
            "messages",
        ),
        (
            "openai",
            "https://opencode.ai/zen/v1/responses",
            "responses",
        ),
    ),
)
def test_zen_profiles_preserve_provider_protocol_and_route(
    provider: str,
    expected_url: str,
    request_api: str,
) -> None:
    profile = replace(
        _profile(provider),
        route_id="opencode_zen",
        api_key_env="OPENCODE_ZEN_API_KEY",
        endpoint="https://opencode.ai/zen",
        endpoint_override_env="OPENCODE_ZEN_BASE_URL",
    )
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200,
            json=_zen_success(provider, profile.model_id),
        )

    response = HttpPolicyClient(
        profile,
        api_key="fake-zen-secret",
        transport=httpx.MockTransport(handler),
    ).submit_action(_request())

    assert str(seen[0].url) == expected_url
    assert profile.request_api == request_api
    assert response.route_id == "opencode_zen"
    assert response.request_api == request_api
    assert response.requested_model_id == profile.model_id
    assert response.model_id == profile.model_id
    assert response.endpoint_identity == "https://opencode.ai/zen"
    if provider == "anthropic":
        assert seen[0].headers["x-api-key"] == "fake-zen-secret"
        assert "authorization" not in seen[0].headers
    else:
        assert seen[0].headers["authorization"] == "Bearer fake-zen-secret"
```

Extend the existing `PolicyResponse` round-trip/runner serialization tests to
assert:

```python
assert restored.response.route_id == original.response.route_id
assert restored.response.request_api == original.response.request_api
assert restored.response.requested_model_id == original.response.requested_model_id
assert restored.response.endpoint_identity == original.response.endpoint_identity
```

Add a model-mismatch test for Zen: if the returned model is not the requested
model, the client must fail with the existing provider response error before a
task result is accepted.

- [ ] **Step 2: Run the tests and confirm the missing field failures**

Run:

```bash
uv run pytest tests/qualification/test_providers.py tests/qualification/test_runner.py tests/qualification/test_mem0_vertical_slice.py -q
```

Expected: failures because `PolicyResponse` does not expose route, request
protocol, or requested-model identity.

- [ ] **Step 3: Add route identity to every policy response**

Add `route_id: str`, `request_api: str`, and `requested_model_id: str` to the
immutable `PolicyResponse` dataclass in `providers.py`. In `HttpPolicyClient`,
set route and protocol from the effective profile; do not infer them from the
provider name or hostname. Change `_validate_model()` to return the validated
provider-returned model string, then construct the response with
`requested_model_id=self.profile.model_id` and `model_id=returned_model_id`.
Keep `endpoint_identity` as the normalized effective base endpoint selected
after the environment override is applied.

Update the runner's response serialization/deserialization with the exact key:

```python
"route_id": response.route_id,
"request_api": response.request_api,
"requested_model_id": response.requested_model_id,
```

and:

```python
route_id=_text(data.get("route_id"), "response.route_id"),
request_api=_text(data.get("request_api"), "response.request_api"),
requested_model_id=_text(
    data.get("requested_model_id"),
    "response.requested_model_id",
),
```

Update all fake `PolicyResponse(...)` constructors in tests to use explicit
values such as `test_route`, `anthropic_direct`, or `opencode_zen`.

- [ ] **Step 4: Verify endpoint normalization and protocol separation**

The path builder must produce exactly one `/v1` component for both a base URL
with no trailing slash and one with a trailing slash. Anthropic Messages uses
`x-api-key` and `anthropic-version`; OpenAI Responses and DeepSeek chat use
Bearer authorization. Add assertions against the fake request headers so a
gateway route cannot silently switch logical protocols.

- [ ] **Step 5: Run focused tests and static checks**

Run:

```bash
uv run pytest tests/qualification/test_providers.py tests/qualification/test_runner.py tests/qualification/test_mem0_vertical_slice.py -q
uv run mypy src/lhmsb/qualification/providers.py src/lhmsb/qualification/runner.py
uv run ruff check src/lhmsb/qualification/providers.py src/lhmsb/qualification/runner.py tests/qualification/test_providers.py tests/qualification/test_runner.py tests/qualification/test_mem0_vertical_slice.py
```

Expected: pass without network access.

- [ ] **Step 6: Commit**

```bash
git add src/lhmsb/qualification/schema.py src/lhmsb/qualification/providers.py src/lhmsb/qualification/runner.py tests/qualification/test_providers.py tests/qualification/test_runner.py tests/qualification/test_mem0_vertical_slice.py
git commit -m "feat: trace policy route identity"
```

## Task 3: Isolate Native OpenAI credentials and seal them into run identity safely

**Files:**

- Create: `src/lhmsb/qualification/credentials.py`
- Modify: `src/lhmsb/qualification/cli.py`
- Modify: `src/lhmsb/qualification/__init__.py`
- Create: `tests/qualification/test_credentials.py`
- Modify: `tests/qualification/test_cli.py`

- [ ] **Step 1: Write failing credential precedence and non-consumption tests**

Create `tests/qualification/test_credentials.py` with these cases:

```python
def test_controlled_only_never_reads_native_or_legacy_openai_credentials() -> None:
    config = load_qualification_config(CONTROLLED_ZEN_CONFIG)
    environment = RaisingEnvironment(
        forbidden={
            "MEM0_NATIVE_OPENAI_API_KEY",
            "MEM0_NATIVE_OPENAI_BASE_URL",
            "OPENAI_API_KEY",
            "OPENAI_BASE_URL",
        },
        values={
            "OPENCODE_ZEN_API_KEY": "zen-secret",
            "DEEPSEEK_API_KEY": "deepseek-secret",
        },
    )

    assert (
        resolve_native_openai_credentials(
            config,
            environment,
            required=False,
        )
        is None
    )
    assert effective_required_secret_env(config, environment) == (
        "OPENCODE_ZEN_API_KEY",
        "DEEPSEEK_API_KEY",
    )


def test_native_prefers_dedicated_openai_credentials() -> None:
    config = load_qualification_config(FULL_CONFIG)
    credentials = resolve_native_openai_credentials(
        config,
        {
            "MEM0_NATIVE_OPENAI_API_KEY": "native-secret",
            "MEM0_NATIVE_OPENAI_BASE_URL": "https://native.example/v1",
            "OPENAI_API_KEY": "legacy-secret",
            "OPENAI_BASE_URL": "https://legacy.example/v1",
        },
        required=True,
    )

    assert credentials is not None
    assert credentials.api_key == "native-secret"
    assert credentials.api_key_env == "MEM0_NATIVE_OPENAI_API_KEY"
    assert credentials.base_url == "https://native.example/v1"
    assert credentials.base_url_env == "MEM0_NATIVE_OPENAI_BASE_URL"


def test_native_falls_back_to_legacy_openai_credentials() -> None:
    config = load_qualification_config(FULL_CONFIG)
    credentials = resolve_native_openai_credentials(
        config,
        {
            "OPENAI_API_KEY": "legacy-secret",
            "OPENAI_BASE_URL": "https://legacy.example/v1",
        },
        required=True,
    )

    assert credentials is not None
    assert credentials.api_key_env == "OPENAI_API_KEY"
    assert credentials.base_url_env == "OPENAI_BASE_URL"


def test_native_without_openai_credentials_fails_before_execution() -> None:
    config = load_qualification_config(FULL_CONFIG)
    with pytest.raises(
        QualificationCredentialError,
        match="MEM0 Native requires MEM0_NATIVE_OPENAI_API_KEY or OPENAI_API_KEY",
    ):
        resolve_native_openai_credentials(config, {}, required=True)
```

`RaisingEnvironment` is a small `Mapping[str, str]` test double whose `get()`
raises when a forbidden key is accessed. Implement it directly in the test:

```python
class RaisingEnvironment(Mapping[str, str]):
    def __init__(
        self,
        *,
        forbidden: set[str],
        values: Mapping[str, str],
    ) -> None:
        self._forbidden = forbidden
        self._values = dict(values)

    def __getitem__(self, key: str) -> str:
        if key in self._forbidden:
            raise AssertionError(f"forbidden credential read: {key}")
        return self._values[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def get(self, key: str, default: str | None = None) -> str | None:
        if key in self._forbidden:
            raise AssertionError(f"forbidden credential read: {key}")
        return self._values.get(key, default)
```

This proves non-consumption rather than merely proving that a missing value is
tolerated.

In `test_cli.py`, add assertions that planning the Controlled-Zen config with
only its two policy secrets succeeds, that its manifest contains no OpenAI or
Anthropic direct secret names, and that changing a route ID or effective
endpoint changes `run_identity`.

- [ ] **Step 2: Run tests and confirm the missing module/behavior**

Run:

```bash
uv run pytest tests/qualification/test_credentials.py tests/qualification/test_cli.py -q
```

Expected: import failure for `lhmsb.qualification.credentials`, followed by
manifest assertion failures once the module is introduced.

- [ ] **Step 3: Implement the credential resolver with non-secret identity**

Create these frozen dataclasses:

```python
@dataclass(frozen=True)
class NativeOpenAICredentials:
    api_key: str
    api_key_env: str
    base_url: str
    base_url_env: str | None


@dataclass(frozen=True)
class NativeOpenAIIdentity:
    enabled: bool
    route_id: str
    api_key_env: str | None
    base_url: str | None
    base_url_env: str | None
    llm_model: str | None
    embedding_model: str | None
```

Implement:

```python
def resolve_native_openai_credentials(
    config: QualificationConfig,
    environment: Mapping[str, str],
    *,
    required: bool,
) -> NativeOpenAICredentials | None:
```

Return immediately when `mem0_native` is absent, before accessing any OpenAI
key. When active, prefer the dedicated pair, fall back to the legacy pair, use
`https://api.openai.com` when no base URL is set, strip surrounding whitespace,
and reject an empty key when `required=True`. When the Native condition is
active but no key exists and `required=False`, return `None`; this preserves
offline planning and repository-only checks.

Implement `native_openai_identity(config, environment)` independently of secret
material. It exposes only the selected environment variable names, endpoint,
route ID `openai_native_mem0`, and the pinned native LLM/embedding model IDs. If
no Native key is present during offline planning, select the dedicated
`MEM0_NATIVE_OPENAI_*` names as the declared future route. If a legacy OpenAI
key is present, record the fallback route. Implement
`effective_required_secret_env()` as the ordered unique policy credential list
plus the selected Native key environment when active. Neither function may
return credential values.

- [ ] **Step 4: Version and extend the run manifest**

Bump `QUALIFICATION_RUN_SCHEMA_VERSION` from 3 to 4. Add:

```python
policy_routes: tuple[PolicyRouteIdentity, ...]
native_openai: NativeOpenAIIdentity
```

Define `PolicyRouteIdentity` in `schema.py` with:

```python
@dataclass(frozen=True)
class PolicyRouteIdentity:
    profile_id: str
    provider: PolicyProvider
    model_id: str
    route_id: str
    request_api: str
    endpoint_identity: str
```

Build policy route identities after applying endpoint overrides. Include the
ordered policy route list, Native identity, and effective required-secret names
in the run-identity payload. Serialize/deserialise the two new fields in
`QualificationRunManifest.to_dict()` and `.from_dict()`. The manifest contains
only environment variable names and endpoint identities, never secret values.

On resume, `_load_run_contract()` must compare these fields as part of the
immutable manifest. A configuration, condition, route, or effective endpoint
change must stop with `QualificationCliError` rather than reusing old results.

- [ ] **Step 5: Use the resolver only for Native task construction**

In `_build_components()`, branch on `task.condition` before resolving Native
credentials:

```python
native_credentials = (
    resolve_native_openai_credentials(
        config,
        environment,
        required=True,
    )
    if task.condition == "mem0_native"
    else None
)
native_api_key = native_credentials.api_key if native_credentials else ""
native_base_url = native_credentials.base_url if native_credentials else ""
```

Pass those values to the existing Mem0 live-config builder. Do not read
`OPENAI_*` in common task setup. Ensure dry-run planning can build a manifest
without materializing secret values.

- [ ] **Step 6: Run focused checks**

Run:

```bash
uv run pytest tests/qualification/test_credentials.py tests/qualification/test_cli.py -q
uv run mypy src/lhmsb/qualification/credentials.py src/lhmsb/qualification/cli.py
uv run ruff check src/lhmsb/qualification/credentials.py src/lhmsb/qualification/cli.py tests/qualification/test_credentials.py tests/qualification/test_cli.py
```

Expected: pass; a Controlled-only manifest remains free of Native credential
names and secret values.

- [ ] **Step 7: Commit**

```bash
git add src/lhmsb/qualification/credentials.py src/lhmsb/qualification/cli.py src/lhmsb/qualification/schema.py src/lhmsb/qualification/__init__.py tests/qualification/test_credentials.py tests/qualification/test_cli.py
git commit -m "feat: isolate native OpenAI qualification credentials"
```

## Task 4: Make live preflight condition-aware and prove both Mem0 lifecycles

**Files:**

- Modify: `src/lhmsb/qualification/preflight.py`
- Modify: `src/lhmsb/qualification/cli.py`
- Modify: `tests/qualification/test_preflight.py`
- Modify: `tests/qualification/test_cli.py`

- [ ] **Step 1: Write failing applicability tests**

Extend `test_preflight.py` so the gate contract distinguishes inactive from
repository-only skips:

```python
def test_controlled_config_marks_native_gates_inactive(tmp_path: Path) -> None:
    calls: list[str] = []

    def controlled(_: PreflightContext) -> dict[str, object]:
        calls.append("controlled")
        return {}

    def native(_: PreflightContext) -> dict[str, object]:
        raise AssertionError("inactive Native gate ran")

    context = PreflightContext(
        repository_root=ROOT,
        dataset_root=tmp_path / "dataset",
        config_path=CONTROLLED_ZEN_CONFIG,
        data_root=tmp_path / "data",
        allow_dirty=True,
        repository_only=False,
        environment={},
    )
    report = run_preflight(
        context,
        gates=(
            PreflightGate(
                "controlled",
                "live",
                controlled,
                required_condition="mem0_controlled",
            ),
            PreflightGate(
                "native",
                "live",
                native,
                required_condition="mem0_native",
            ),
        ),
    )

    assert report.ok is True
    assert [(check.name, check.status, check.applicable) for check in report.checks] == [
        ("controlled", "pass", True),
        ("native", "skip", False),
    ]
    assert calls == ["controlled"]


def test_native_zen_config_requires_native_credentials_before_lifecycle(
    tmp_path: Path,
) -> None:
    copied_configs = tmp_path / "repo" / "configs"
    shutil.copytree(ROOT / "configs", copied_configs)
    config_path = copied_configs / "experiments" / "mem0_controlled_zen.yaml"
    config_path.write_text(
        config_path.read_text(encoding="utf-8").replace(
            "  - mem0_controlled\n",
            "  - mem0_controlled\n  - mem0_native\n",
        ),
        encoding="utf-8",
    )
    context = PreflightContext(
        repository_root=ROOT,
        dataset_root=tmp_path / "dataset",
        config_path=config_path,
        data_root=tmp_path / "data",
        allow_dirty=True,
        repository_only=False,
        environment={
            "OPENCODE_ZEN_API_KEY": "fake-zen-secret",
            "DEEPSEEK_API_KEY": "fake-deepseek-secret",
        },
    )

    with pytest.raises(PreflightError, match="MEM0_NATIVE_OPENAI_API_KEY"):
        _gate_provider_credentials(context)
```

Add a default-gate ordering test asserting:

```python
names = [gate.name for gate in default_preflight_gates()]
assert names.index("native_mem0_profile") < names.index("native_mem0_lifecycle")
assert names.index("native_mem0_lifecycle") < names.index("trace_and_prompt_contract")
```

Patch the Mem0/Qdrant/provider collaborators and test that the Controlled
lifecycle probes all three configured policy profiles without reading Native
OpenAI keys. Add the parallel Native lifecycle test with exactly one native
profile, one write/history/search cycle, and cleanup of the isolated collection.

- [ ] **Step 2: Run tests and confirm failures**

Run:

```bash
uv run pytest tests/qualification/test_preflight.py tests/qualification/test_cli.py -q
```

Expected: failures because gates do not carry conditions, checks do not expose
applicability, and no Native lifecycle gate exists.

- [ ] **Step 3: Extend gate and check schemas**

Change the dataclasses to:

```python
@dataclass(frozen=True)
class PreflightGate:
    name: str
    scope: PreflightScope
    probe: PreflightProbe
    required_condition: QualificationCondition | None = None


@dataclass(frozen=True)
class PreflightCheck:
    name: str
    scope: PreflightScope
    status: PreflightStatus
    applicable: bool = True
    error_class: str | None = None
    message: str | None = None
    details: Mapping[str, object] | None = None
```

Materialize the selected gate tuple once. Load the config once, and only when at
least one selected gate has a non-null `required_condition`; this preserves the
existing isolated unit tests that pass custom condition-free gates with no
config file. For a gate whose `required_condition` is absent from the loaded
config, append a skip with `applicable=False`, message
`inactive qualification condition`, and the condition in redacted details.
For repository-only skipping of an otherwise active live gate, retain
`applicable=True`. Only an applicable failure can set `stopped_at`.

- [ ] **Step 4: Annotate Mem0-specific gates and credential checks**

Set `required_condition="mem0_controlled"` on
`controlled_mem0_lifecycle`. Set `required_condition="mem0_native"` on
`native_mem0_profile` and the new `native_mem0_lifecycle`. Provider gates remain
applicable for all configs because they always verify the configured policy
profiles; `_gate_provider_credentials()` additionally invokes the Native
resolver only when the Native condition is active.

The provider structured smoke details must include `route_id`, effective
endpoint, requested model, and returned model for each policy. Keep secret
redaction active over both success details and exception text.

- [ ] **Step 5: Implement the real Native lifecycle gate**

Mirror the Controlled lifecycle's isolated resource creation and cleanup, but
use `config.native_mem0` and resolved Native OpenAI credentials. Exercise:

1. adapter construction with the pinned `gpt-5-mini` and
   `text-embedding-3-small` profile;
2. one public write transcript;
3. history/inventory retrieval;
4. one search and trace capture;
5. explicit Qdrant collection and local-history cleanup in `finally`.

Return non-secret details containing the native profile ID, model IDs, event
count, live-memory count, candidate count, and endpoint identity. Do not return
prompt bodies, memory text, credentials, or provider response bodies.

- [ ] **Step 6: Tighten live runtime acceptance**

Update CLI `_runtime_identity()` so a live run accepts:

- `pass` for every applicable gate;
- `skip` only when `applicable=False` because the condition is inactive.

It must reject repository-only reports, applicable live skips, any failed gate,
and a report missing a default applicable gate. Pass the planned config hash
into `_runtime_identity()` and require it to equal
`repository_and_config.details.config_hash` from the preflight report. Include
the config hash plus each default gate's `(name, status, applicable)` tuple in
the hardware/runtime identity hash alongside the GPU allocation. This prevents
a Controlled preflight from being reused for the full track, or a preflight for
one config from authorizing another.

Add CLI tests that replace the latest preflight report's config hash or flip an
inactive Native check to `applicable=True`; both must fail before run planning
or task execution.

- [ ] **Step 7: Run tests and checks**

Run:

```bash
uv run pytest tests/qualification/test_preflight.py tests/qualification/test_cli.py -q
uv run mypy src/lhmsb/qualification/preflight.py src/lhmsb/qualification/cli.py
uv run ruff check src/lhmsb/qualification/preflight.py src/lhmsb/qualification/cli.py tests/qualification/test_preflight.py tests/qualification/test_cli.py
```

Expected: pass with no live network access.

- [ ] **Step 8: Commit**

```bash
git add src/lhmsb/qualification/preflight.py src/lhmsb/qualification/cli.py tests/qualification/test_preflight.py tests/qualification/test_cli.py
git commit -m "feat: make qualification preflight condition aware"
```

## Task 5: Emit route-auditable, Controlled-only-valid reports and metrics

**Files:**

- Modify: `src/lhmsb/qualification/report.py`
- Modify: `src/lhmsb/qualification/metrics.py`
- Modify: `src/lhmsb/qualification/validate.py`
- Modify: `src/lhmsb/qualification/cli.py`
- Modify: `tests/qualification/test_report.py`
- Modify: `tests/qualification/test_metrics.py`
- Modify: `tests/qualification/test_validate.py`
- Modify: `tests/qualification/test_cli.py`

- [ ] **Step 1: Write failing report and metric tests**

Add a Controlled-only matrix fixture and assert:

```python
def test_controlled_only_report_is_complete_without_native_cells(
    controlled_matrix: QualificationMatrixResult,
    specs: dict[str, SoftwareMem0VerticalSpec],
    tmp_path: Path,
) -> None:
    artifacts = write_qualification_report(
        controlled_matrix,
        specs,
        tmp_path / "report",
        run_metadata=controlled_run_metadata(),
    )
    validation = validate_qualification_artifacts(artifacts.root)

    assert validation.ok is True
    groups = json.loads(
        (artifacts.root / "metrics_by_cell.json").read_text(encoding="utf-8")
    )["groups"]
    assert {group["condition"] for group in groups} == {
        "workspace_only",
        "oracle_current_state",
        "mem0_controlled",
    }
    metrics = json.loads(
        (artifacts.root / "metrics.json").read_text(encoding="utf-8")
    )
    assert metrics["mem0_native_gain_beyond_workspace"] == {
        "numerator": 0.0,
        "denominator": 0.0,
        "value": None,
    }
```

Assert `policy_calls.jsonl` exists, contains exactly the API usage rows with a
non-empty `policy_request_hash`, and every row includes:

```python
{
    "provider": "anthropic",
    "request_api": "messages",
    "requested_model_id": "claude-opus-4-8",
    "model_id": "claude-opus-4-8",
    "route_id": "opencode_zen",
    "endpoint_identity": "https://opencode.ai/zen",
}
```

Add validator corruption tests for a missing route ID, missing endpoint,
unknown policy call ID, a non-empty but wrong route/protocol, and a
`policy_calls.jsonl` row that does not match its corresponding
`api_usage.jsonl` row.

- [ ] **Step 2: Run tests and confirm failures**

Run:

```bash
uv run pytest tests/qualification/test_report.py tests/qualification/test_metrics.py tests/qualification/test_validate.py tests/qualification/test_cli.py -q
```

Expected: report artifact/route assertions fail and the report schema version
is still 2.

- [ ] **Step 3: Version the report and add policy call rows**

Bump `REPORT_SCHEMA_VERSION` from 2 to 3. Add `policy_calls.jsonl` to both
`REQUIRED_REPORT_ARTIFACTS` and `_JSONL_ARTIFACTS`.

Add `route_id`, `request_api`, and `requested_model_id` to
`_append_api_usage()` from the policy response. Internal Mem0 usage rows use
`route_id="mem0_internal"`, `request_api=usage.component`, and
`requested_model_id=usage.model_id`; local reranker rows use
`route_id="local_tei"`, `request_api="rerank"`, and their local model as the
requested model. This keeps `api_usage.jsonl` column-consistent without
mislabeling internal calls as policy calls.

After flattening all task results, derive policy calls by copying only rows
whose `policy_request_hash` is a non-empty string:

```python
rows["policy_calls.jsonl"] = [
    dict(row)
    for row in rows["api_usage.jsonl"]
    if isinstance(row.get("policy_request_hash"), str)
    and bool(row["policy_request_hash"])
]
```

Let the existing deterministic JSONL writer sort the copied rows. Compute
`summary.json:n_policy_calls` from `len(rows["policy_calls.jsonl"])`, not from a
negative list of internal call kinds.

- [ ] **Step 4: Preserve denominator-safe not-applicable metrics**

Keep `safe_ratio(0, 0)` as the only representation for a comparison cell that
is absent. Add regression coverage around `_behavior_metrics()` so Native gain
and Native oracle-gap keys are always emitted but have denominator zero and
`value=None` for Controlled-only matrices. Do not synthesize Native scorecard
or `metrics_by_cell` groups.

- [ ] **Step 5: Validate route and policy-call lineage**

Teach `validate.py` to read `policy_calls.jsonl`. Require unique call IDs,
non-empty `provider`, `request_api`, `requested_model_id`, `model_id`,
`route_id`, `endpoint_identity`, `request_hash`, `response_hash`, and
`policy_request_hash`. Require requested and returned model IDs to match. For
every policy call, require an exactly matching `api_usage.jsonl` row on all
those fields. Resolve `policy_profile_id` against the report manifest's
`policy_routes` and require provider, protocol, requested model, route ID, and
endpoint identity to match the planned route. Reject unknown or duplicate
manifest profile IDs.
Derive expected scorecard/metric cells from task results as today; no fixed
Native cell list is allowed.

Pass the manifest's `policy_routes`, Native identity, effective configuration
path, and condition list into report `run_metadata` from aggregation. Make sure
the report manifest remains redacted and hash-addresses every new artifact.

- [ ] **Step 6: Run focused checks**

Run:

```bash
uv run pytest tests/qualification/test_report.py tests/qualification/test_metrics.py tests/qualification/test_validate.py tests/qualification/test_cli.py -q
uv run mypy src/lhmsb/qualification/report.py src/lhmsb/qualification/metrics.py src/lhmsb/qualification/validate.py
uv run ruff check src/lhmsb/qualification/report.py src/lhmsb/qualification/metrics.py src/lhmsb/qualification/validate.py tests/qualification/test_report.py tests/qualification/test_metrics.py tests/qualification/test_validate.py
```

Expected: pass; the validator treats the 3-condition report as complete and
does not invent Native result cells.

- [ ] **Step 7: Commit**

```bash
git add src/lhmsb/qualification/report.py src/lhmsb/qualification/metrics.py src/lhmsb/qualification/validate.py src/lhmsb/qualification/cli.py tests/qualification/test_report.py tests/qualification/test_metrics.py tests/qualification/test_validate.py tests/qualification/test_cli.py
git commit -m "feat: emit route-auditable controlled reports"
```

## Task 6: Route all server entry points through one validated config

**Files:**

- Modify: `scripts/lib/mem0_common.sh`
- Modify: `scripts/bootstrap_server.sh`
- Modify: `scripts/preflight_mem0.sh`
- Modify: `scripts/run_mem0_smoke.sh`
- Modify: `scripts/run_mem0_qualification.sh`
- Modify: `deploy/slurm/mem0_preflight.sbatch`
- Modify: `deploy/slurm/mem0_qualification.sbatch`
- Modify: `deploy/compose.mem0.yaml`
- Modify: `.env.example`
- Modify: `tests/qualification/test_scripts.py`
- Modify: `tests/qualification/test_deploy_assets.py`

- [ ] **Step 1: Write failing path-validation and dry-run tests**

Add a parametrized test covering every Bash wrapper. Invoke each with
`--dry-run --config configs/experiments/mem0_controlled_zen.yaml` and assert its
printed Python/Compose command contains the same container path:

```text
/app/configs/experiments/mem0_controlled_zen.yaml
```

Run the wrappers from a copied repository path containing spaces. Add rejection
cases for:

```text
/tmp/outside.yaml
../outside.yaml
configs/experiments/../models/deepseek-v4-pro.yaml
configs/models/deepseek-v4-pro.yaml
configs/experiments/missing.yaml
```

Each must exit non-zero before Docker or Python execution and print a message
that names the invalid config path without printing any environment values.
Also create a symlink under `configs/experiments/` pointing outside that tree
and assert it is rejected.

Assert both Slurm scripts consume `LHMSB_EXPERIMENT_CONFIG`, and Compose passes
the Zen, DeepSeek, and Native variable names into the worker without embedding
values.

- [ ] **Step 2: Run tests and confirm hard-coded-path failures**

Run:

```bash
uv run pytest tests/qualification/test_scripts.py tests/qualification/test_deploy_assets.py -q
```

Expected: failures because wrappers and Slurm currently hard-code
`mem0_qualification.yaml`.

- [ ] **Step 3: Implement one shared repository-relative path validator**

Add to `scripts/lib/mem0_common.sh`:

```bash
mem0_validate_experiment_config() {
  local repo_root="$1"
  local relative_path="$2"

  case "${relative_path}" in
    /*|*"/../"*|../*|*"/./"*|./*)
      printf 'invalid experiment config path: %s\n' "${relative_path}" >&2
      return 2
      ;;
    configs/experiments/*.yaml)
      ;;
    *)
      printf 'experiment config must be under configs/experiments/: %s\n' "${relative_path}" >&2
      return 2
      ;;
  esac

  if [[ ! -f "${repo_root}/${relative_path}" ]]; then
    printf 'experiment config does not exist: %s\n' "${relative_path}" >&2
    return 2
  fi
  if [[ -L "${repo_root}/${relative_path}" ]]; then
    printf 'experiment config must not be a symbolic link: %s\n' "${relative_path}" >&2
    return 2
  fi

  local allowed_root
  local resolved_parent
  allowed_root="$(cd "${repo_root}/configs/experiments" && pwd -P)" || return
  resolved_parent="$(cd "$(dirname "${repo_root}/${relative_path}")" && pwd -P)" || return
  case "${resolved_parent}/" in
    "${allowed_root}/"*)
      ;;
    *)
      printf 'experiment config resolves outside configs/experiments/: %s\n' "${relative_path}" >&2
      return 2
      ;;
  esac

  printf '%s\n' "${relative_path}"
}


mem0_container_config_path() {
  local repo_root="$1"
  local relative_path
  relative_path="$(mem0_validate_experiment_config "$repo_root" "$2")" || return
  printf '/app/%s\n' "${relative_path}"
}
```

Use shell pattern checks before `realpath` so the function behaves consistently
on macOS and Linux. Quoting must preserve repository paths containing spaces.

- [ ] **Step 4: Add the same config option to every wrapper**

Use this default in all scripts:

```bash
CONFIG_RELATIVE="${LHMSB_EXPERIMENT_CONFIG:-configs/experiments/mem0_controlled_zen.yaml}"
```

Parse `--config PATH`, validate it once, compute both:

```bash
CONFIG_HOST="${REPO_ROOT}/${CONFIG_RELATIVE}"
CONFIG_CONTAINER="/app/${CONFIG_RELATIVE}"
```

Use `CONFIG_HOST` for host-side bootstrap/config checks and
`CONFIG_CONTAINER` for Compose worker commands. Export
`LHMSB_EXPERIMENT_CONFIG=${CONFIG_RELATIVE}` when one script calls another.
Update usage text and dry-run output. No wrapper may retain a literal
`/app/configs/experiments/mem0_qualification.yaml`.

- [ ] **Step 5: Update Compose, Slurm, and the environment template**

Compose must pass only variable names/value substitutions:

```yaml
OPENCODE_ZEN_API_KEY: "${OPENCODE_ZEN_API_KEY:-}"
OPENCODE_ZEN_BASE_URL: "${OPENCODE_ZEN_BASE_URL:-https://opencode.ai/zen}"
DEEPSEEK_API_KEY: "${DEEPSEEK_API_KEY:-}"
DEEPSEEK_BASE_URL: "${DEEPSEEK_BASE_URL:-https://api.deepseek.com}"
MEM0_NATIVE_OPENAI_API_KEY: "${MEM0_NATIVE_OPENAI_API_KEY:-}"
MEM0_NATIVE_OPENAI_BASE_URL: "${MEM0_NATIVE_OPENAI_BASE_URL:-https://api.openai.com}"
```

Keep direct Anthropic/OpenAI variables for the optional full config. Set both
Slurm scripts' config from the same environment default and validate/map it
through the shared helper before invoking the worker.

In `.env.example`, put the runnable Controlled section first:

```dotenv
LHMSB_EXPERIMENT_CONFIG=configs/experiments/mem0_controlled_zen.yaml
OPENCODE_ZEN_API_KEY=
OPENCODE_ZEN_BASE_URL=https://opencode.ai/zen
DEEPSEEK_API_KEY=
DEEPSEEK_BASE_URL=https://api.deepseek.com
```

Put Native OpenAI and direct-provider credentials under an explicitly optional
full-track section. State in comments that a ChatGPT subscription is not an
OpenAI API credential.

- [ ] **Step 6: Run script and syntax checks**

Run:

```bash
uv run pytest tests/qualification/test_scripts.py tests/qualification/test_deploy_assets.py -q
bash -n scripts/lib/mem0_common.sh scripts/bootstrap_server.sh scripts/preflight_mem0.sh scripts/run_mem0_smoke.sh scripts/run_mem0_qualification.sh deploy/slurm/mem0_preflight.sbatch deploy/slurm/mem0_qualification.sbatch
uv run ruff check tests/qualification/test_scripts.py tests/qualification/test_deploy_assets.py
```

Expected: pass, including the repository path containing spaces.

- [ ] **Step 7: Commit**

```bash
git add scripts deploy .env.example tests/qualification/test_scripts.py tests/qualification/test_deploy_assets.py
git commit -m "ops: parameterize Mem0 experiment configuration"
```

## Task 7: Document the exact two-A100 Controlled workflow and Native activation boundary

**Files:**

- Modify: `README.md`
- Modify: `docs/mem0-server-workflow.md`
- Modify: `tests/qualification/test_config.py`
- Modify: `tests/qualification/test_deploy_assets.py`

- [ ] **Step 1: Write failing documentation contract tests**

Require the README and server workflow to include:

- the exact default config `configs/experiments/mem0_controlled_zen.yaml`;
- three logical policy models and their three route labels;
- two local A100 assignments for embedding and reranking;
- `workspace_only`, `oracle_current_state`, and `mem0_controlled` as the first
  execution matrix;
- the four-session smoke before the 16-session pilot;
- `policy_calls.jsonl`, `metrics.json`, `metrics_by_cell.json`, and
  `scorecard.csv` as expected outputs;
- the requirement for standard OpenAI API billing before enabling
  `mem0_native`;
- commands for switching to the full config without editing scripts.

Also assert that the docs do not claim Zen supplies Native embeddings and do
not describe a ChatGPT membership as an API credential.

- [ ] **Step 2: Run the documentation tests and confirm failures**

Run:

```bash
uv run pytest tests/qualification/test_config.py tests/qualification/test_deploy_assets.py -q
```

Expected: missing Controlled-Zen workflow text.

- [ ] **Step 3: Write the server runbook**

Document this operational sequence with exact commands:

```bash
cp .env.example .env
${EDITOR:-vi} .env
./scripts/bootstrap_server.sh --config configs/experiments/mem0_controlled_zen.yaml
./scripts/preflight_mem0.sh --config configs/experiments/mem0_controlled_zen.yaml
./scripts/run_mem0_smoke.sh --config configs/experiments/mem0_controlled_zen.yaml
./scripts/run_mem0_qualification.sh --config configs/experiments/mem0_controlled_zen.yaml
```

Describe the two-A100 allocation already encoded by Compose: one TEI embedding
service on GPU 0 and one TEI reranker service on GPU 1. Policy models remain API
calls, so adding GPUs does not change the policy route.

List the report directory and explain the measurement chain:

```text
state/write records
→ retrieval_trace.jsonl
→ policy_calls.jsonl
→ task_results.jsonl
→ metrics.json / metrics_by_cell.json / scorecard.csv
```

Explain that the Controlled report is final for the Controlled track even
though Native metrics are null with zero denominators. Native is enabled later
by setting `MEM0_NATIVE_OPENAI_API_KEY`, optionally setting its base URL, and
passing `--config configs/experiments/mem0_qualification.yaml` consistently to
all stages.

- [ ] **Step 4: Run docs tests and spelling-sensitive searches**

Run:

```bash
uv run pytest tests/qualification/test_config.py tests/qualification/test_deploy_assets.py -q
rg -n "mem0_controlled_zen|opencode_zen|deepseek_direct|policy_calls.jsonl|MEM0_NATIVE_OPENAI_API_KEY" README.md docs/mem0-server-workflow.md
```

Expected: pass and each operational contract is discoverable.

- [ ] **Step 5: Commit**

```bash
git add README.md docs/mem0-server-workflow.md tests/qualification/test_config.py tests/qualification/test_deploy_assets.py
git commit -m "docs: add controlled Zen server workflow"
```

## Task 8: Run the integration gate and prepare the server handoff

**Files:**

- Modify only if a verified defect is found: files already listed in Tasks 1–7
- Do not modify: frozen dataset archives, evaluator gold, or user-owned drafts

- [ ] **Step 1: Run the focused qualification suite**

Run:

```bash
uv run pytest tests/qualification tests/contract/test_mem0_qualification.py tests/longhorizon/test_attribution.py -q
```

Expected: pass.

- [ ] **Step 2: Run static analysis**

Run:

```bash
uv run ruff check src/lhmsb/qualification src/lhmsb/adapters/mem0_qualification.py tests/qualification
uv run mypy src/lhmsb/qualification src/lhmsb/adapters/mem0_qualification.py
bash -n scripts/lib/mem0_common.sh scripts/bootstrap_server.sh scripts/preflight_mem0.sh scripts/run_mem0_smoke.sh scripts/run_mem0_qualification.sh deploy/slurm/mem0_preflight.sbatch deploy/slurm/mem0_qualification.sbatch
uv lock --check
git diff --check
```

Expected: all commands exit 0.

- [ ] **Step 3: Verify frozen releases without mutating them**

Verify both tracked archives, extract v0.2 into a temporary directory, then run
the existing verification and regeneration checks against that extracted
frozen dataset. Do not regenerate either tracked archive in place:

```bash
(
  cd datasets/releases/software-vertical-v0.1.0
  shasum -a 256 -c software_v1-6b4edbf.tar.gz.sha256
)
(
  cd datasets/releases/software-vertical-mem0-v0.2.0
  shasum -a 256 -c software_mem0_v2.tar.gz.sha256
)
TMP_ROOT="$(mktemp -d)"
trap 'rm -rf "${TMP_ROOT}"' EXIT
tar -xzf \
  datasets/releases/software-vertical-mem0-v0.2.0/software_mem0_v2.tar.gz \
  -C "${TMP_ROOT}"
FROZEN_ROOT="${TMP_ROOT}/software_mem0_v2"
uv run python -m lhmsb.datasets verify-mem0-stateful \
  --frozen "${FROZEN_ROOT}"
uv run python -m lhmsb.datasets regen-check-mem0-stateful \
  --frozen "${FROZEN_ROOT}"
uv run python -m lhmsb.qualification preflight \
  --dataset "${FROZEN_ROOT}" \
  --config configs/experiments/mem0_controlled_zen.yaml \
  --data-root runs/preflight-controlled \
  --repository-only \
  --allow-dirty
```

Expected: both checksums, frozen verification, and deterministic regeneration
pass; all repository preflight gates pass and active live gates are skipped as
applicable repository-only checks. Inactive Native gates remain explicitly
`applicable=false`.

- [ ] **Step 4: Exercise every wrapper in dry-run mode**

Run from the repository root:

```bash
./scripts/bootstrap_server.sh --dry-run --config configs/experiments/mem0_controlled_zen.yaml
./scripts/preflight_mem0.sh --dry-run --config configs/experiments/mem0_controlled_zen.yaml
./scripts/run_mem0_smoke.sh --dry-run --config configs/experiments/mem0_controlled_zen.yaml
./scripts/run_mem0_qualification.sh --dry-run --config configs/experiments/mem0_controlled_zen.yaml
```

Expected: every printed stage uses the same config path, no secret values are
printed, and no Docker container or paid provider call starts.

- [ ] **Step 5: Run the supported full local regression suite**

Run:

```bash
uv run pytest -q -k 'not test_resource_module_is_linux'
```

Expected: all supported macOS tests pass. The excluded Linux resource-module
test is exercised later inside the server image.

- [ ] **Step 6: Audit the diff and secret surface**

Run:

```bash
git status --short
git diff --stat main...HEAD
git diff --check main...HEAD
rg -n "(sk-|Bearer [A-Za-z0-9]|api[_-]?key:[[:space:]]*[^$])" . \
  --glob '!uv.lock' \
  --glob '!docs/superpowers/plans/2026-07-17-zen-controlled-readiness.md'
```

Manually inspect every match. Expected: only environment variable names,
fixtures explicitly named as fake secrets, and documentation examples; no real
credential value or frozen-data modification.

- [ ] **Step 7: Commit any verification-only correction, then request review**

If verification required a code correction, rerun the narrow failing test first,
then the complete integration gate, and commit only that correction:

Inspect `git diff --name-only`, stage each correction path explicitly, and
commit it with:

```bash
git commit -m "fix: close controlled qualification verification gap"
```

Use `superpowers:requesting-code-review`, address only evidence-backed findings,
and rerun Step 1 through Step 6 after review changes.

- [ ] **Step 8: Prepare the execution handoff**

Record in the final handoff:

- branch name and commit SHA;
- frozen dataset release ID and checksum;
- Controlled config hash and planned task/result counts;
- local test, lint, type-check, shell-check, and preflight evidence;
- that Docker/GPU/live-provider checks remain intentionally unexecuted locally;
- the first four server commands from Task 7;
- the report paths and validation command;
- the exact Native activation prerequisites.

Do not merge or push until the user explicitly asks after reviewing the
implementation evidence.

# Zen-Controlled Qualification Readiness Design

## Status

Approved direction from the 2026-07-17 credential and server-readiness
discussion. This document fixes the implementation boundary before code changes.

## Goal

Make the frozen Mem0 qualification runnable with the credentials currently
available:

- Claude Opus 4.8 through OpenCode Zen;
- GPT-5.6 Sol through OpenCode Zen;
- DeepSeek V4 Pro through the official DeepSeek API;
- BGE-M3 and BGE-reranker-v2-m3 on two local A100 GPUs;
- Qdrant as the isolated vector store.

The first executable experiment is the Controlled track. The Native track stays
defined but inactive until a standard OpenAI API key is available for its
`gpt-5-mini` and `text-embedding-3-small` dependencies.

## Alternatives considered

### A. Run the existing full matrix unchanged

This requires a standard OpenAI API key and API billing. A ChatGPT subscription
does not satisfy that contract. Reusing a Zen key as `OPENAI_API_KEY` would also
send the Native Mem0 embedding request to a gateway that does not document the
required embedding model. This approach is rejected for the current pilot.

### B. Controlled-only with mixed, explicit routes (selected)

Serve Opus and GPT through Zen, keep DeepSeek on its official API, and run only
`workspace_only`, `oracle_current_state`, and `mem0_controlled`. This uses the
available credentials without changing the benchmark's local retrieval stack.
Every result records its route so the gateway is not conflated with a direct
provider call.

### C. Route all three policies through Zen

This gives one credential and gateway, but makes every policy result depend on
the gateway. It remains useful as a later DeepSeek route-sensitivity check, not
as the first primary run while an official DeepSeek key is already available.

## Configuration contract

`QualificationConfig` gains an ordered `conditions` tuple. The resolved tuple is
included in canonical serialization and therefore in the configuration and run
identity hashes. Supported conditions remain:

1. `workspace_only`
2. `oracle_current_state`
3. `mem0_controlled`
4. `mem0_native`

The existing full configuration declares all four explicitly. A new
`mem0_controlled_zen.yaml` declares the first three. Task construction uses the
configuration tuple instead of a module-level hard-coded matrix.

Existing schema-version-1 configurations without `conditions` retain the old
four-condition default for compatibility, but all repository-owned experiment
configs declare the field explicitly.

Each policy profile gains a non-secret `route_id`. Direct profiles use values
such as `deepseek_direct`; Zen profiles use `opencode_zen`. The route ID and
effective endpoint appear in policy-call artifacts and participate in the
effective-profile run identity. Route IDs do not alter the request protocol:

- Zen Opus continues to use Anthropic Messages format;
- Zen GPT continues to use OpenAI Responses format;
- direct DeepSeek continues to use Chat Completions format.

## Credential contract

The Controlled-Zen configuration requires only:

- `OPENCODE_ZEN_API_KEY`;
- `DEEPSEEK_API_KEY`.

It accepts `OPENCODE_ZEN_BASE_URL`, defaulting to
`https://opencode.ai/zen`, and `DEEPSEEK_BASE_URL`, defaulting to
`https://api.deepseek.com`.

Native OpenAI credentials are separated from policy credentials:

- `MEM0_NATIVE_OPENAI_API_KEY`;
- `MEM0_NATIVE_OPENAI_BASE_URL`.

For backward compatibility, the runtime may fall back to `OPENAI_API_KEY` and
`OPENAI_BASE_URL` only when a run actually contains `mem0_native`. A
Controlled-only run must neither require nor consume either Native credential.
Credentials remain environment-only and are never serialized into a run or
report artifact.

## Preflight and execution flow

Server scripts accept one repository-relative experiment configuration path and
use the same path for bootstrap, live preflight, smoke, qualification, Slurm,
and resume. Paths must remain under `configs/experiments/`; absolute paths and
parent traversal are rejected.

Preflight behavior is condition-aware:

- provider credential and structured-output probes cover every configured
  policy profile;
- Controlled Mem0 lifecycle probes run only when `mem0_controlled` is enabled;
- Native-specific credential and lifecycle probes run only when `mem0_native`
  is enabled;
- inactive-track gates return an auditable inactive result rather than causing
  a false failure;
- a live qualification still requires every applicable gate to pass.

The first server sequence is:

1. build assets and images;
2. run repository-only preflight;
3. run live Controlled-Zen preflight;
4. run a frozen four-session smoke;
5. validate all artifacts;
6. run the frozen 16-session Controlled pilot;
7. freeze the raw run and report before any route-sensitivity experiment.

## Output and metric behavior

The report remains schema-valid when `mem0_native` is absent. Native-specific
gain and oracle-gap metrics are emitted as not-applicable values with zero
denominators; they are never silently computed from Controlled cells.

`policy_calls.jsonl`, task results, and the run manifest expose:

- logical provider protocol;
- exact requested and returned model ID;
- `route_id`;
- effective endpoint identity;
- request and response hashes;
- observed usage and latency.

`metrics_by_cell.json` contains only cells planned by the selected conditions.
The validator derives expected coverage from the task table, so a
Controlled-only report is complete when every planned Controlled cell exists.

## Error handling

The run stops before paid execution when:

- a configured secret is missing;
- Zen or DeepSeek returns a different model ID;
- a route uses the wrong request protocol;
- the chosen configuration differs between plan and resume;
- a Native condition is requested without Native OpenAI credentials;
- a script configuration path escapes `configs/experiments/`.

Independent tasks continue under the existing `--keep-going` contract after a
runtime task failure, but aggregation remains incomplete until every planned
task has a terminal artifact.

## Test strategy

Tests are offline and follow red-green TDD. They cover:

- explicit and backward-compatible condition parsing;
- nine tasks and twelve result cells for one Controlled-only episode;
- unchanged twelve-task/fifteen-cell full matrix;
- Zen profile endpoint, route, secret, and request-protocol resolution;
- run-identity changes when route or conditions change;
- Native credentials are unused for Controlled-only runs;
- Native credentials are mandatory when Native is selected;
- condition-aware preflight gate applicability;
- script config-path mapping and traversal rejection;
- provider route serialization and report validation;
- dry-run commands using the same config at every stage.

No test calls Zen, DeepSeek, OpenAI, Mem0, Qdrant, or TEI over the network.

## Non-goals

- Running the real server experiment from this workstation;
- using ChatGPT or OpenCode OAuth tokens as benchmark credentials;
- changing the frozen Software episode or evaluator gold;
- relabeling a local-embedding configuration as Native Mem0;
- adding another memory system;
- performing the DeepSeek route-sensitivity experiment in this change.

## Acceptance criteria

The readiness change is complete when:

1. the Controlled-Zen config plans exactly the intended three policies and
   three conditions;
2. no OpenAI or Anthropic direct key is needed for that plan;
3. all repository and Controlled live gates can be exercised independently of
   Native credentials;
4. route identity is present in immutable run and result artifacts;
5. all wrapper dry-runs preserve the selected config;
6. focused tests, the supported full local test suite, Ruff, mypy, shell syntax,
   frozen-data verification, and repository-only preflight pass;
7. user-owned research drafts remain untouched.

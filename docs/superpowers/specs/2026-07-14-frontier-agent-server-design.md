# Frontier Agent Server Design

## Goal

Prepare the Wide Research benchmark for a frontier agent API that will be selected when server access exists. The repository must not depend on DiffRWKV, and a formal run must not start with a placeholder model or missing credentials.

## Scope

- Keep one provider contract: OpenAI-compatible `POST /chat/completions`.
- Remove the StateDiffRWKV provider, configuration fields, tests, and worker special cases.
- Run formal experiments on a server. Local development covers code and documentation only.
- Preserve the completed gold-free Wide trace as data construction, not as an agent experiment.

This change does not choose a model vendor, add vendor SDKs, or run a local agent evaluation.

## Runtime Configuration

`configs/wide_research.yaml` will reference three server environment variables:

- `LHMSB_AGENT_MODEL`: exact frontier model ID used for the run.
- `LHMSB_AGENT_BASE_URL`: OpenAI-compatible API base URL ending at the version root.
- `LHMSB_AGENT_API_KEY`: API credential; the manifest records only the variable name.

The YAML loader resolves exact `${VARIABLE}` references for non-secret fields. It includes the resolved model ID and base URL in the config hash and run manifest. The provider reads the secret at runtime and never writes it to disk.

## Model Freeze Rule

The project owner selects the strongest frontier reasoning model available to the server at the experiment freeze date. The model must support the benchmark context window, temperature-zero generation, and a stable API model identifier. Before the full run, the team records the exact model ID and any provider revision exposed by the API. Every memory condition in one comparison uses that same frozen model configuration.

Moving aliases such as `latest` are not valid for a formal run. A provider without a stable model identifier cannot be used for the formal benchmark.

## Fail-Fast Behavior

Before creating datasets, manifests, or result directories, the pilot validates:

1. `agent_provider` equals `openai_compatible` for a live run.
2. The model and base URL environment references resolve to non-empty values.
3. The configured API-key environment variable exists.
4. The model ID is not a placeholder or moving `latest` alias.

The command exits with a configuration error and names the missing variable. Smoke tests remain offline and do not require these variables.

## Server Flow

1. Transfer the repository and frozen data artifacts to the server.
2. Verify question, index, and trace hashes before attaching evaluator gold.
3. Export the three agent environment variables.
4. Run unit and provider contract tests on the server.
5. Run a small qualified-data integration slice.
6. Freeze the model configuration, then run `no_mem`, `mem`, and `wrong_mem` on identical episodes.

The run manifest stores the effective model configuration before the matrix starts. The server keeps raw responses and scorecard artifacts under the run directory.

## Test Contract

Server tests cover environment resolution, missing-variable failures, secret redaction, OpenAI-compatible request and response parsing, manifest provenance, and the three-condition matrix. No test or formal result may use StateDiffRWKV. Existing StateDiffRWKV pilot artifacts are historical local outputs and remain excluded from the formal run.

## Acceptance Criteria

- Source and test code contain no StateDiffRWKV provider path.
- `wide_research.yaml` contains no machine-local path or model name.
- A missing frontier API configuration fails before writing run artifacts.
- A configured OpenAI-compatible endpoint receives the pinned model ID and generation parameters.
- The manifest records effective non-secret configuration and never records the API key.
- Documentation gives one server-side command sequence for tests and the formal run.

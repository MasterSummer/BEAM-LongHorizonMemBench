# Real-repository experiment runner

The `lhmsb.realworld` package runs paired, multi-session code-editing experiments
from clean Git snapshots. Each task-condition-repeat cell gets a separate working
copy. The repository is materialized from a pinned commit, the same workspace is
kept between sessions, and hidden/regression commands run only after the agent
sessions finish. Results include session transcripts, test output, a patch, and a
JSON ledger.

## Server quick start (framework checks only)

From the repository root, with Python 3.11 or newer:

```bash
git pull --ff-only origin main
python3 -m venv .venv-realworld
source .venv-realworld/bin/activate
python -m pip install -e '.[dev]'
bash scripts/check_realworld.sh
```

The check uses a temporary toy Git repository and a deterministic editing executor,
not an LLM or SWE-agent. It does not call a model API or produce research results.
The code below describes the runner interface; it is not a ready-to-run experiment.
Formal runs still need a real pinned repository task, an external hidden grader,
and a validated SWE-agent session adapter. Memory integration, file access controls,
and failure checkpoint repair remain to be implemented and verified.

Do not discard server-side edits to make `git pull` succeed. If it refuses to
update, inspect `git status` and preserve those edits before merging.

## Define a task

Create an experiment YAML. Repository paths are relative to the YAML file unless
they are absolute. Pin `revision` to a commit hash for a real run.

```yaml
experiment_id: cross_session_api_change
repeats: 3
timeout_seconds: 1800
conditions:
  - condition_id: no_memory_files_visible
    memory_mode: none
    workspace_files_visible: true
  - condition_id: memory_files_visible
    memory_mode: explicit
    workspace_files_visible: true
tasks:
  - task_id: issue_001
    repository: ../repos/example
    revision: 0123456789abcdef0123456789abcdef01234567
    setup_commands:
      - python -m pip install -e '.[test]'
    sessions:
      - session_id: initial_request
        prompt: |-
          Implement the requested API behavior. Inspect the repository and run relevant tests.
      - session_id: changed_requirement
        prompt: |-
          A new requirement takes effect for this API only. Update the implementation
          and tests while preserving unrelated behavior.
    hidden_test_commands:
      - python -m pytest -q /grader/issue_001_hidden.py
    regression_test_commands:
      - python -m pytest -q tests
```

The hidden grader should live outside the agent-visible repository and be mounted
or copied by the experiment operator. Do not expose hidden-test source or expected
outputs to the agent. For safety and reproducibility, run untrusted repositories
and agent commands in an isolated container; the local runner itself does not
provide OS-level sandboxing.

## Agent command interface

Run the CLI with an agent command:

```bash
uv run python -m lhmsb.realworld.cli \
  --experiment configs/realworld/example.yaml \
  --out runs/realworld/example-001 \
  --agent-command 'python /opt/agent/session_adapter.py'
```

The command is invoked once per session with its working directory set to the
task workspace. It receives:

- `LHMSB_WORKSPACE`: persistent checkout for this task-condition-repeat;
- `LHMSB_PROMPT_FILE`: UTF-8 file containing the current session request;
- `LHMSB_SESSION_ID`: current session identifier;
- `LHMSB_CONDITION_JSON`: condition id, memory mode, file visibility, and settings;
- `LHMSB_TRANSCRIPT_PATH`: path where the runner records process output.

The adapter should construct a fresh agent session for each invocation, preserve
only the explicitly configured memory channel between sessions, operate on
`LHMSB_WORKSPACE`, and return exit status zero only when that session completed.
The runner saves stdout/stderr and exit status in the transcript JSON. It does not
yet implement SWE-agent memory hooks or hide workspace files inside SWE-agent;
those controls must be implemented by the adapter before interpreting conditions
as a valid memory/file ablation.

## Output and scoring

Each unit directory contains `result.json`, session prompt/transcript files,
stdout/stderr for setup and graders, and `changes.patch`. The root contains
`results.jsonl`, one row per completed unit. `task_success` requires successful
setup, all configured agent sessions, and every hidden test command. Regression
tests are reported separately; the current `task_success` field does not include
them. For formal research scoring, require both `task_success` and
`regression_pass`, as well as the task-specific build and scope checks. Do not use
`task_success` alone to claim the task passed without regressions.

Output directories are append-protected: rerunning into a directory with an
existing unit or result ledger fails rather than overwriting evidence. Use a new
directory for a new run. Failed agent sessions remain recorded and are followed
by the graders when possible, so failure artifacts are preserved.

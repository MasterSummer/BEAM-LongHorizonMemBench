# Three-System Controlled Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the frozen Software long-horizon dataset directly testable on an A100 server across Mem0, A-MEM, MemOS-Tree, Workspace-only, Full-context, Oracle-current-state, and Flat retrieval, with three policy models and complete programmatic metrics.

**Architecture:** Preserve the schema-v1 Mem0 qualification path while adding a schema-v2 multisystem path. Prefix preparation is backend-specific and runs once per episode for Flat, Mem0, A-MEM, and MemOS-Tree. It emits immutable hash-addressed artifacts. Policy evaluation is backend-neutral and reuses those artifacts for all three policies without mutating memory. Dedicated images isolate incompatible upstream dependencies; a core worker plans, evaluates, aggregates, and validates.

**Tech Stack:** Python 3.11, frozen dataclasses, PyYAML, HTTPX, Qdrant, Neo4j 5, Hugging Face TEI, BGE-M3, BGE-Reranker-v2-M3, DeepSeek Chat Completions, OpenCode Zen Messages/Responses, pytest, mypy, ruff, Docker Compose, Bash, and Slurm.

---

## Invariants

- Keep the frozen Software v0.2 dataset and evaluator gold byte-identical.
- Keep the current schema-v1 Mem0 config, CLI, imports, scripts, and results readable.
- The new matrix has 4 prefix tasks, 21 policy tasks, and 30 scored cells per episode.
- Mem0, A-MEM, and MemOS-Tree use one fixed DeepSeek writer prefix per episode, shared by every tested policy.
- Policy evaluation never writes, updates, or deletes memory.
- Full-context includes every and only public observations/tool results before the checkpoint and never silently truncates.
- Flat retrieval stores unchanged public trajectory units and performs no LLM or lifecycle operation.
- Memory-object count, not token count, is the primary scale variable.
- Managed-system headline scores use native readout; common rerank is separately labeled.
- No adapter silently falls back to Flat retrieval or another memory backend.
- Offline tests use fake upstream packages and mocked HTTP; live API, Docker, GPU, Neo4j, Qdrant, and upstream acceptance run only on the server.
- Secrets never enter hashes, manifests, traces, reports, exception messages, or test snapshots.

## Task 1: Add backend-neutral conditions and canonical public history

**Files**

- Create: src/lhmsb/qualification/conditions.py
- Create: src/lhmsb/qualification/context.py
- Modify: src/lhmsb/qualification/schema.py
- Modify: src/lhmsb/qualification/config.py
- Test: tests/qualification/test_conditions.py
- Test: tests/qualification/test_context.py
- Modify: tests/qualification/test_config.py

- [ ] Write failing tests for the seven ordered condition definitions. Assert the exact kinds, required prefix backend, readouts, intervention support, and visible-k behavior. Assert managed systems expose native and common_rerank, Flat exposes common_rerank only, and controls expose none.
- [ ] Write failing tests that build canonical PublicHistoryUnit objects from SoftwareMem0VerticalSpec.write_transcript. Verify deterministic IDs, source session/kind/ordinal, exact unchanged content, and exclusion of evaluator IDs, validity labels, checker output, future sessions, and prior policy responses.
- [ ] Test Full-context at every SCEU: history sessions equal range(checkpoint_session), order is stable, current surface is not duplicated, and any rendered history above full_context_max_chars raises FullContextLimitError rather than truncating.
- [ ] Implement frozen ConditionDefinition and a registry lookup that rejects unknown conditions; remove startswith-based semantics from new code but retain schema-v1 aliases.
- [ ] Implement PublicHistoryUnit, build_public_history_units, render_full_context, and full_context_hash from the same canonical source later consumed by Flat retrieval.
- [ ] Extend QualificationCondition to the seven new names without changing old names. Add explicit condition definitions to config serialization so declaration order remains part of config_hash.
- [ ] Run:

    uv run pytest tests/qualification/test_conditions.py tests/qualification/test_context.py tests/qualification/test_config.py -q
    uv run mypy src/lhmsb/qualification/conditions.py src/lhmsb/qualification/context.py
    uv run ruff check src/lhmsb/qualification/conditions.py src/lhmsb/qualification/context.py tests/qualification/test_conditions.py tests/qualification/test_context.py

- [ ] Commit with message: feat: add canonical multisystem conditions and context

## Task 2: Extract the generic memory trace contract

**Files**

- Create: src/lhmsb/qualification/memory_runtime.py
- Modify: src/lhmsb/adapters/mem0_qualification.py
- Modify: src/lhmsb/qualification/runner.py
- Modify: src/lhmsb/qualification/__init__.py
- Test: tests/qualification/test_memory_runtime.py
- Modify: tests/qualification/test_runner.py
- Create: tests/qualification/test_mem0_adapter.py

- [ ] Write failing round-trip and validation tests for MemoryMutationEvent, MemoryObject, InventorySnapshot, RetrievalCandidate, CandidateSearch, ProviderUsageEvent, WriteSessionResult, and StorageFootprint.
- [ ] Require normalized event kind plus native event name, backend ID, content hash, provenance, and optional graph metadata. Require retrieval candidates to retain native rank/score and candidate origin.
- [ ] Define MemoryRuntime.write_session, snapshot_inventory, search_candidates, restore_write_count, storage_footprints, and close protocols. Define lifecycle capabilities explicitly rather than by backend name.
- [ ] Move the existing Mem0 DTO implementation behind this contract. Preserve NativeMemoryEvent, InventoryItem, SearchCandidate, and MemoryRuntime import compatibility through aliases/re-exports.
- [ ] Update the existing runner to consume generic types without changing serialized schema-v1 result bytes.
- [ ] Test that malformed inventories, duplicate memory IDs, inconsistent n_live, invalid content hashes, and retrieved IDs outside the inventory fail with typed errors.
- [ ] Run:

    uv run pytest tests/qualification/test_memory_runtime.py tests/qualification/test_mem0_adapter.py tests/qualification/test_runner.py -q
    uv run mypy src/lhmsb/qualification/memory_runtime.py src/lhmsb/adapters/mem0_qualification.py
    uv run ruff check src/lhmsb/qualification/memory_runtime.py src/lhmsb/adapters/mem0_qualification.py tests/qualification/test_memory_runtime.py

- [ ] Commit with message: refactor: generalize qualification memory traces

## Task 3: Add schema-v2 system profiles and two-stage task identities

**Files**

- Modify: src/lhmsb/qualification/schema.py
- Modify: src/lhmsb/qualification/config.py
- Create: src/lhmsb/qualification/prefix.py
- Create: configs/experiments/systems_controlled_zen.yaml
- Create: configs/models/deepseek-v4-pro-writer.yaml
- Create: configs/systems/flat/controlled.yaml
- Create: configs/systems/amem/controlled.yaml
- Create: configs/systems/memos/tree-controlled.yaml
- Modify: configs/systems/mem0/controlled.yaml
- Modify: configs/systems.lock.yaml
- Test: tests/qualification/test_multisystem_config.py
- Test: tests/qualification/test_prefix.py

- [ ] Write failing tests that load schema-v2 and assert exact condition order, three policies, fixed DeepSeek writer, common BGE profiles, source pins, candidate_k=20, visible_k=5, and full_context_max_chars=100000.
- [ ] Assert repository pins: Mem0 2.0.12, A-MEM commit ceffb860f0712bbae97b184d440df62bc910ca8d, and MemOS v2.0.23 commit 583b07b998afc4debb6c5078439b0b3896f5b097.
- [ ] Add frozen FlatRetrievalProfile, AMemProfile, MemOSTreeProfile, and a discriminated SystemProfile union. Reject A-MEM package identity a-mem, MemOS modes other than tree, fallback declarations, different embedding models, writer/profile mismatches, or managed profiles without both readouts.
- [ ] Add PreparationTask, EvaluationTaskTemplate, and EvaluationTask. Initial planning produces four preparations plus stable non-executable templates. Only finalize_evaluation_plan, after verified artifacts exist, produces exactly 7 conditions × 3 policies and 30 unique ScoredCondition IDs with managed native/common branches. Assert stable indices across finalization and reject execution of templates.
- [ ] Add immutable MemoryPrefixCheckpoint and MemoryPrefixArtifact schemas. Include source/profile/config/model identities, dataset/surface hashes, writes, inventories, retrievals, common reranks, graph diagnostics, storage footprints, and canonical artifact_hash. Round-trip must reproduce byte-identical canonical JSON.
- [ ] Add a frozen causal-use/sampling profile with temperature=0, max_output_tokens=512, baseline_repeats=2, intervention_repeats=2, format-repair policy, and explicit null provider seed. Include it in config and run identity.
- [ ] Ensure evaluation task identity contains the required prefix artifact hash while controls use a canonical no-prefix marker. Changing any prefix hash must change only dependent evaluation task hashes.
- [ ] Preserve schema-v1 parser behavior and build_qualification_tasks output for old configs.
- [ ] Run:

    uv run pytest tests/qualification/test_multisystem_config.py tests/qualification/test_prefix.py tests/qualification/test_config.py -q
    uv run mypy src/lhmsb/qualification/schema.py src/lhmsb/qualification/config.py src/lhmsb/qualification/prefix.py
    uv run ruff check src/lhmsb/qualification/schema.py src/lhmsb/qualification/config.py src/lhmsb/qualification/prefix.py tests/qualification/test_multisystem_config.py tests/qualification/test_prefix.py

- [ ] Commit with message: feat: add two-stage multisystem experiment schema

## Task 4: Implement deterministic Flat retrieval

**Files**

- Create: src/lhmsb/adapters/flat_retrieval.py
- Create: src/lhmsb/qualification/qdrant.py
- Test: tests/adapters/test_flat_retrieval.py
- Test: tests/qualification/test_qdrant.py

- [ ] Write failing tests that ingest each PublicHistoryUnit exactly once with deterministic hash ID, unchanged text, session/kind/ordinal provenance, and BGE-M3 vector.
- [ ] Use a fake EmbeddingRuntime and fake Qdrant transport to prove that ingestion never calls a policy/writer and emits OBSERVED_ADD only; duplicate calls are idempotent and conflicting content for an existing ID is terminal.
- [ ] Implement the minimal Qdrant HTTP boundary needed for create-empty-collection, count, upsert, scroll inventory, search, delete namespace, and health. Validate point IDs, vector dimension, scores, and namespace isolation.
- [ ] Implement FlatRetrievalAdapter as MemoryRuntime. Store one object per canonical public unit, expose full inventory, search top candidate_k in Qdrant, then rely on the benchmark common reranker for visible_k.
- [ ] Test candidate shortfall, deterministic tie order, query/content hashes, object-count accounting, and zero writer/provider usage.
- [ ] Run:

    uv run pytest tests/adapters/test_flat_retrieval.py tests/qualification/test_qdrant.py -q
    uv run mypy src/lhmsb/adapters/flat_retrieval.py src/lhmsb/qualification/qdrant.py
    uv run ruff check src/lhmsb/adapters/flat_retrieval.py src/lhmsb/qualification/qdrant.py tests/adapters/test_flat_retrieval.py tests/qualification/test_qdrant.py

- [ ] Commit with message: feat: add deterministic flat retrieval baseline

## Task 5: Implement immutable prefix preparation

**Files**

- Create: src/lhmsb/qualification/prepare.py
- Modify: src/lhmsb/qualification/storage.py
- Modify: src/lhmsb/longhorizon/attribution.py
- Test: tests/qualification/test_prepare.py
- Create: tests/qualification/test_storage.py

- [ ] Write failing tests with a fake MemoryRuntime proving that sessions replay once in order, each write is followed by inventory/alignment, and each SCEU receives native candidates plus common-reranked ordering. At checkpoint c the eligible store must contain exactly sessions 0..c-1; retrieve before writing the current checkpoint surface and test SCEUs on write boundaries.
- [ ] Implement prepare_prefix(task, spec, runtime, reranker, storage). It must atomically write a complete artifact only after all checkpoints succeed; failures leave an explicit failed preparation and no valid artifact.
- [ ] Use canonical public transcripts only. Pass no latent state, state IDs, valid actions, future values, or ideal summaries to the runtime.
- [ ] Record the complete stored→candidate→native retrieved/common reranked chain, normalized mutations, N_write/N_live, content attribution, latency/usage, and backend diagnostics.
- [ ] Add artifact load/verify methods that recalculate every nested hash and reject profile/source/model/config/surface mismatches. A-MEM interruption policy is full rerun from session zero.
- [ ] Test deterministic repeated preparation, atomic replacement, corrupt nested checkpoint rejection, empty-start enforcement, candidate subset/order validation, and close-on-success/failure.
- [ ] Run:

    uv run pytest tests/qualification/test_prepare.py tests/qualification/test_storage.py -q
    uv run mypy src/lhmsb/qualification/prepare.py src/lhmsb/qualification/storage.py
    uv run ruff check src/lhmsb/qualification/prepare.py src/lhmsb/qualification/storage.py tests/qualification/test_prepare.py

- [ ] Commit with message: feat: prepare immutable memory prefix artifacts

## Task 6: Split policy evaluation from memory mutation and add controls

**Files**

- Create: src/lhmsb/qualification/evaluate.py
- Modify: src/lhmsb/qualification/runner.py
- Modify: src/lhmsb/qualification/providers.py
- Test: tests/qualification/test_evaluate.py
- Modify: tests/qualification/test_runner.py

- [ ] Write failing tests covering all seven conditions from one episode and shared prefix artifacts. Assert identical current workspace and action options in every condition.
- [ ] Implement evaluate_task so Workspace-only adds nothing, Full-context adds canonical history, Oracle adds sanitized minimal current state, Flat adds common-reranked objects, and each managed system emits separate native/common results from the artifact.
- [ ] Assert evaluation never receives a MemoryRuntime and cannot mutate a store. Prefix artifacts are read-only and hash-verified before every task.
- [ ] Keep repeated baselines and leave-one-visible-object-out interventions for external-memory branches. Add stale replacement when a current supporting candidate exists. Controls use one baseline and no memory intervention.
- [ ] Use explicit condition capabilities for intervention count; remove startswith checks from schema-v2 execution.
- [ ] Record visible object count/chars, candidate/retrieved/visible IDs, prefix hash, prompt/transcript/workspace hashes, selected action, programmatic behavior, four drift flags, provider route/model identity, and usage.
- [ ] Test Full-context overflow hard failure, no gold leakage, prefix-hash mismatch, missing readout, visible IDs outside candidates, deterministic hashes, and old runner parity.
- [ ] Run:

    uv run pytest tests/qualification/test_evaluate.py tests/qualification/test_runner.py tests/qualification/test_providers.py -q
    uv run mypy src/lhmsb/qualification/evaluate.py src/lhmsb/qualification/runner.py
    uv run ruff check src/lhmsb/qualification/evaluate.py src/lhmsb/qualification/runner.py tests/qualification/test_evaluate.py

- [ ] Commit with message: refactor: separate prefix preparation from policy evaluation

## Task 7: Adapt Mem0 to the shared prefix contract

**Files**

- Modify: src/lhmsb/adapters/mem0_qualification.py
- Create: src/lhmsb/qualification/factory.py
- Test: tests/qualification/test_mem0_prefix.py
- Modify: tests/qualification/test_mem0_adapter.py
- Modify: tests/qualification/test_mem0_vertical_slice.py

- [ ] Write failing contract tests that prepare one four-session episode with a fake Mem0 v2 backend and compare normalized writes, inventory, candidates, common rerank, and usage against the prior runner.
- [ ] Add a factory entry for the controlled Mem0 profile using fixed DeepSeek writer, common TEI embedding, isolated Qdrant collection, and isolated SQLite history.
- [ ] Ensure the prefix namespace excludes policy identity and includes run, episode, backend, and profile identity. All three policies must reference the same artifact hash.
- [ ] Preserve native ADD/UPDATE/DELETE history and existing Responses bridge instrumentation. Do not enable the optional Mem0 reranker.
- [ ] Test empty-start checks, teardown, source/package mismatch, native-ID preservation, provider/model mismatch, and no schema-v1 regression.
- [ ] Run:

    uv run pytest tests/qualification/test_mem0_prefix.py tests/qualification/test_mem0_adapter.py tests/qualification/test_mem0_vertical_slice.py -q
    uv run mypy src/lhmsb/adapters/mem0_qualification.py src/lhmsb/qualification/factory.py
    uv run ruff check src/lhmsb/adapters/mem0_qualification.py src/lhmsb/qualification/factory.py tests/qualification/test_mem0_prefix.py

- [ ] Commit with message: feat: prepare shared Mem0 controlled prefixes

## Task 8: Implement the official A-MEM controlled adapter

**Files**

- Create: src/lhmsb/adapters/amem_qualification.py
- Create: src/lhmsb/qualification/deepseek_writer.py
- Test: tests/qualification/test_amem_adapter.py
- Test: tests/qualification/test_deepseek_writer.py

- [ ] Build a fake module matching the pinned official AgenticMemorySystem API. Write failing tests for add_note, read, update, delete, search_agentic, in-memory notes, Chroma rows, and malformed upstream responses.
- [ ] Implement a DeepSeek JSON bridge that converts the official json_schema request into json_object, retains the schema in the prompt, validates locally, and records model/route/usage. It must never use an OpenAI key or default OpenAI base URL.
- [ ] Inject the common BGE-M3 embedding boundary into A-MEM without downloading another embedding model.
- [ ] Implement AMemQualificationAdapter using add_note exactly as pinned. Do not call the unused analyze_content helper and do not repair upstream links or neighbor-update behavior.
- [ ] Normalize each MemoryNote as one live object. Derive mutations and link changes from before/after snapshots. Record native scores with their lower-is-better distance semantics and preserve link-expanded rows without inventing scores.
- [ ] Emit link-validity, dangling-link, missing-target, in-memory-versus-Chroma consistency, and silent-degradation diagnostics.
- [ ] Test deterministic benchmark-owned UUID input where supported, fixed-process/fresh-directory behavior, full-rerun requirement, native search ordering, common-rerank candidate freezing, and hard failure when official package identity/commit/API differs.
- [ ] Run:

    uv run pytest tests/qualification/test_amem_adapter.py tests/qualification/test_deepseek_writer.py -q
    uv run mypy src/lhmsb/adapters/amem_qualification.py src/lhmsb/qualification/deepseek_writer.py
    uv run ruff check src/lhmsb/adapters/amem_qualification.py src/lhmsb/qualification/deepseek_writer.py tests/qualification/test_amem_adapter.py tests/qualification/test_deepseek_writer.py

- [ ] Commit with message: feat: add controlled A-MEM qualification adapter

## Task 9: Implement the official MemOS-Tree controlled adapter

**Files**

- Create: src/lhmsb/adapters/memos_qualification.py
- Create: src/lhmsb/qualification/neo4j.py
- Test: tests/qualification/test_memos_adapter.py
- Test: tests/qualification/test_neo4j.py

- [ ] Build fake TreeTextMemory, SimpleStructMemReader, MemoryManager, reorganizer, and graph-store boundaries matching the pinned release. Write failing tests for synchronous add, reorganize queue/wait, vector/graph-expanded search, archived nodes, MERGED_TO lineage, structural nodes, and graph failures.
- [ ] Use the pinned official Neo4j Python Bolt driver for the minimal inventory boundary needed to list nodes/edges and count storage. Each preparation must receive a freshly created Neo4j Community volume; validate an empty graph before use and reject volume reuse/contamination.
- [ ] Implement MemOSTreeQualificationAdapter with reorganize=true, internet retrieval disabled, synchronous add, frozen fast search, common BGE-M3 embedding, and fixed DeepSeek reader/reorganizer/dispatcher. Add a MemOS-specific DeepSeek transport/config contract proving every LLM component uses only the configured endpoint/model, records usage, rejects substitution, and never invokes an upstream default provider.
- [ ] After every session, wait until the reorganizer is idle or fail on timeout. Snapshot graph before/after and derive node add/update/archive/delete plus edge add/remove events.
- [ ] Count all live retrievable graph nodes in N_live, label structural/topic versus content nodes, exclude archived/deleted nodes from N_live, and keep edges separate from object count.
- [ ] Never instantiate GeneralTextMemory. Missing tree APIs, unsupported configuration, reorganizer timeout, package/source drift, or graph namespace contamination are terminal.
- [ ] Test native candidate order, graph expansion provenance, common-rerank freezing, storage footprints, mutation lineage, cleanup, and deterministic normalized trace hashes.
- [ ] Run:

    uv run pytest tests/qualification/test_memos_adapter.py tests/qualification/test_neo4j.py -q
    uv run mypy src/lhmsb/adapters/memos_qualification.py src/lhmsb/qualification/neo4j.py
    uv run ruff check src/lhmsb/adapters/memos_qualification.py src/lhmsb/qualification/neo4j.py tests/qualification/test_memos_adapter.py tests/qualification/test_neo4j.py

- [ ] Commit with message: feat: add controlled MemOS Tree qualification adapter

## Task 10: Add multisystem CLI orchestration and resumability

**Files**

- Modify: src/lhmsb/qualification/cli.py
- Modify: src/lhmsb/qualification/__main__.py
- Modify: src/lhmsb/qualification/factory.py
- Test: tests/qualification/test_multisystem_cli.py
- Modify: tests/qualification/test_cli.py

- [ ] Write failing CLI tests for plan-systems, prepare-task, evaluate-task, run-evaluation-matrix, aggregate-systems, validate-systems, preflight-systems, and smoke-systems --dry-run.
- [ ] plan-systems writes run_manifest.json, prepare_tasks.jsonl, and evaluation_task_templates.jsonl. Add finalize-evaluation-plan to verify all required artifacts and atomically materialize tasks.jsonl with 21 executable tasks/30 cells in stable template index order.
- [ ] prepare-task dispatches only the task's backend factory, produces an atomic artifact, and is independently retryable. A failed backend does not block other preparations but blocks dependent evaluation tasks.
- [ ] evaluate-task loads no upstream memory package, verifies the artifact hash, and writes one atomic policy-condition task result. Completed cells with the same input hash are reused; mismatches require explicit force.
- [ ] Keep task indices stable, deterministic, and unique. Include code commit/dirty bit, dataset, config, policy routes, writer, source pins, dependency locks, image IDs, model files, hardware profile, and prefix hashes in identities.
- [ ] Preserve every existing schema-v1 command and default. Add explicit --config selection; do not redirect old Mem0 commands to the new matrix.
- [ ] Test missing artifacts, corrupt artifacts, partial completion, independent task continuation, secret redaction, force semantics, deterministic replanning, and four-session dry-run with fake factories.
- [ ] Run:

    uv run pytest tests/qualification/test_multisystem_cli.py tests/qualification/test_cli.py -q
    uv run mypy src/lhmsb/qualification/cli.py src/lhmsb/qualification/factory.py
    uv run ruff check src/lhmsb/qualification/cli.py src/lhmsb/qualification/factory.py tests/qualification/test_multisystem_cli.py

- [ ] Commit with message: feat: orchestrate resumable multisystem qualification

## Task 11: Generalize metrics, report, and validation

**Files**

- Modify: src/lhmsb/qualification/metrics.py
- Modify: src/lhmsb/qualification/report.py
- Modify: src/lhmsb/qualification/validate.py
- Test: tests/qualification/test_multisystem_metrics.py
- Modify: tests/qualification/test_metrics.py
- Modify: tests/qualification/test_report.py
- Create: tests/qualification/test_validate.py

- [ ] Add hand-computed failing fixtures for each backend/readout and every controlled comparison. Keep undefined denominators null.
- [ ] Compute behavior score/correct rate and constraint-loss, plan-deviation, local-over-global, and stale-state drift.
- [ ] Compute N_write/N_live per checkpoint, mutation counts, current coverage, stale retention/coexistence, candidate/visible coverage, stale exposure, shortfall, retrieval-to-visible yield, and causal-use labels.
- [ ] Compute gain beyond Workspace, oracle-gap closure, gain over Flat, gap to Full-context, and native-common delta per policy and matched SCEU. Never average native and common readouts into one cell.
- [ ] Stratify behavior and drift by live memory-object count. Keep characters, tokens, bytes, calls, and latency auxiliary.
- [ ] Emit required prefix manifests, prepare/evaluation tasks/results, normalized memory events/inventory/retrieval, graph diagnostics, interventions, provider usage, metrics.json, metrics_by_cell.json, scorecard.csv, summary.json, and validation.json.
- [ ] Generalize validation from startswith(mem0) to condition definitions. Reconstruct stored→candidate→retrieved→visible→behavior, verify all foreign keys/hashes/order/subsets, and require exactly 30 scored cells per complete episode.
- [ ] Preserve schema-v1 report/validation behavior.
- [ ] Run:

    uv run pytest tests/qualification/test_multisystem_metrics.py tests/qualification/test_metrics.py tests/qualification/test_report.py tests/qualification/test_validate.py -q
    uv run mypy src/lhmsb/qualification/metrics.py src/lhmsb/qualification/report.py src/lhmsb/qualification/validate.py
    uv run ruff check src/lhmsb/qualification/metrics.py src/lhmsb/qualification/report.py src/lhmsb/qualification/validate.py tests/qualification/test_multisystem_metrics.py

- [ ] Commit with message: feat: report multisystem memory lifecycle metrics

## Task 12: Build reproducible server images and orchestration

**Files**

- Create: docker/core-worker.Dockerfile
- Create: docker/amem-worker.Dockerfile
- Create: docker/memos-worker.Dockerfile
- Create: docker/locks/amem-requirements.txt
- Create: docker/locks/memos-requirements.txt
- Create: docker/locks/amem-wheelhouse-manifest.json
- Create: docker/locks/memos-wheelhouse-manifest.json
- Create: deploy/compose.systems.yaml
- Create: deploy/slurm/systems_qualification.sbatch
- Create: scripts/lib/systems_common.sh
- Create: scripts/bootstrap_systems_server.sh
- Create: scripts/preflight_systems.sh
- Create: scripts/run_systems_smoke.sh
- Create: scripts/run_systems_qualification.sh
- Create: scripts/verify_system_images.sh
- Create: docs/systems-server-workflow.md
- Modify: src/lhmsb/qualification/preflight.py
- Modify: .env.example
- Modify: .dockerignore
- Test: tests/qualification/test_systems_deploy_assets.py
- Test: tests/qualification/test_systems_scripts.py
- Modify: tests/qualification/test_preflight.py

- [ ] Write static failing tests for four isolated worker images, one benchmark commit, exact upstream pins, shared Qdrant/Neo4j/embedding/reranker services, two distinct A100 assignments, non-root workers, read-only source mounts, persistent data root, healthchecks, pull_policy never, and no credential copy.
- [ ] Pin Qdrant, Neo4j 5, TEI, and CUDA base images by immutable digest in a generated image manifest. Bootstrap must reject unresolved tags in live mode.
- [ ] Build core and Mem0 from the repository lock. Commit exact transitive, hash-locked requirements for A-MEM and MemOS; clone only the pinned commits, verify source archives, populate per-image wheelhouses with --require-hashes, and build dedicated dependency-isolated images without online resolution. Save every wheel manifest, image archive, and digest under the data root.
- [ ] Use only OPENCODE_ZEN_API_KEY and DEEPSEEK_API_KEY plus documented base-URL overrides. Only worker services receive provider credentials/egress.
- [ ] Split preflight into reusable repository/data/provider/service gates and condition-aware lifecycle gates. Add repository-only checks for config/dataset/source/image/model contracts and live checks for two A100s, TEI dimensions/order, Qdrant/Neo4j isolation, exact package identities, writer/policy model identities, lifecycle add/search/update/delete where supported, and trace reconstruction. Unselected backend gates must skip without requiring their packages or services.
- [ ] Add verify_system_images.sh to restore and start every image offline, then assert benchmark commit, upstream package/source identity, CLI entrypoint, and cross-image prefix-artifact schema compatibility before any provider call. Add a four-session smoke that prepares all four prefixes, finalizes the evaluation plan, and evaluates all 21 tasks/30 cells before the 16-session run.
- [ ] The Slurm script requests at least two A100 GPUs, maps one to embedding and one to reranking, uses a unique Compose project and data namespace, takes a shared-state lock, restores pinned images, cleans up on signals, prepares prefixes serially, evaluates resumably, aggregates, and validates.
- [ ] Make bootstrap and run wrappers support --dry-run without network, Docker, GPUs, or secrets. Preserve old Mem0 server scripts unchanged.
- [ ] Document exact migration commands: git checkout pinned commit, bootstrap, repository preflight, sbatch smoke, inspect validation, then sbatch full qualification. Document expected output paths and failure recovery.
- [ ] Run:

    uv run pytest tests/qualification/test_systems_deploy_assets.py tests/qualification/test_systems_scripts.py -q
    uv run ruff check tests/qualification/test_systems_deploy_assets.py tests/qualification/test_systems_scripts.py
    bash -n scripts/lib/systems_common.sh scripts/bootstrap_systems_server.sh scripts/preflight_systems.sh scripts/run_systems_smoke.sh scripts/run_systems_qualification.sh deploy/slurm/systems_qualification.sbatch

- [ ] Commit with message: ops: add reproducible multisystem A100 workflow

## Task 13: Integration, backward compatibility, and handoff gate

**Files**

- Create: tests/qualification/test_multisystem_vertical_slice.py
- Modify: README.md
- Modify: docs/systems-server-workflow.md

- [ ] Write an offline four-session vertical slice using fake Mem0/A-MEM/MemOS, fake Qdrant/Neo4j, fake TEI, and fake policies. It must produce 4 valid prefix artifacts, 21 policy tasks, 30 scored cells, at least one reconstructable causal-use chain, distinct Workspace/Full/Oracle/Flat/system outcomes, and a valid report.
- [ ] Run the new repository-only preflight and dry-run commands from a clean checkout. Confirm no network or secret access.
- [ ] Run the frozen 16-session planning path and verify the Full-context maximum is below 100000 characters.
- [ ] Run the complete tests:

    uv run pytest -q

  On macOS, only the existing Linux resource-module test may remain explicitly deselected. Record the exact pass/skip/deselect counts.
- [ ] Run static verification:

    uv run ruff check src tests
    uv run mypy src/lhmsb
    bash -n scripts/*.sh scripts/lib/*.sh deploy/slurm/*.sbatch
    git diff --check

- [ ] Request a specification-compliance review and then a code-quality review. Resolve every finding with focused tests and rerun the relevant gates.
- [ ] Generate a server handoff manifest containing branch/commit, dataset/config hashes, exact source pins, image/model contracts, required secret names, expected 4/21/30 counts, and the two server commands.
- [ ] Commit with message: test: gate three-system controlled benchmark
- [ ] Use superpowers:verification-before-completion before claiming readiness.

## Server acceptance after migration

These checks are intentionally not claimed locally:

1. Run bootstrap_systems_server.sh on the server and verify archived images/source/model manifests.
2. Submit the four-session smoke through Slurm with two A100s.
3. Confirm exact Zen and DeepSeek model identities; reject substitutions.
4. Confirm all four prefixes, 21 tasks, 30 cells, and validation.json ok=true.
5. Inspect a stored→candidate→retrieved→visible→behavior chain for each backend.
6. Run the frozen 16-session qualification only after the smoke passes.

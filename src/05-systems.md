# 05 — Systems Under Test & Adapter Interface

> **Status**: canonical for v1. Task 5 implements the `MemorySystemAdapter` ABC
> defined here. Tasks 12–16 implement per-system adapters against these exact
> signatures. Any deviation from these signatures is a spec violation.

---

## 1. MemorySystemAdapter — Canonical Interface

All memory systems under test (including the no-memory control) are wrapped
behind a common abstract interface. The adapter is the ONLY path through which
the agent harness reads or writes memory. LangGraph's built-in checkpointer and
any framework-level caches MUST be disabled — all persistence flows through the
adapter.

### 1.1 Required Methods (Exact Signatures)

```python
class MemorySystemAdapter(ABC):

    @abstractmethod
    def initialize(self, *, user_id: str, session_id: str | None = None, **config) -> None:
        """Set up the memory backend for a user. Called once per user before
        the first session. session_id=None means no session scope yet."""
        ...

    @abstractmethod
    def reset(self, *, user_id: str) -> None:
        """Delete ALL memory for a user. Used between episodes to ensure clean
        state. Must be idempotent."""
        ...

    @abstractmethod
    def add_memory(self, content: str, *, user_id: str,
                   session_id: str | None = None,
                   metadata: dict | None = None) -> str:
        """Ingest content into the memory system. Returns a memory_id
        (unique, stable string identifier for this memory entry).

        All internal LLM/embedding calls triggered by this operation MUST
        be wrapped in memory_scope() so their tokens are counted."""
        ...

    @abstractmethod
    def search(self, query: str, *, user_id: str,
               session_id: str | None = None,
               top_k: int = 10, **filters) -> SearchResult:
        """Retrieve relevant memories for a query.

        Returns a SearchResult containing a list of MemoryEntry objects
        and a total_count. Results should be relevance-ranked.

        Internal LLM/embedding calls triggered by this operation MUST be
        wrapped in memory_scope()."""
        ...

    @abstractmethod
    def update_memory(self, memory_id: str, *,
                      content: str | None = None,
                      metadata: dict | None = None) -> None:
        """Update the content and/or metadata of an existing memory entry.
        content=None means keep existing content; metadata=None means keep
        existing metadata. At least one must be provided."""
        ...

    @abstractmethod
    def delete_memory(self, memory_id: str) -> None:
        """Remove a memory entry. The entry should no longer appear in
        search results after deletion. Idempotent — deleting a non-existent
        entry is a no-op, not an error.

        Note: the benchmark scores BEHAVIOR, not implementation. Whether
        the system implements delete via removal, tombstone, edge
        invalidation, or retrieval-filtering is irrelevant — only that
        the retracted/deleted fact no longer influences search results."""
        ...
```

**Return types**:

```python
@dataclass(frozen=True)
class MemoryEntry:
    memory_id: str
    content: str
    metadata: dict | None
    created_at: str       # ISO 8601 timestamp
    updated_at: str       # ISO 8601 timestamp
    score: float | None   # relevance score from search, None for direct retrieval

@dataclass(frozen=True)
class SearchResult:
    results: list[MemoryEntry]
    total_count: int      # total matching results (may be > len(results))
```

### 1.2 Graceful Capability Degradation

Not all memory systems support every operation. The `Capabilities` introspection
mechanism allows the harness to query what a backend supports:

```python
@dataclass(frozen=True)
class Capabilities:
    supports_add: bool = True
    supports_search: bool = True
    supports_update: bool = True
    supports_delete: bool = True
    supports_reset: bool = True
    supports_sessions: bool = False
    supports_reflection: bool = False
    supports_forgetting: bool = False
```

Adapters expose `get_capabilities() -> Capabilities`. When the harness or agent
calls an unsupported operation, the adapter must raise `UnsupportedOperation`
(a logged but non-fatal exception) — never crash, never silently ignore.

### 1.3 Optional Capability Mixins

Adapters MAY implement these mixins to expose additional memory lifecycle
operations. The benchmark uses them when available; does not require them.

```python
class ReflectionCapability(ABC):
    """Memory systems that support consolidation / self-reorganization."""

    @abstractmethod
    def reflect(self, *, user_id: str, session_id: str | None = None) -> None:
        """Trigger a reflection/consolidation pass. Internal LLM tokens
        from this operation MUST be counted under memory_scope()."""
        ...

    @abstractmethod
    def summarize(self, *, user_id: str, session_id: str | None = None,
                  query: str | None = None) -> str:
        """Produce a summary of stored memories, optionally scoped to a query."""
        ...


class ForgettingCapability(ABC):
    """Memory systems with explicit decay / forgetting mechanisms."""

    @abstractmethod
    def apply_decay(self, *, user_id: str, **params) -> None:
        """Apply a forgetting/decay step. May reduce relevance scores,
        archive old memories, or physically delete low-importance entries."""
        ...


class SessionCapability(ABC):
    """Memory systems with explicit session/thread grouping."""

    @abstractmethod
    def list_sessions(self, *, user_id: str) -> list[str]:
        """Return all session IDs for a user."""
        ...

    @abstractmethod
    def get_session_memories(self, *, user_id: str,
                              session_id: str) -> list[MemoryEntry]:
        """Return all memory entries scoped to a session."""
        ...

    @abstractmethod
    def promote_session(self, *, user_id: str, session_id: str) -> None:
        """Promote session-scoped memories to global/user scope."""
        ...
```

---

## 2. Systems Under Test

### 2.1 Leaderboard Conditions (6 systems)

These six conditions appear on the real leaderboard (native track primary,
controlled track secondary, never mixed).

| # | Condition | System | Description | Key API |
|---|---|---|---|---|
| 1 | `no_memory` | No-Memory Control | Stores nothing across sessions. `search()` always returns empty. The counterfactual baseline for ROI. | `add`/`update`/`delete` are no-ops returning valid IDs |
| 2 | `chroma` | ChromaDB | Plain vector-store baseline. In-memory/offline. | `collection.add/query/upsert/delete` |
| 3 | `mem0` | Mem0 | Hybrid semantic + BM25 + entity memory. Internal LLM on `add`. | `Memory.add/search/update/delete` |
| 4 | `letta` | Letta / AI-Memory-SDK | Agent self-editing memory blocks with sleeptime reflection. | `add_messages/search/get_memory/delete_block` |
| 5 | `graphiti` | Graphiti (Zep) | Temporal knowledge graph with auto time-invalidation. | `add_episode/search/remove_episode` |
| 6 | `cognee` | Cognee | Multi-stage pipeline (`cognify`) with self-reorg (`memify`). File-based defaults. | `remember/recall/forget/improve` |

### 2.2 Sensitivity / Calibration Conditions (2 fakes)

These are NOT on the real leaderboard. They are calibration conditions used to
validate metric sensitivity: the task score under `fake_perfect` must exceed the
score under `fake_bad` by a clear margin, or the metrics are broken.

| # | Condition | Behavior |
|---|---|---|
| F1 | `fake_perfect` | Oracle memory. Returns exactly the relevant current (non-retracted) facts for any query. Uses the episode's ground-truth fact store. Upper bound for what a memory system could achieve. |
| F2 | `fake_bad` | Adversarial memory. Returns plausible but incorrect or retracted facts. Lower bound — any real memory system should beat this. |

### 2.3 Capability Matrix

| Condition | Reflection | Forgetting | Sessions |
|---|---|---|---|
| `no_memory` | — | — | — |
| `chroma` | — | — | — |
| `mem0` | implicit on add | — | — |
| `letta` | `reflect()` (sleeptime) | — | via blocks |
| `graphiti` | — | temporal auto-invalidation | via `group_id` |
| `cognee` | `memify()` / `improve()` | — | via `session_id` |
| `fake_*` | — | — | — |

---

## 3. Track Rules — Native vs Controlled

### 3.1 Native Track (PRIMARY)

Each memory system is tested as it ships: its default configuration, its own
internal LLM model, its own embedder, its own default parameters. This is the
primary leaderboard because it represents *what a practitioner would actually
deploy*.

**All internal LLM/embedding cost is instrumented and reported**, so a system
that uses a more expensive internal model is accountable for that cost in its
ROI. The native track is NOT "unfair" — it's "fully accounted."

### 3.2 Controlled Track (SECONDARY)

Where a memory system supports configuration of its internal LLM and embedder,
it is also tested with a pinned shared model (the same open-weights model used
by the agent and controlled-track peers). This isolates the memory system's
architecture from its model choice.

**Rules**:
- The controlled track uses the SAME pinned agent model across all systems
  that support model configuration.
- Systems that do NOT support model configuration (e.g., ChromaDB, no-memory)
  are trivially present in both tracks (no internal LLM to pin).
- Controlled-track results are reported in a separate table/section, never
  merged with native-track results in a single leaderboard.
- The `RunConfig.track` field records which track a run belongs to.

### 3.3 Track Comparison

The difference between native and controlled ROI for a system quantifies how
much of its performance is attributable to its internal model choice vs its
architecture. A system whose native ROI drops significantly in the controlled
track is model-dependent; a system that maintains ROI is architecture-robust.

---

## 4. Full-Lifecycle Cost Instrumentation

### 4.1 Requirement

Every adapter MUST be wrapped so that ALL LLM and embedding tokens consumed
by the memory system internally are counted. This includes:

- **Add-time processing**: Mem0's extraction LLM, Cognee's `cognify` pipeline,
  Graphiti's entity extraction on `add_episode`, Letta's block self-edits.
- **Search-time processing**: any query rewriting, embedding generation,
  reranking LLMs the memory system invokes during `search()`.
- **Reflection/consolidation**: Letta's sleeptime, Cognee's `memify()`/`improve()`.
- **Embeddings**: tokens embedded for vector storage AND tokens consumed by
  embedding API calls.

These tokens land in `CostVector.mem_internal_in_tokens` and
`mem_internal_out_tokens` (and `embedding_tokens` / `embedding_calls`),
separate from the agent loop's tokens (`agent_input_tokens` / `agent_output_tokens`).

### 4.2 Mechanism

The cost instrumentation layer (Task 6) provides:

- `CostMeter`: a thread-safe accumulator with scoped attribution.
- `memory_scope()`: a context manager. Any LLM/embedding call made inside
  `with meter.memory_scope():` is attributed to the memory system.
- `instrumented_llm(client)` and `instrumented_embedder(fn)`: wrappers that
  automatically count tokens and respect the active scope.

Adapter code looks like:

```python
def add_memory(self, content, *, user_id, session_id=None, metadata=None):
    with self.cost_meter.memory_scope():
        result = self._backend.add(content, user_id=user_id, ...)
    return result.memory_id
```

The `memory_scope()` ensures any LLM calls the backend makes internally are
counted as `mem_internal_*`, not as agent tokens.

### 4.3 Exclusion Rules

The following costs are NOT counted in system CostVectors:

- **Dataset generation**: one-time cost of creating frozen episodes.
- **Judge tokens**: the sparse judge's LLM calls.
- **Surface rendering**: rendering structured events to natural text (frozen
  cache, excluded from episode cost).
- **Harness overhead**: the LangGraph framework's own token usage (minimal,
  deterministic, tracked but excluded from system comparisons).

### 4.4 Strict Mode

When `strict_instrumentation=True` in the run config, any LLM or embedding call
made outside of an explicit scope (agent or memory) raises a
`CostInstrumentationError`. This prevents silently uncounted tokens. In
non-strict mode, unscoped calls are attributed to a catch-all `unscoped` bucket
and flagged with a warning.

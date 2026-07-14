"""Full-lifecycle cost meter and scoped LLM/embedding instrumentation.

This module is the correctness backbone of the headline **Memory ROI** metric:
attribution between *agent-loop* cost and *memory-system-internal* cost must be
exact, or the ROI is dishonest (see ``spec/02-metrics.md`` §1 and
``spec/05-systems.md`` §4).

Core ideas
----------
* ``CostMeter`` is a thread-safe accumulator that produces a single
  :class:`lhmsb.types.CostVector` per episode x condition via
  :meth:`CostMeter.to_cost_vector`.  The full vector is ALWAYS retained; it is
  collapsed to a scalar ONLY in :func:`lhmsb.cost.scalarize.scalarize`.

* **Scopes** decide where auto-instrumented tokens land.  A memory adapter wraps
  its backend call so any LLM the backend invokes internally is attributed to
  the memory system::

      def add_memory(self, content, *, user_id, session_id=None, metadata=None):
          with self.cost_meter.memory_scope():
              result = self._backend.add(content, user_id=user_id, ...)
          return result.memory_id

  Available scopes:
    - :meth:`CostMeter.agent_scope`       -> ``agent_*`` fields
    - :meth:`CostMeter.memory_scope`      -> ``mem_internal_*`` fields
    - :meth:`CostMeter.reflection_scope`  -> ``reflection_tokens``
    - :meth:`CostMeter.excluded_scope`    -> NOT counted (dataset-gen / judge /
      surface-rendering / harness overhead per ``spec/05-systems.md`` §4.3)

* **Strict mode** (``strict_instrumentation=True``): an instrumented LLM or
  embedding call made outside any explicit scope raises
  :class:`CostInstrumentationError` so uninstrumented internal calls fail loudly.
  In non-strict mode such a call is routed to a catch-all ``unscoped`` bucket and
  a :class:`CostInstrumentationWarning` is emitted (never silently dropped).

* **Token counting** uses the model tokenizer (``tiktoken``) where available and
  otherwise a deterministic whitespace word-count fallback (see
  :func:`count_tokens`).
"""

from __future__ import annotations

import threading
import warnings
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from enum import Enum, auto
from typing import Generic, TypeVar

from lhmsb.types import CostVector

__all__ = [
    "CostInstrumentationError",
    "CostInstrumentationWarning",
    "CostMeter",
    "InstrumentedEmbedder",
    "InstrumentedLLM",
    "Scope",
    "count_tokens",
    "instrumented_embedder",
    "instrumented_llm",
]

OutT = TypeVar("OutT")

_LATENCY_KINDS = ("retrieval", "write", "update")


class CostInstrumentationError(RuntimeError):
    """Raised in strict mode when an LLM/embedding call is made outside any
    explicit cost scope (agent / memory / reflection / excluded).

    Signals uninstrumented internal cost so accounting stays honest.
    """


class CostInstrumentationWarning(UserWarning):
    """Emitted in non-strict mode when an instrumented call is made outside any
    scope; the tokens are routed to the ``unscoped`` bucket, not dropped.
    """


class Scope(Enum):
    """Cost-attribution scope for instrumented LLM/embedding calls."""

    AGENT = auto()
    MEMORY = auto()
    REFLECTION = auto()
    EXCLUDED = auto()


def _fallback_count(text: str) -> int:
    """Deterministic, model-independent token estimate: whitespace word count.

    Used when no model tokenizer is available.  Deterministic and trivially
    hand-computable (``"a b c"`` -> 3, ``""`` -> 0), which matters for
    reproducible cost accounting and TDD fixtures.
    """
    return len(text.split())


def _model_token_count(text: str, model: str) -> int | None:
    """Count tokens with the model's ``tiktoken`` encoding, or ``None`` if no
    encoding is available (tiktoken not installed or model unknown).

    ``tiktoken`` is an optional dependency; the import is local so the package
    works without it.
    """
    try:
        import tiktoken
    except ImportError:
        return None
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        return None
    return len(encoding.encode(text))


def count_tokens(text: str, model: str | None = None) -> int:
    """Return the token count of ``text``.

    Uses the model tokenizer where available (``model`` maps to a ``tiktoken``
    encoding); otherwise falls back to a deterministic whitespace word count.
    ``model=None`` always uses the fallback.
    """
    if model is not None:
        counted = _model_token_count(text, model)
        if counted is not None:
            return counted
    return _fallback_count(text)


class _ScopeStack(threading.local):
    """Per-thread stack of active scopes.

    Subclassing ``threading.local`` gives every thread its own fresh ``stack``
    (the overridden ``__init__`` runs once per thread on first access), so
    concurrent agent/memory work never clobbers each other's scope.
    """

    stack: list[Scope]

    def __init__(self) -> None:
        super().__init__()
        self.stack = []


class CostMeter:
    """Thread-safe full-lifecycle cost accumulator with scoped attribution.

    One ``CostMeter`` is created per episode x condition.  Mutations are guarded
    by a re-entrant lock; the active scope is tracked per thread.
    """

    def __init__(
        self,
        *,
        strict_instrumentation: bool = False,
        model: str | None = None,
    ) -> None:
        self._lock = threading.RLock()
        self._scope_state = _ScopeStack()
        self._strict = strict_instrumentation
        self._model = model

        # ---- system CostVector accumulators (the honest, reported cost) ----
        self._agent_in = 0
        self._agent_out = 0
        self._mem_in = 0
        self._mem_out = 0
        self._embedding_tokens = 0
        self._embedding_calls = 0
        self._storage_bytes = 0
        self._retrieval_latency_ms = 0.0
        self._write_latency_ms = 0.0
        self._update_latency_ms = 0.0
        self._reflection_tokens = 0
        self._num_retrieval_calls = 0

        # ---- non-system buckets (NEVER part of the CostVector) ----
        # Excluded: dataset-gen / judge / surface-rendering / harness overhead.
        self._excluded_in = 0
        self._excluded_out = 0
        self._excluded_embedding_tokens = 0
        self._excluded_embedding_calls = 0
        self._excluded_labels: dict[str, int] = {}
        # Unscoped: non-strict catch-all for uninstrumented calls (+ warning).
        self._unscoped_in = 0
        self._unscoped_out = 0
        self._unscoped_embedding_tokens = 0
        self._unscoped_embedding_calls = 0

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #

    @property
    def model(self) -> str | None:
        """Default tokenizer model used by instrumented wrappers (may be None)."""
        return self._model

    @property
    def strict_instrumentation(self) -> bool:
        """Whether unscoped instrumented calls raise instead of warn."""
        return self._strict

    # ------------------------------------------------------------------ #
    # Scope management
    # ------------------------------------------------------------------ #

    def _current_scope(self) -> Scope | None:
        stack = self._scope_state.stack
        return stack[-1] if stack else None

    @contextmanager
    def agent_scope(self) -> Iterator[None]:
        """Within this block, instrumented tokens attribute to ``agent_*``."""
        self._scope_state.stack.append(Scope.AGENT)
        try:
            yield
        finally:
            self._scope_state.stack.pop()

    @contextmanager
    def memory_scope(self) -> Iterator[None]:
        """Within this block, instrumented tokens attribute to ``mem_internal_*``
        (and embeddings to ``embedding_*``).

        This is the hook memory adapters use so the internal LLM/embedding calls
        their backend makes on add/search/update land in the memory system's
        cost rather than the agent loop's.
        """
        self._scope_state.stack.append(Scope.MEMORY)
        try:
            yield
        finally:
            self._scope_state.stack.pop()

    @contextmanager
    def reflection_scope(self) -> Iterator[None]:
        """Within this block, instrumented LLM tokens attribute to
        ``reflection_tokens`` (explicit ``reflect()`` / consolidation passes).
        """
        self._scope_state.stack.append(Scope.REFLECTION)
        try:
            yield
        finally:
            self._scope_state.stack.pop()

    @contextmanager
    def excluded_scope(self, label: str) -> Iterator[None]:
        """Within this block, instrumented tokens are NOT counted in the system
        ``CostVector``.

        Used for dataset-generation, the LLM judge, surface rendering, and
        harness overhead (``spec/05-systems.md`` §4.3).  The tokens are tracked
        in a separate, labelled bucket for auditing but never enter the ROI.
        """
        with self._lock:
            self._excluded_labels[label] = self._excluded_labels.get(label, 0) + 1
        self._scope_state.stack.append(Scope.EXCLUDED)
        try:
            yield
        finally:
            self._scope_state.stack.pop()

    # ------------------------------------------------------------------ #
    # Direct accumulator methods (scope-independent, explicit attribution)
    # ------------------------------------------------------------------ #

    def add_agent_tokens(self, input_tokens: int, output_tokens: int) -> None:
        """Add agent-loop LLM tokens (prompt/context in, generated out)."""
        with self._lock:
            self._agent_in += input_tokens
            self._agent_out += output_tokens

    def add_memory_internal_tokens(self, input_tokens: int, output_tokens: int) -> None:
        """Add tokens consumed by the memory system's internal LLM calls."""
        with self._lock:
            self._mem_in += input_tokens
            self._mem_out += output_tokens

    def add_embedding(self, tokens: int, calls: int) -> None:
        """Add embedding tokens and embedding API call count."""
        with self._lock:
            self._embedding_tokens += tokens
            self._embedding_calls += calls

    def add_storage_bytes(self, n: int) -> None:
        """Add bytes of backend storage used by the memory system."""
        with self._lock:
            self._storage_bytes += n

    def add_reflection_tokens(self, tokens: int) -> None:
        """Add tokens consumed by an explicit reflect()/consolidation pass."""
        with self._lock:
            self._reflection_tokens += tokens

    def record_latency(self, kind: str, ms: float) -> None:
        """Record wall-clock latency (ms) for an adapter operation.

        ``kind`` must be one of ``retrieval`` / ``write`` / ``update``.
        """
        with self._lock:
            if kind == "retrieval":
                self._retrieval_latency_ms += ms
            elif kind == "write":
                self._write_latency_ms += ms
            elif kind == "update":
                self._update_latency_ms += ms
            else:
                raise ValueError(
                    f"Unknown latency kind {kind!r}; expected one of {_LATENCY_KINDS}."
                )

    def incr_retrieval(self) -> None:
        """Increment the count of ``search()`` invocations."""
        with self._lock:
            self._num_retrieval_calls += 1

    # ------------------------------------------------------------------ #
    # Scope-aware recording (used by the instrumented wrappers and by adapters
    # that read token usage directly from a backend response)
    # ------------------------------------------------------------------ #

    def record_llm_call(self, input_tokens: int, output_tokens: int) -> None:
        """Attribute one LLM call's tokens to the currently active scope."""
        scope = self._current_scope()
        if scope is Scope.AGENT:
            self.add_agent_tokens(input_tokens, output_tokens)
        elif scope is Scope.MEMORY:
            self.add_memory_internal_tokens(input_tokens, output_tokens)
        elif scope is Scope.REFLECTION:
            self.add_reflection_tokens(input_tokens + output_tokens)
        elif scope is Scope.EXCLUDED:
            with self._lock:
                self._excluded_in += input_tokens
                self._excluded_out += output_tokens
        else:
            self._handle_unscoped_llm(input_tokens, output_tokens)

    def record_embedding_call(self, tokens: int, calls: int = 1) -> None:
        """Attribute one embedding call's tokens to the currently active scope."""
        scope = self._current_scope()
        if scope in (Scope.AGENT, Scope.MEMORY, Scope.REFLECTION):
            self.add_embedding(tokens, calls)
        elif scope is Scope.EXCLUDED:
            with self._lock:
                self._excluded_embedding_tokens += tokens
                self._excluded_embedding_calls += calls
        else:
            self._handle_unscoped_embedding(tokens, calls)

    def _handle_unscoped_llm(self, input_tokens: int, output_tokens: int) -> None:
        if self._strict:
            raise CostInstrumentationError(
                "LLM call made outside any cost scope "
                "(agent / memory / reflection / excluded). Wrap the call in an "
                "explicit scope, or disable strict_instrumentation."
            )
        with self._lock:
            self._unscoped_in += input_tokens
            self._unscoped_out += output_tokens
        warnings.warn(
            "LLM call made outside any cost scope; tokens routed to the "
            "'unscoped' bucket and excluded from the system CostVector.",
            CostInstrumentationWarning,
            stacklevel=3,
        )

    def _handle_unscoped_embedding(self, tokens: int, calls: int) -> None:
        if self._strict:
            raise CostInstrumentationError(
                "Embedding call made outside any cost scope "
                "(agent / memory / reflection / excluded). Wrap the call in an "
                "explicit scope, or disable strict_instrumentation."
            )
        with self._lock:
            self._unscoped_embedding_tokens += tokens
            self._unscoped_embedding_calls += calls
        warnings.warn(
            "Embedding call made outside any cost scope; tokens routed to the "
            "'unscoped' bucket and excluded from the system CostVector.",
            CostInstrumentationWarning,
            stacklevel=3,
        )

    # ------------------------------------------------------------------ #
    # Output / audit
    # ------------------------------------------------------------------ #

    def to_cost_vector(self) -> CostVector:
        """Snapshot the accumulated system cost as a :class:`CostVector`.

        This is the ONLY structured cost output; it is never collapsed to a
        scalar here (that happens only in ``scalarize``).
        """
        with self._lock:
            return CostVector(
                agent_input_tokens=self._agent_in,
                agent_output_tokens=self._agent_out,
                mem_internal_in_tokens=self._mem_in,
                mem_internal_out_tokens=self._mem_out,
                embedding_tokens=self._embedding_tokens,
                embedding_calls=self._embedding_calls,
                storage_bytes=self._storage_bytes,
                retrieval_latency_ms=self._retrieval_latency_ms,
                write_latency_ms=self._write_latency_ms,
                update_latency_ms=self._update_latency_ms,
                reflection_tokens=self._reflection_tokens,
                num_retrieval_calls=self._num_retrieval_calls,
            )

    def excluded_totals(self) -> tuple[int, int]:
        """Return (input_tokens, output_tokens) attributed to excluded scopes."""
        with self._lock:
            return (self._excluded_in, self._excluded_out)

    def excluded_embedding_totals(self) -> tuple[int, int]:
        """Return (embedding_tokens, embedding_calls) in excluded scopes."""
        with self._lock:
            return (self._excluded_embedding_tokens, self._excluded_embedding_calls)

    def excluded_labels(self) -> dict[str, int]:
        """Return a copy of {excluded_scope_label: entry_count}."""
        with self._lock:
            return dict(self._excluded_labels)

    def unscoped_totals(self) -> tuple[int, int]:
        """Return (input_tokens, output_tokens) routed to the unscoped bucket."""
        with self._lock:
            return (self._unscoped_in, self._unscoped_out)

    def unscoped_embedding_totals(self) -> tuple[int, int]:
        """Return (embedding_tokens, embedding_calls) routed to unscoped."""
        with self._lock:
            return (self._unscoped_embedding_tokens, self._unscoped_embedding_calls)

    def has_unscoped(self) -> bool:
        """True if any tokens were routed to the unscoped bucket."""
        with self._lock:
            return bool(
                self._unscoped_in
                or self._unscoped_out
                or self._unscoped_embedding_tokens
                or self._unscoped_embedding_calls
            )


def _as_text(value: object) -> str:
    """Best-effort textual view of an LLM result for output-token counting."""
    return value if isinstance(value, str) else str(value)


class InstrumentedLLM(Generic[OutT]):
    """Wraps an LLM client so each call auto-counts tokens under the active scope.

    The wrapped client is a callable taking a single string prompt and returning
    a result of type ``OutT`` (a string by default; otherwise provide
    ``extract_output`` to obtain the generated text for counting).
    """

    def __init__(
        self,
        client: Callable[[str], OutT],
        meter: CostMeter,
        *,
        model: str | None = None,
        extract_output: Callable[[OutT], str] | None = None,
    ) -> None:
        self._client = client
        self._meter = meter
        self._model = model
        self._extract_output = extract_output

    def __call__(self, prompt: str) -> OutT:
        model = self._model if self._model is not None else self._meter.model
        input_tokens = count_tokens(prompt, model)
        result = self._client(prompt)
        if self._extract_output is not None:
            text = self._extract_output(result)
        else:
            text = _as_text(result)
        output_tokens = count_tokens(text, model)
        self._meter.record_llm_call(input_tokens, output_tokens)
        return result


class InstrumentedEmbedder(Generic[OutT]):
    """Wraps an embedding function so each call auto-counts embedding tokens and
    one embedding call under the active scope.

    The wrapped function accepts a single string or a sequence of strings (a
    batch) and returns the embeddings (``OutT``). One invocation counts as one
    embedding call; tokens are summed across the batch.
    """

    def __init__(
        self,
        fn: Callable[[str | Sequence[str]], OutT],
        meter: CostMeter,
        *,
        model: str | None = None,
    ) -> None:
        self._fn = fn
        self._meter = meter
        self._model = model

    def __call__(self, text_input: str | Sequence[str]) -> OutT:
        model = self._model if self._model is not None else self._meter.model
        texts: list[str] = [text_input] if isinstance(text_input, str) else list(text_input)
        tokens = sum(count_tokens(t, model) for t in texts)
        result = self._fn(text_input)
        self._meter.record_embedding_call(tokens, 1)
        return result


def instrumented_llm(
    client: Callable[[str], OutT],
    meter: CostMeter,
    *,
    model: str | None = None,
    extract_output: Callable[[OutT], str] | None = None,
) -> InstrumentedLLM[OutT]:
    """Wrap ``client`` so its calls auto-count tokens under the active scope."""
    return InstrumentedLLM(client, meter, model=model, extract_output=extract_output)


def instrumented_embedder(
    fn: Callable[[str | Sequence[str]], OutT],
    meter: CostMeter,
    *,
    model: str | None = None,
) -> InstrumentedEmbedder[OutT]:
    """Wrap ``fn`` so its calls auto-count embedding tokens under the active scope."""
    return InstrumentedEmbedder(fn, meter, model=model)

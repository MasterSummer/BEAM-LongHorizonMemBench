"""TDD tests for lhmsb full-lifecycle cost instrumentation (task 6).

Written FIRST (RED) before the implementation in ``src/lhmsb/cost/``.

Validates, per ``spec/02-metrics.md`` §1.3-1.4 and ``spec/05-systems.md`` §4:
  - Scope attribution: tokens inside ``memory_scope()`` -> ``mem_internal_*``;
    inside ``agent_scope()`` -> ``agent_*``; inside ``reflection_scope()`` ->
    ``reflection_tokens``; inside ``excluded_scope()`` -> NOT counted.
  - Direct accumulator methods + fieldwise aggregation into a ``CostVector``.
  - ``scalarize`` reproduces a hand-computed tokens-equivalent EXACTLY.
  - Strict mode raises ``CostInstrumentationError`` on an unscoped call;
    non-strict routes to the ``unscoped`` bucket + warning.
  - The full ``CostVector`` is always retained (collapse only in ``scalarize``).
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from lhmsb.cost import (
    ConversionSheet,
    CostInstrumentationError,
    CostInstrumentationWarning,
    CostMeter,
    ScalarizationWeights,
    count_tokens,
    instrumented_embedder,
    instrumented_llm,
    load_cost_config,
    scalarize,
)
from lhmsb.types import CostVector

CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "cost_weights.yaml"


class FakeLLM:
    """Deterministic fake LLM client: returns a fixed response per call.

    The response is constant so output-token counts are predictable
    independent of the prompt.
    """

    def __init__(self, response: str) -> None:
        self._response = response
        self.calls = 0

    def __call__(self, prompt: str) -> str:
        self.calls += 1
        return self._response


def fake_embedder(text_input: str | list[str]) -> list[list[float]]:
    """Deterministic fake embedder: one vector per input text."""
    n = 1 if isinstance(text_input, str) else len(text_input)
    return [[0.0, 1.0] for _ in range(n)]


# ---------------------------------------------------------------------------
# count_tokens
# ---------------------------------------------------------------------------


class TestCountTokens:
    """count_tokens uses a model tokenizer where available, else a
    deterministic whitespace-word-count fallback."""

    def test_fallback_word_count_no_model(self) -> None:
        assert count_tokens("alpha beta gamma", None) == 3

    def test_fallback_empty_string(self) -> None:
        assert count_tokens("", None) == 0

    def test_fallback_collapses_whitespace(self) -> None:
        assert count_tokens("  a   b \t c \n", None) == 3

    def test_unknown_model_falls_back_to_word_count(self) -> None:
        # An unknown model name has no tokenizer -> deterministic fallback.
        assert count_tokens("a b c d", "definitely-not-a-real-model-xyz") == 4

    def test_known_model_returns_positive_int(self) -> None:
        # If tiktoken is available, a real model maps to an encoding;
        # if not, the fallback still returns a positive int. Either way:
        n = count_tokens("hello world from the benchmark", "gpt-4")
        assert isinstance(n, int)
        assert n >= 1

    def test_deterministic(self) -> None:
        a = count_tokens("the quick brown fox", None)
        b = count_tokens("the quick brown fox", None)
        assert a == b == 4


# ---------------------------------------------------------------------------
# Direct accumulator methods
# ---------------------------------------------------------------------------


class TestDirectMethods:
    """The explicit add_* / record_* methods populate CostVector fields
    independent of the active scope."""

    def test_all_direct_methods_aggregate(self) -> None:
        meter = CostMeter()
        meter.add_agent_tokens(10, 5)
        meter.add_memory_internal_tokens(3, 2)
        meter.add_embedding(100, 4)
        meter.add_storage_bytes(2048)
        meter.record_latency("retrieval", 50.0)
        meter.record_latency("write", 30.0)
        meter.record_latency("update", 20.0)
        meter.add_reflection_tokens(7)
        for _ in range(5):
            meter.incr_retrieval()

        cv = meter.to_cost_vector()
        assert cv == CostVector(
            agent_input_tokens=10,
            agent_output_tokens=5,
            mem_internal_in_tokens=3,
            mem_internal_out_tokens=2,
            embedding_tokens=100,
            embedding_calls=4,
            storage_bytes=2048,
            retrieval_latency_ms=50.0,
            write_latency_ms=30.0,
            update_latency_ms=20.0,
            reflection_tokens=7,
            num_retrieval_calls=5,
        )

    def test_add_is_cumulative(self) -> None:
        meter = CostMeter()
        meter.add_agent_tokens(1, 1)
        meter.add_agent_tokens(2, 3)
        cv = meter.to_cost_vector()
        assert cv.agent_input_tokens == 3
        assert cv.agent_output_tokens == 4

    def test_record_latency_rejects_unknown_kind(self) -> None:
        meter = CostMeter()
        with pytest.raises(ValueError, match="retrieval"):
            meter.record_latency("bogus", 1.0)

    def test_record_latency_each_kind_targets_distinct_field(self) -> None:
        meter = CostMeter()
        meter.record_latency("retrieval", 1.0)
        meter.record_latency("write", 2.0)
        meter.record_latency("update", 4.0)
        cv = meter.to_cost_vector()
        assert cv.retrieval_latency_ms == 1.0
        assert cv.write_latency_ms == 2.0
        assert cv.update_latency_ms == 4.0


# ---------------------------------------------------------------------------
# Scope attribution via instrumented_llm
# ---------------------------------------------------------------------------


class TestScopeAttribution:
    """instrumented_llm auto-attributes tokens to the active scope."""

    def test_agent_scope_attributes_to_agent(self) -> None:
        meter = CostMeter()
        llm = instrumented_llm(FakeLLM("out token"), meter)  # output = 2 tokens
        with meter.agent_scope():
            llm("in one two three")  # input = 4 tokens
        cv = meter.to_cost_vector()
        assert cv.agent_input_tokens == 4
        assert cv.agent_output_tokens == 2
        assert cv.mem_internal_in_tokens == 0
        assert cv.mem_internal_out_tokens == 0

    def test_memory_scope_attributes_to_mem_internal(self) -> None:
        meter = CostMeter()
        llm = instrumented_llm(FakeLLM("a b c"), meter)  # output = 3 tokens
        with meter.memory_scope():
            llm("extract this fact")  # input = 3 tokens
        cv = meter.to_cost_vector()
        assert cv.mem_internal_in_tokens == 3
        assert cv.mem_internal_out_tokens == 3
        assert cv.agent_input_tokens == 0
        assert cv.agent_output_tokens == 0

    def test_agent_and_memory_do_not_cross_contaminate(self) -> None:
        meter = CostMeter()
        llm = instrumented_llm(FakeLLM("r1 r2"), meter)  # 2 output tokens
        with meter.agent_scope():
            llm("a b c")  # agent in 3
        with meter.memory_scope():
            llm("x y")  # mem in 2
        cv = meter.to_cost_vector()
        assert cv.agent_input_tokens == 3
        assert cv.agent_output_tokens == 2
        assert cv.mem_internal_in_tokens == 2
        assert cv.mem_internal_out_tokens == 2

    def test_reflection_scope_attributes_to_reflection_tokens(self) -> None:
        meter = CostMeter()
        llm = instrumented_llm(FakeLLM("s1 s2"), meter)  # output 2 tokens
        with meter.reflection_scope():
            llm("consolidate these three")  # input 3 tokens
        cv = meter.to_cost_vector()
        # reflection_tokens is a single field: input + output summed.
        assert cv.reflection_tokens == 5
        assert cv.mem_internal_in_tokens == 0
        assert cv.mem_internal_out_tokens == 0
        assert cv.agent_input_tokens == 0

    def test_excluded_scope_is_not_counted_in_vector(self) -> None:
        meter = CostMeter()
        llm = instrumented_llm(FakeLLM("j1 j2"), meter)  # 2 output tokens
        with meter.excluded_scope("judge"):
            llm("score this answer now please")  # 5 input tokens
        cv = meter.to_cost_vector()
        assert cv.total_tokens() == 0
        # but tracked separately (never silently dropped)
        excl_in, excl_out = meter.excluded_totals()
        assert excl_in == 5
        assert excl_out == 2

    def test_record_llm_call_respects_scope(self) -> None:
        # The scope-aware recording method (used by adapters that read usage
        # directly from a backend response) attributes by active scope.
        meter = CostMeter()
        with meter.memory_scope():
            meter.record_llm_call(11, 7)
        cv = meter.to_cost_vector()
        assert cv.mem_internal_in_tokens == 11
        assert cv.mem_internal_out_tokens == 7


# ---------------------------------------------------------------------------
# Embedder attribution
# ---------------------------------------------------------------------------


class TestEmbedderAttribution:
    def test_embedder_counts_tokens_and_one_call_per_invocation(self) -> None:
        meter = CostMeter()
        emb = instrumented_embedder(fake_embedder, meter)
        with meter.memory_scope():
            emb(["alpha beta", "gamma"])  # tokens = 2 + 1 = 3, calls = 1
        cv = meter.to_cost_vector()
        assert cv.embedding_tokens == 3
        assert cv.embedding_calls == 1

    def test_embedder_single_string(self) -> None:
        meter = CostMeter()
        emb = instrumented_embedder(fake_embedder, meter)
        with meter.memory_scope():
            emb("one two three four")  # tokens = 4
        cv = meter.to_cost_vector()
        assert cv.embedding_tokens == 4
        assert cv.embedding_calls == 1

    def test_embedder_excluded_scope_not_counted(self) -> None:
        meter = CostMeter()
        emb = instrumented_embedder(fake_embedder, meter)
        with meter.excluded_scope("dataset_gen"):
            emb(["a b c"])
        cv = meter.to_cost_vector()
        assert cv.embedding_tokens == 0
        assert cv.embedding_calls == 0


# ---------------------------------------------------------------------------
# Scope nesting / restoration
# ---------------------------------------------------------------------------


class TestScopeNesting:
    def test_nested_scopes_restore_outer(self) -> None:
        meter = CostMeter()
        llm = instrumented_llm(FakeLLM("o"), meter)  # output 1 token
        with meter.agent_scope():
            llm("a b")  # agent in 2
            with meter.memory_scope():
                llm("c d e")  # mem in 3
            llm("f")  # back to agent: in 1
        cv = meter.to_cost_vector()
        assert cv.agent_input_tokens == 3  # 2 + 1
        assert cv.agent_output_tokens == 2  # 1 + 1
        assert cv.mem_internal_in_tokens == 3
        assert cv.mem_internal_out_tokens == 1

    def test_scope_cleared_after_exit(self) -> None:
        meter = CostMeter()  # non-strict
        llm = instrumented_llm(FakeLLM("z"), meter)
        with meter.agent_scope():
            llm("a b")
        # outside any scope now -> non-strict routes to unscoped bucket
        with pytest.warns(CostInstrumentationWarning):
            llm("c d e")
        cv = meter.to_cost_vector()
        assert cv.agent_input_tokens == 2  # only the in-scope call


# ---------------------------------------------------------------------------
# Strict mode
# ---------------------------------------------------------------------------


class TestStrictMode:
    def test_strict_raises_on_unscoped_llm_call(self) -> None:
        meter = CostMeter(strict_instrumentation=True)
        llm = instrumented_llm(FakeLLM("x"), meter)
        with pytest.raises(CostInstrumentationError):
            llm("unscoped call here")

    def test_strict_raises_on_unscoped_embedding_call(self) -> None:
        meter = CostMeter(strict_instrumentation=True)
        emb = instrumented_embedder(fake_embedder, meter)
        with pytest.raises(CostInstrumentationError):
            emb("unscoped embed")

    def test_strict_allows_memory_scope(self) -> None:
        meter = CostMeter(strict_instrumentation=True)
        llm = instrumented_llm(FakeLLM("ok"), meter)
        with meter.memory_scope():
            llm("counted now")  # must NOT raise
        cv = meter.to_cost_vector()
        assert cv.mem_internal_in_tokens == 2

    def test_strict_allows_excluded_scope(self) -> None:
        # dataset-gen / judge / render run under excluded scope -> no raise,
        # and not counted as system cost.
        meter = CostMeter(strict_instrumentation=True)
        llm = instrumented_llm(FakeLLM("g"), meter)
        with meter.excluded_scope("dataset_gen"):
            llm("generate episode text")  # must NOT raise
        cv = meter.to_cost_vector()
        assert cv.total_tokens() == 0


class TestNonStrictUnscoped:
    def test_unscoped_call_warns_and_buckets(self) -> None:
        meter = CostMeter()  # non-strict default
        llm = instrumented_llm(FakeLLM("o1 o2"), meter)  # 2 output tokens
        with pytest.warns(CostInstrumentationWarning):
            llm("a b c")  # 3 input tokens
        cv = meter.to_cost_vector()
        assert cv.agent_input_tokens == 0
        assert cv.mem_internal_in_tokens == 0
        # not dropped: lands in the unscoped bucket
        u_in, u_out = meter.unscoped_totals()
        assert u_in == 3
        assert u_out == 2


# ---------------------------------------------------------------------------
# scalarize
# ---------------------------------------------------------------------------


class TestScalarize:
    def test_pure_token_vector_is_exact(self) -> None:
        # No latency/storage -> integer token sum, exactly representable.
        cv = CostVector(
            agent_input_tokens=10,
            agent_output_tokens=5,
            mem_internal_in_tokens=3,
            mem_internal_out_tokens=2,
            embedding_tokens=4,
            reflection_tokens=1,
        )
        result = scalarize(cv, ScalarizationWeights(), ConversionSheet())
        assert result == 25.0  # 10+5+3+2+4+1

    def test_full_vector_matches_hand_computed(self) -> None:
        # Hand-computed tokens-equivalent example:
        #   token_equiv  = 1000 + 500 + 300 + 200 + 400 + 100 = 2500
        #   latency      = 0.1 * (50 + 30 + 20)               = 10.0
        #   storage      = 0.01 * (2048 / 1024)               = 0.02
        #   total                                             = 2510.02
        cv = CostVector(
            agent_input_tokens=1000,
            agent_output_tokens=500,
            mem_internal_in_tokens=300,
            mem_internal_out_tokens=200,
            embedding_tokens=400,
            embedding_calls=10,
            storage_bytes=2048,
            retrieval_latency_ms=50.0,
            write_latency_ms=30.0,
            update_latency_ms=20.0,
            reflection_tokens=100,
            num_retrieval_calls=5,
        )
        weights = ScalarizationWeights()  # all 1.0
        conversion = ConversionSheet(ms_to_token_equiv=0.1, kb_to_token_equiv=0.01)
        result = scalarize(cv, weights, conversion)
        assert result == pytest.approx(2510.02)

    def test_weights_scale_token_fields(self) -> None:
        cv = CostVector(agent_input_tokens=100, agent_output_tokens=100)
        # weight output 2x input
        weights = ScalarizationWeights(agent_input_tokens=1.0, agent_output_tokens=2.0)
        result = scalarize(cv, weights, ConversionSheet())
        assert result == pytest.approx(100 * 1.0 + 100 * 2.0)  # 300.0

    def test_per_call_conversion(self) -> None:
        cv = CostVector(embedding_calls=10, num_retrieval_calls=4)
        conversion = ConversionSheet(
            per_embedding_call_token_equiv=2.0,
            per_retrieval_call_token_equiv=3.0,
        )
        result = scalarize(cv, ScalarizationWeights(), conversion)
        assert result == pytest.approx(10 * 2.0 + 4 * 3.0)  # 32.0

    def test_empty_vector_is_zero(self) -> None:
        assert scalarize(CostVector(), ScalarizationWeights(), ConversionSheet()) == 0.0

    def test_returns_float_not_collapsed_meter(self) -> None:
        # scalarize is the ONLY place the vector is collapsed to a scalar.
        result = scalarize(
            CostVector(agent_input_tokens=1), ScalarizationWeights(), ConversionSheet()
        )
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


class TestConfigLoading:
    def test_config_file_exists(self) -> None:
        assert CONFIG_PATH.is_file()

    def test_load_declared_sheet(self) -> None:
        cfg = load_cost_config(CONFIG_PATH)
        assert cfg.weights.agent_input_tokens == 1.0
        assert cfg.weights.agent_output_tokens == 1.0
        assert cfg.weights.mem_internal_in_tokens == 1.0
        assert cfg.weights.reflection_tokens == 1.0
        # declared conversion sheet (spec §1.3 example)
        assert cfg.conversion.ms_to_token_equiv == 0.1
        assert cfg.conversion.kb_to_token_equiv == 0.01

    def test_loaded_config_reproduces_hand_computed_scalar(self) -> None:
        cfg = load_cost_config(CONFIG_PATH)
        cv = CostVector(
            agent_input_tokens=1000,
            agent_output_tokens=500,
            mem_internal_in_tokens=300,
            mem_internal_out_tokens=200,
            embedding_tokens=400,
            storage_bytes=2048,
            retrieval_latency_ms=50.0,
            write_latency_ms=30.0,
            update_latency_ms=20.0,
            reflection_tokens=100,
        )
        result = scalarize(cv, cfg.weights, cfg.conversion)
        assert result == pytest.approx(2510.02)


# ---------------------------------------------------------------------------
# Vector retained (never collapsed except scalarize)
# ---------------------------------------------------------------------------


class TestVectorRetained:
    def test_to_cost_vector_returns_costvector(self) -> None:
        meter = CostMeter()
        meter.add_agent_tokens(5, 5)
        cv = meter.to_cost_vector()
        assert isinstance(cv, CostVector)

    def test_repeated_to_cost_vector_is_stable(self) -> None:
        meter = CostMeter()
        meter.add_agent_tokens(5, 5)
        assert meter.to_cost_vector() == meter.to_cost_vector()


# ---------------------------------------------------------------------------
# QA scenario: combined agent turn + memory add internal LLM
# ---------------------------------------------------------------------------


class TestQAScenarioAttribution:
    def test_agent_turn_plus_memory_add_internal_llm(self) -> None:
        meter = CostMeter()
        agent_llm = instrumented_llm(FakeLLM("agent replied here"), meter)  # out 3
        mem_llm = instrumented_llm(FakeLLM("mem extracted"), meter)  # out 2

        # agent turn
        with meter.agent_scope():
            agent_llm("user asks a question")  # in 4

        # an add_memory that internally calls an LLM inside memory_scope()
        with meter.memory_scope():
            mem_llm("internal extraction prompt")  # in 3

        cv = meter.to_cost_vector()
        # agent counts only agent tokens
        assert cv.agent_input_tokens == 4
        assert cv.agent_output_tokens == 3
        # mem_internal counts only the memory system's LLM tokens
        assert cv.mem_internal_in_tokens == 3
        assert cv.mem_internal_out_tokens == 2

        # exact tokens-equivalent (all token weights 1.0, no latency/storage)
        cfg_w = ScalarizationWeights()
        cfg_c = ConversionSheet()
        # 4 + 3 + 3 + 2 = 12
        assert scalarize(cv, cfg_w, cfg_c) == 12.0


# ---------------------------------------------------------------------------
# Thread-safety: per-thread scope isolation + locked accumulation
# ---------------------------------------------------------------------------


class TestThreadSafety:
    def test_concurrent_scopes_isolated_per_thread(self) -> None:
        meter = CostMeter()
        n = 50

        def worker(kind: str) -> None:
            llm = instrumented_llm(FakeLLM("o o"), meter)  # output 2 tokens
            ctx = meter.agent_scope() if kind == "agent" else meter.memory_scope()
            with ctx:
                for _ in range(n):
                    llm("a b")  # input 2 tokens

        t_agent = threading.Thread(target=worker, args=("agent",))
        t_mem = threading.Thread(target=worker, args=("memory",))
        t_agent.start()
        t_mem.start()
        t_agent.join()
        t_mem.join()

        cv = meter.to_cost_vector()
        assert cv.agent_input_tokens == n * 2
        assert cv.agent_output_tokens == n * 2
        assert cv.mem_internal_in_tokens == n * 2
        assert cv.mem_internal_out_tokens == n * 2

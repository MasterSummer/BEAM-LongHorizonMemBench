from __future__ import annotations

import pytest

from lhmsb.families.software.mem0_vertical import SoftwareMem0VerticalFamily
from lhmsb.longhorizon.attribution import (
    FactSignature,
    attribute_memory,
    build_software_fact_signatures,
    eligible_write_state_ids,
    normalize_fact_text,
)


def _signature(
    state_id: str,
    *groups: tuple[str, ...],
    negative: tuple[str, ...] = (),
) -> FactSignature:
    return FactSignature(
        state_id=state_id,
        required_anchor_groups=groups,
        allowed_surface_variants=(),
        negative_anchors=negative,
        polarity="positive",
        version=1,
        scope="project",
        authority="owner",
        source_sessions=(0,),
        source_event_ids=(f"event-{state_id}",),
    )


def test_normalization_is_unicode_case_punctuation_and_whitespace_stable() -> None:
    assert normalize_fact_text("  Pipeline—OFFLINE,\nNo Cloud!  ") == (
        "pipeline offline no cloud"
    )


def test_exact_signature_requires_every_anchor_group() -> None:
    signature = _signature(
        "C1",
        ("offline",),
        ("cloud services", "cloud api"),
        negative=("may call cloud", "online execution"),
    )
    result = attribute_memory(
        "m1",
        "Pipeline execution remains OFFLINE; it must not call cloud services.",
        (signature,),
    )
    assert result.method == "exact_signature"
    assert result.state_ids == ("C1",)
    assert result.contributes_positive_coverage


def test_negated_or_contradictory_memory_does_not_match() -> None:
    signature = _signature(
        "C1",
        ("offline",),
        ("cloud services",),
        negative=("may call cloud",),
    )
    result = attribute_memory(
        "m1",
        "The offline pipeline may call cloud services for speed.",
        (signature,),
    )
    assert result.method == "no_match"
    assert result.state_ids == ()
    assert not result.contributes_positive_coverage


def test_allowed_surface_variant_can_encode_the_complete_fact() -> None:
    signature = FactSignature(
        state_id="C2",
        required_anchor_groups=(("held out",), ("never modified",)),
        allowed_surface_variants=("the evaluation split is frozen",),
        negative_anchors=("the evaluation split may change",),
        polarity="positive",
        version=1,
        scope="tests",
        authority="project-owner",
        source_sessions=(0,),
        source_event_ids=("e-02-heldout",),
    )
    result = attribute_memory(
        "m1",
        "For this project, the evaluation split is frozen.",
        (signature,),
    )
    assert result.method == "exact_signature"
    assert result.state_ids == ("C2",)


def test_multiple_complete_matches_are_resolved_as_multi_signature() -> None:
    first = _signature("P1", ("branch",), ("v1",))
    second = _signature("U1", ("branch",), ("v1",))
    result = attribute_memory("m1", "The branch is v1.", (first, second))
    assert result.method == "multi_signature"
    assert result.state_ids == ("P1", "U1")
    assert result.contributes_positive_coverage


def test_multiple_partial_matches_remain_ambiguous() -> None:
    first = _signature("P1", ("branch",), ("v1",))
    second = _signature("U1", ("branch",), ("leakage",))
    result = attribute_memory("m1", "The branch was discussed.", (first, second))
    assert result.method == "ambiguous"
    assert result.state_ids == ("P1", "U1")
    assert not result.contributes_positive_coverage


def test_unique_provenance_can_attribute_a_partial_but_uncontested_memory() -> None:
    signature = _signature("U1", ("data leakage",), ("v1",))
    result = attribute_memory(
        "m1",
        "A data leakage issue was found.",
        (signature,),
        unique_write_state_ids=("U1",),
    )
    assert result.method == "unique_provenance"
    assert result.state_ids == ("U1",)
    assert result.contributes_positive_coverage


def test_source_session_provenance_disambiguates_overlapping_partial_matches() -> None:
    signatures = (
        _signature("P2", ("v2",), ("current branch",)),
        _signature("V2", ("v2",), ("integrity audit",)),
    )
    result = attribute_memory(
        "m1",
        'Opened results/session_6.json: {"branch": "v2"}',
        signatures,
        unique_write_state_ids=("P2",),
    )
    assert result.method == "unique_provenance"
    assert result.state_ids == ("P2",)
    assert result.contributes_positive_coverage


def test_unique_provenance_rejects_multiple_eligible_states() -> None:
    signatures = (
        _signature("U1", ("data leakage",), ("v1",)),
        _signature("P2", ("data leakage",), ("v2",)),
    )
    result = attribute_memory(
        "m1",
        "Data leakage was found.",
        signatures,
        unique_write_state_ids=("U1", "P2"),
    )
    assert result.method == "ambiguous"
    assert not result.contributes_positive_coverage


def test_signature_rejects_invalid_state_predicates() -> None:
    with pytest.raises(ValueError, match="polarity"):
        FactSignature(
            state_id="C1",
            required_anchor_groups=(("offline",),),
            allowed_surface_variants=(),
            negative_anchors=(),
            polarity="unknown",  # type: ignore[arg-type]
            version=1,
            scope="all-code",
            authority="project-owner",
            source_sessions=(0,),
            source_event_ids=("e-01-offline",),
        )
    with pytest.raises(ValueError, match="version"):
        FactSignature(
            state_id="C1",
            required_anchor_groups=(("offline",),),
            allowed_surface_variants=(),
            negative_anchors=(),
            polarity="positive",
            version=0,
            scope="all-code",
            authority="project-owner",
            source_sessions=(0,),
            source_event_ids=("e-01-offline",),
        )


def test_software_signature_catalog_covers_every_latent_state_with_provenance() -> None:
    spec = SoftwareMem0VerticalFamily.generate(42, n_sessions=16)
    signatures = build_software_fact_signatures(spec.plan)
    assert {item.state_id for item in signatures} == {
        state.state_id for state in spec.plan.state_units
    }
    c1 = next(item for item in signatures if item.state_id == "C1")
    assert "offline" in c1.required_anchor_groups[0]
    assert c1.scope == "all-code"
    assert c1.authority == "project-owner"
    assert c1.source_sessions == (0,)
    assert c1.source_event_ids == ("e-01-offline",)
    p2 = next(item for item in signatures if item.state_id == "P2")
    assert p2.version == 1
    assert p2.scope == "pipeline"
    assert p2.authority == "engineering-lead"


def test_generated_fact_surfaces_cover_every_semantic_scenario() -> None:
    for seed in range(42, 47):
        spec = SoftwareMem0VerticalFamily.generate(seed, n_sessions=16)
        signatures = build_software_fact_signatures(spec.plan)
        for state in spec.plan.state_units:
            value = state.value
            if isinstance(value, dict) and isinstance(value.get("text"), str):
                surface = value["text"]
            elif isinstance(value, dict):
                surface = f"branch {value['branch']} status {value['status']}"
            else:
                surface = str(value)
            result = attribute_memory(f"m-{state.state_id}", surface, signatures)
            assert result.state_ids == (state.state_id,), (seed, state.state_id, result)
            assert result.method == "exact_signature"
            assert result.contributes_positive_coverage


@pytest.mark.parametrize(
    ("state_id", "text"),
    (
        (
            "G0",
            "User is building a deterministic and traceable benchmark execution service "
            "as a software project.",
        ),
        (
            "C1",
            "For the benchmark service, scored benchmark runs must not use remote "
            "endpoints and must keep evaluation execution locally isolated.",
        ),
        (
            "C2",
            "The sealed scoring fixtures for the benchmark service must never be altered.",
        ),
        (
            "L1",
            "The benchmark owner authorized a remote accelerator exclusively for the "
            "isolated latency profiler; scored runs remain locally isolated.",
        ),
        (
            "P2",
            "The benchmark service now has a v2 branch, the current runner after "
            "completing scoring-isolation repair.",
        ),
    ),
)
def test_generated_lexical_signatures_cover_writer_paraphrases(
    state_id: str,
    text: str,
) -> None:
    spec = SoftwareMem0VerticalFamily.generate(45, n_sessions=16)
    result = attribute_memory(
        f"m-{state_id}",
        text,
        build_software_fact_signatures(spec.plan),
    )
    assert result.state_ids == (state_id,)
    assert result.method in {"exact_signature", "lexical_signature"}
    assert result.contributes_positive_coverage


@pytest.mark.parametrize(
    ("state_id", "text"),
    (
        ("G0", "用户的目标是构建一个确定性和可追溯的基准执行服务。"),
        (
            "C1",
            "评分的基准运行不得使用远程端点，评估执行必须保持本地隔离。",
        ),
        ("C2", "用户强调密封的评分夹具绝不能被修改。"),
        ("P1", "项目当前分支为v1，状态为初始实现。"),
        (
            "L1",
            "基准测试所有者仅授权远程加速器用于隔离的延迟分析器。",
        ),
        ("V2", "v2运行器成功通过了密封夹具完整性审计。"),
    ),
)
def test_programmatic_signatures_cover_chinese_native_memory(
    state_id: str,
    text: str,
) -> None:
    spec = SoftwareMem0VerticalFamily.generate(42, n_sessions=16)
    result = attribute_memory(
        f"m-{state_id}",
        text,
        build_software_fact_signatures(spec.plan),
    )
    assert result.state_ids == (state_id,)
    assert result.method == "exact_signature"
    assert result.contributes_positive_coverage


def test_supported_chinese_non_fact_is_no_match() -> None:
    spec = SoftwareMem0VerticalFamily.generate(42, n_sessions=16)
    result = attribute_memory(
        "m-note",
        "用户打开了一个文件并继续处理软件项目。",
        build_software_fact_signatures(spec.plan),
    )
    assert result.method == "no_match"
    assert result.state_ids == ()


def test_write_eligibility_includes_current_updates_and_excludes_retirements() -> None:
    spec = SoftwareMem0VerticalFamily.generate(42, n_sessions=16)

    assert eligible_write_state_ids(spec.plan, 0) == ("C1", "C2", "G0", "P1")
    assert eligible_write_state_ids(spec.plan, 5) == ("P1",)
    assert eligible_write_state_ids(spec.plan, 6) == ("P2",)
    assert eligible_write_state_ids(spec.plan, 8) == ("D1",)
    assert eligible_write_state_ids(spec.plan, 9) == ("L1", "V2")

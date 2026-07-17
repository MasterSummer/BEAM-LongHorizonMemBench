from __future__ import annotations

import hashlib
import json

import pytest

from lhmsb.families.software.mem0_vertical import SoftwareMem0VerticalFamily
from lhmsb.qualification.context import (
    DEFAULT_FULL_CONTEXT_MAX_CHARS,
    FullContextLimitError,
    build_public_history_units,
    full_context_hash,
    render_full_context,
)


def test_public_history_is_one_unchanged_unit_per_observation_or_tool_result() -> None:
    spec = SoftwareMem0VerticalFamily.generate(42)
    units = build_public_history_units(spec, checkpoint_session=1)
    transcript = json.loads(spec.write_transcript(0))

    assert [unit.content for unit in units] == [
        *transcript["observations"],
        *transcript["tool_results"],
    ]
    assert [
        (unit.source_session, unit.source_kind, unit.source_ordinal)
        for unit in units
    ] == [
        *((0, "observation", ordinal) for ordinal in range(5)),
        *((0, "tool_result", ordinal) for ordinal in range(2)),
    ]
    assert all(
        unit.content_sha256
        == hashlib.sha256(unit.content.encode("utf-8")).hexdigest()
        for unit in units
    )
    assert len({unit.unit_id for unit in units}) == len(units)
    assert all(len(unit.unit_id) == 64 for unit in units)


def test_public_history_ids_and_order_are_deterministic_and_episode_scoped() -> None:
    first = SoftwareMem0VerticalFamily.generate(42)
    repeated = SoftwareMem0VerticalFamily.generate(42)
    other_episode = SoftwareMem0VerticalFamily.generate(43)

    first_units = build_public_history_units(first)
    assert first_units == build_public_history_units(repeated)
    assert tuple(unit.unit_id for unit in first_units) != tuple(
        unit.unit_id for unit in build_public_history_units(other_episode)
    )


def test_checkpoint_history_excludes_current_future_and_evaluator_information() -> None:
    spec = SoftwareMem0VerticalFamily.generate(42)
    checkpoint = 3
    units = build_public_history_units(spec, checkpoint_session=checkpoint)
    combined = "\n".join(unit.content for unit in units)

    assert {unit.source_session for unit in units} == set(range(checkpoint))
    assert "Data leakage was found" not in combined
    assert "selected_action" not in combined
    assert "checker" not in combined
    assert "stale-state" not in combined
    assert all(state.state_id not in combined for state in spec.plan.state_units)
    assert all(action.action_id not in combined for action in spec.actions)


def test_full_context_at_each_sceu_contains_only_prior_public_units() -> None:
    spec = SoftwareMem0VerticalFamily.generate(42)
    all_units = build_public_history_units(spec)

    for continuation in spec.public_continuations:
        checkpoint = continuation.checkpoint_session
        rendered = render_full_context(
            all_units,
            checkpoint_session=checkpoint,
        )
        payload = json.loads(rendered)
        history = payload["public_history"]

        assert sorted({item["session_index"] for item in history}) == list(
            range(checkpoint)
        )
        assert all(item["session_index"] < checkpoint for item in history)
        provenance = [
            (item["session_index"], item["kind"], item["ordinal"])
            for item in history
        ]
        assert provenance == sorted(
            provenance,
            key=lambda item: (
                item[0],
                0 if item[1] == "observation" else 1,
                item[2],
            ),
        )
        assert rendered == render_full_context(
            tuple(reversed(all_units)),
            checkpoint_session=checkpoint,
        )


def test_current_surface_is_not_duplicated_in_full_context() -> None:
    spec = SoftwareMem0VerticalFamily.generate(42)
    checkpoint = 4
    rendered = render_full_context(
        build_public_history_units(spec),
        checkpoint_session=checkpoint,
    )
    payload = json.loads(rendered)

    assert all(
        item["session_index"] != checkpoint for item in payload["public_history"]
    )
    assert "Data leakage was found" not in rendered


def test_full_context_never_truncates_and_hashes_exact_rendering() -> None:
    spec = SoftwareMem0VerticalFamily.generate(42)
    units = build_public_history_units(spec)
    rendered = render_full_context(
        units,
        checkpoint_session=spec.plan.n_sessions,
    )

    assert DEFAULT_FULL_CONTEXT_MAX_CHARS == 100_000
    assert full_context_hash(rendered) == hashlib.sha256(
        rendered.encode("utf-8")
    ).hexdigest()
    with pytest.raises(FullContextLimitError) as exc_info:
        render_full_context(
            units,
            checkpoint_session=spec.plan.n_sessions,
            full_context_max_chars=len(rendered) - 1,
        )
    assert exc_info.value.rendered_chars == len(rendered)
    assert exc_info.value.limit_chars == len(rendered) - 1
    assert "truncate" not in str(exc_info.value).lower()


@pytest.mark.parametrize("checkpoint", (-1, 17))
def test_invalid_checkpoint_is_rejected(checkpoint: int) -> None:
    spec = SoftwareMem0VerticalFamily.generate(42)

    with pytest.raises(ValueError, match="checkpoint_session"):
        build_public_history_units(spec, checkpoint_session=checkpoint)

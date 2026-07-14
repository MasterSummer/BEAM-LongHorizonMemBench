"""LangGraph agent harness for LongHorizonMemSysBench (task 9).

Public API:
  - :func:`run_episode` / :func:`run_episode_traced` — execute one episode under a
    chosen memory condition; the traced variant also returns the transcript.
  - :class:`HarnessRuntime` + :func:`build_agent_graph` — the compiled
    perceive->decide->act->memory_io graph (built WITHOUT any checkpointer/store).
  - :func:`load_agent_model` — the (offline) agent-model loader hook.
  - :func:`plan_steps` / :class:`Step` — deterministic episode decomposition.
  - :class:`Transcript` / :class:`TranscriptEntry` / :class:`EpisodeRun` — the
    deterministic transcript + content hash.
  - :func:`check_cross_session_leakage` / :class:`StatelessnessReport` — the
    no-memory control statelessness proof.
"""

from __future__ import annotations

from lhmsb.harness.agent import (
    AgentModel,
    AgentState,
    Clock,
    HarnessConfigurationError,
    HarnessRuntime,
    PaperSearch,
    build_agent_graph,
    load_agent_model,
    run_episode,
    run_episode_traced,
)
from lhmsb.harness.control import StatelessnessReport, check_cross_session_leakage
from lhmsb.harness.sessions import Step, plan_steps
from lhmsb.harness.transcript import EpisodeRun, Transcript, TranscriptEntry

__all__ = [
    "AgentModel",
    "AgentState",
    "Clock",
    "EpisodeRun",
    "HarnessConfigurationError",
    "HarnessRuntime",
    "PaperSearch",
    "StatelessnessReport",
    "Step",
    "Transcript",
    "TranscriptEntry",
    "build_agent_graph",
    "check_cross_session_leakage",
    "load_agent_model",
    "plan_steps",
    "run_episode",
    "run_episode_traced",
]

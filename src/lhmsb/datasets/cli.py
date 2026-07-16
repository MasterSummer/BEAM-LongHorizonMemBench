"""``python -m lhmsb.datasets`` command-line interface.

Subcommands implement the spec/04-datasets.md §5 lifecycle:

  * ``generate`` — build + validate episodes into a staging dir.
  * ``freeze``   — seal a staging dir into a versioned, checksummed dataset.
  * ``verify``   — recompute checksums and assert they match the manifest.
  * ``regen-check`` — regenerate from stored seeds and assert identical hashes.

Each subcommand returns a process exit code (0 = success). Validation,
integrity, and reproducibility failures return non-zero with a concise report,
so the CLI is safe to gate CI on.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from lhmsb.datasets.pipeline import (
    DatasetError,
    RegenReport,
    VerifyReport,
    freeze_dataset,
    generate_to_staging,
    import_wide_research_to_staging,
    regen_check,
    verify_dataset,
)
from lhmsb.datasets.stateful_pipeline import (
    StatefulDatasetError,
    StatefulRegenReport,
    StatefulVerifyReport,
    freeze_stateful,
    generate_stateful_to_staging,
    regen_check_stateful,
    verify_stateful,
)
from lhmsb.families.research import (
    ArxivSearch,
    LocalArxivSearch,
    OpenAlexSearch,
    WideTraceError,
    attach_wide_traces,
    build_arxiv_metadata_index,
    export_wide_questions,
    generate_wide_traces,
)
from lhmsb.families.research.leakage import RealEntityLeakError
from lhmsb.sim.core import RenderValidationError, ScheduleError

_PROG = "python -m lhmsb.datasets"

# Generation failures that should surface as a clean exit code (not a traceback).
_GENERATION_ERRORS = (
    DatasetError,
    StatefulDatasetError,
    RenderValidationError,
    RealEntityLeakError,
    ScheduleError,
    ValueError,
    WideTraceError,
)


def _parse_scale(tokens: Sequence[str] | None) -> dict[str, int]:
    """Parse ``key=value`` scale tokens into an int dict (raises ValueError on bad input)."""
    overrides: dict[str, int] = {}
    for token in tokens or []:
        if "=" not in token:
            raise ValueError(f"--scale expects key=value tokens, got {token!r}")
        key, _, value = token.partition("=")
        key = key.strip()
        try:
            overrides[key] = int(value)
        except ValueError as exc:
            raise ValueError(f"--scale value for {key!r} must be an int, got {value!r}") from exc
    return overrides


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=_PROG,
        description="Generate, freeze, verify, and regenerate LongHorizonMemSysBench datasets.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="build + validate episodes into a staging dir")
    gen.add_argument("--family", required=True, choices=["research", "software"])
    gen.add_argument("--seeds", required=True, nargs="+", type=int, help="base seeds")
    gen.add_argument("--n-episodes", type=int, default=1, help="episodes per seed (>=1)")
    gen.add_argument(
        "--scale", nargs="*", default=None, metavar="KEY=VALUE", help="scale overrides"
    )
    gen.add_argument("--out", required=True, type=Path, help="staging output dir")

    wide = sub.add_parser(
        "import-wide", help="import AutoResearchBench Wide Research JSONL into staging"
    )
    wide.add_argument("--input", required=True, type=Path, help="decrypted Wide Research JSONL")
    wide.add_argument("--seed", type=int, default=0, help="replay seed recorded on episodes")
    wide.add_argument("--limit", type=int, default=None, help="optional record limit")
    wide.add_argument("--out", required=True, type=Path, help="staging output dir")

    wide_questions = sub.add_parser(
        "wide-questions", help="export a gold-free question artifact from official Wide data"
    )
    wide_questions.add_argument("--input", required=True, type=Path)
    wide_questions.add_argument("--limit", type=int, default=None)
    wide_questions.add_argument("--out", required=True, type=Path)

    wide_traces = sub.add_parser(
        "wide-traces", help="generate frozen question-only research traces"
    )
    wide_traces.add_argument("--questions", required=True, type=Path)
    wide_traces.add_argument("--sessions", type=int, default=3)
    wide_traces.add_argument("--top-k", type=int, default=10)
    wide_traces.add_argument("--max-workers", type=int, default=8)
    wide_traces.add_argument(
        "--search-backend", choices=["local", "arxiv", "openalex"], default="local"
    )
    wide_traces.add_argument(
        "--index", type=Path, default=None, help="SQLite FTS5 index for --search-backend local"
    )
    wide_traces.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="raw request cache (defaults to <out>/.request-cache for arXiv)",
    )
    wide_traces.add_argument(
        "--arxiv-min-interval",
        type=float,
        default=3.0,
        help="minimum seconds between uncached arXiv API requests",
    )
    wide_traces.add_argument(
        "--arxiv-candidates",
        type=int,
        default=100,
        help="remote candidates fetched once per question before local session reranking",
    )
    wide_traces.add_argument("--out", required=True, type=Path)

    attach_traces = sub.add_parser(
        "attach-wide-traces", help="join frozen traces with evaluator gold after generation"
    )
    attach_traces.add_argument("--input", required=True, type=Path)
    attach_traces.add_argument("--traces", required=True, type=Path)
    attach_traces.add_argument("--limit", type=int, default=None)
    attach_traces.add_argument(
        "--min-gold-observed",
        type=int,
        default=0,
        help="post-freeze qualification threshold (0 keeps every trace)",
    )
    attach_traces.add_argument("--out", required=True, type=Path)

    metadata_index = sub.add_parser(
        "build-arxiv-index", help="build a frozen SQLite FTS5 arXiv metadata index"
    )
    metadata_index.add_argument("--input", required=True, type=Path, nargs="+")
    metadata_index.add_argument("--out", required=True, type=Path)

    frz = sub.add_parser("freeze", help="seal a staging dir into a checksummed dataset")
    frz.add_argument("--src", required=True, type=Path, help="staging dir from `generate`")
    frz.add_argument("--out", required=True, type=Path, help="frozen dataset dir")

    ver = sub.add_parser("verify", help="recompute checksums and assert manifest match")
    ver.add_argument("--frozen", required=True, type=Path, help="frozen dataset dir")

    regen = sub.add_parser("regen-check", help="regenerate from seeds and assert identical hashes")
    regen.add_argument("--frozen", required=True, type=Path, help="frozen dataset dir")

    stateful_gen = sub.add_parser(
        "generate-stateful", help="build the offline state-first Software vertical staging tree"
    )
    stateful_gen.add_argument("--family", required=True, choices=["software"])
    stateful_gen.add_argument("--seeds", required=True, nargs="+", type=int)
    stateful_gen.add_argument("--n-episodes", type=int, default=1)
    stateful_gen.add_argument("--n-sessions", type=int, default=16)
    stateful_gen.add_argument("--out", required=True, type=Path)

    stateful_freeze = sub.add_parser(
        "freeze-stateful", help="seal a state-first staging tree with checksums"
    )
    stateful_freeze.add_argument("--src", required=True, type=Path)
    stateful_freeze.add_argument("--out", required=True, type=Path)

    stateful_verify = sub.add_parser(
        "verify-stateful", help="verify checksums in a state-first frozen dataset"
    )
    stateful_verify.add_argument("--frozen", required=True, type=Path)

    stateful_regen = sub.add_parser(
        "regen-check-stateful", help="regenerate a state-first dataset from stored seeds"
    )
    stateful_regen.add_argument("--frozen", required=True, type=Path)
    return parser


def _cmd_generate(args: argparse.Namespace) -> int:
    try:
        overrides = _parse_scale(args.scale)
        episodes = generate_to_staging(
            args.out,
            family=args.family,
            seeds=args.seeds,
            n_episodes=args.n_episodes,
            scale_overrides=overrides,
        )
    except _GENERATION_ERRORS as exc:
        print(f"generate FAILED: {type(exc).__name__}: {exc}")
        return 1
    print(f"generated {len(episodes)} episode(s) [{args.family}] -> {args.out}")
    return 0


def _cmd_import_wide(args: argparse.Namespace) -> int:
    try:
        episodes = import_wide_research_to_staging(
            args.input, args.out, seed=args.seed, limit=args.limit
        )
    except _GENERATION_ERRORS as exc:
        print(f"import-wide FAILED: {type(exc).__name__}: {exc}")
        return 1
    print(f"imported {len(episodes)} Wide Research episode(s) -> {args.out}")
    return 0


def _cmd_wide_questions(args: argparse.Namespace) -> int:
    try:
        manifest = export_wide_questions(args.input, args.out, limit=args.limit)
    except _GENERATION_ERRORS as exc:
        print(f"wide-questions FAILED: {type(exc).__name__}: {exc}")
        return 1
    print(
        f"exported {manifest.record_count} gold-free Wide question(s) -> {args.out} "
        f"(sha256={manifest.questions_sha256})"
    )
    return 0


def _cmd_wide_traces(args: argparse.Namespace) -> int:
    try:
        if args.search_backend == "local":
            if args.index is None:
                raise WideTraceError("--index is required for --search-backend local")
            search = LocalArxivSearch(args.index)
        elif args.search_backend == "arxiv":
            cache_dir = args.cache_dir or args.out / ".request-cache"
            search = ArxivSearch(
                cache_dir=cache_dir,
                min_interval_seconds=args.arxiv_min_interval,
                session_candidate_pool=args.arxiv_candidates,
            )
        else:
            search = OpenAlexSearch()
        manifest = generate_wide_traces(
            args.questions,
            args.out,
            search=search,
            session_count=args.sessions,
            top_k=args.top_k,
            max_workers=args.max_workers,
        )
    except _GENERATION_ERRORS as exc:
        print(f"wide-traces FAILED: {type(exc).__name__}: {exc}")
        return 1
    print(
        f"generated {manifest.record_count} leakage-audited Wide trace(s) -> {args.out} "
        f"(sha256={manifest.traces_sha256})"
    )
    return 0


def _cmd_build_arxiv_index(args: argparse.Namespace) -> int:
    try:
        manifest = build_arxiv_metadata_index(args.input, args.out)
    except _GENERATION_ERRORS as exc:
        print(f"build-arxiv-index FAILED: {type(exc).__name__}: {exc}")
        return 1
    print(
        f"indexed {manifest.record_count} arXiv metadata record(s) -> {args.out} "
        f"(sha256={manifest.index_sha256})"
    )
    return 0


def _cmd_attach_wide_traces(args: argparse.Namespace) -> int:
    try:
        audit = attach_wide_traces(
            args.input,
            args.traces,
            args.out,
            limit=args.limit,
            min_gold_observed=args.min_gold_observed,
        )
    except _GENERATION_ERRORS as exc:
        print(f"attach-wide-traces FAILED: {type(exc).__name__}: {exc}")
        return 1
    print(
        f"attached {audit.record_count}/{audit.source_record_count} Wide trace(s) -> "
        f"{args.out}; observed gold coverage={audit.gold_coverage:.4f}"
    )
    return 0


def _cmd_freeze(args: argparse.Namespace) -> int:
    try:
        manifest = freeze_dataset(args.src, args.out)
    except _GENERATION_ERRORS as exc:
        print(f"freeze FAILED: {type(exc).__name__}: {exc}")
        return 1
    print(
        f"froze {len(manifest.episodes)} episode(s) -> {args.out} "
        f"({len(manifest.files)} files checksummed)"
    )
    return 0


def _report_verify(report: VerifyReport, frozen: Path) -> int:
    if report.ok:
        print(f"verify OK: {report.n_checked} file(s) match the manifest in {frozen}")
        return 0
    print(f"verify FAILED for {frozen}:")
    for rel, want, got in report.mismatches:
        print(f"  checksum mismatch: {rel}\n    expected {want}\n    actual   {got}")
    for rel in report.missing:
        print(f"  missing file: {rel}")
    return 1


def _cmd_verify(args: argparse.Namespace) -> int:
    try:
        report = verify_dataset(args.frozen)
    except DatasetError as exc:
        print(f"verify FAILED: {exc}")
        return 1
    return _report_verify(report, args.frozen)


def _report_regen(report: RegenReport, frozen: Path) -> int:
    if report.ok:
        detail = (
            f"; {report.skipped} external episode(s) checked by verify only"
            if report.skipped
            else ""
        )
        print(
            f"regen-check OK: {report.checked} episode(s) regenerated to identical "
            f"world_event_hash + episode_hash in {frozen}{detail}"
        )
        return 0
    print(f"regen-check FAILED for {frozen}:")
    for episode_id, reason in report.mismatches:
        print(f"  {episode_id}: {reason}")
    return 1


def _cmd_regen_check(args: argparse.Namespace) -> int:
    try:
        report = regen_check(args.frozen)
    except _GENERATION_ERRORS as exc:
        print(f"regen-check FAILED: {type(exc).__name__}: {exc}")
        return 1
    return _report_regen(report, args.frozen)


def _cmd_generate_stateful(args: argparse.Namespace) -> int:
    try:
        generated = generate_stateful_to_staging(
            args.out,
            family=args.family,
            seeds=args.seeds,
            n_episodes=args.n_episodes,
            n_sessions=args.n_sessions,
        )
    except _GENERATION_ERRORS as exc:
        print(f"generate-stateful FAILED: {type(exc).__name__}: {exc}")
        return 1
    print(f"generated {len(generated)} stateful episode(s) -> {args.out}")
    return 0


def _cmd_freeze_stateful(args: argparse.Namespace) -> int:
    try:
        manifest = freeze_stateful(args.src, args.out)
    except _GENERATION_ERRORS as exc:
        print(f"freeze-stateful FAILED: {type(exc).__name__}: {exc}")
        return 1
    print(
        f"froze {manifest.n_episodes} stateful episode(s) -> {args.out} "
        f"({len(manifest.files)} files checksummed)"
    )
    return 0


def _report_stateful_verify(report: StatefulVerifyReport, frozen: Path) -> int:
    if report.ok:
        print(f"verify-stateful OK: {report.n_checked} file(s) match {frozen}")
        return 0
    print(f"verify-stateful FAILED for {frozen}:")
    for relative, expected, actual in report.mismatches:
        print(f"  checksum mismatch: {relative}\n    expected {expected}\n    actual   {actual}")
    for relative in report.missing:
        print(f"  missing file: {relative}")
    return 1


def _cmd_verify_stateful(args: argparse.Namespace) -> int:
    try:
        report = verify_stateful(args.frozen)
    except StatefulDatasetError as exc:
        print(f"verify-stateful FAILED: {exc}")
        return 1
    return _report_stateful_verify(report, args.frozen)


def _report_stateful_regen(report: StatefulRegenReport, frozen: Path) -> int:
    if report.ok:
        print(f"regen-check-stateful OK: {report.checked} episode(s) regenerated in {frozen}")
        return 0
    print(f"regen-check-stateful FAILED for {frozen}:")
    for episode_id, reason in report.mismatches:
        print(f"  {episode_id}: {reason}")
    return 1


def _cmd_regen_check_stateful(args: argparse.Namespace) -> int:
    try:
        report = regen_check_stateful(args.frozen)
    except (StatefulDatasetError, KeyError, TypeError, ValueError) as exc:
        print(f"regen-check-stateful FAILED: {type(exc).__name__}: {exc}")
        return 1
    return _report_stateful_regen(report, args.frozen)


_DISPATCH = {
    "generate": _cmd_generate,
    "import-wide": _cmd_import_wide,
    "wide-questions": _cmd_wide_questions,
    "wide-traces": _cmd_wide_traces,
    "attach-wide-traces": _cmd_attach_wide_traces,
    "build-arxiv-index": _cmd_build_arxiv_index,
    "freeze": _cmd_freeze,
    "verify": _cmd_verify,
    "regen-check": _cmd_regen_check,
    "generate-stateful": _cmd_generate_stateful,
    "freeze-stateful": _cmd_freeze_stateful,
    "verify-stateful": _cmd_verify_stateful,
    "regen-check-stateful": _cmd_regen_check_stateful,
}


def main(argv: Sequence[str] | None = None) -> int:
    """Parse ``argv`` and dispatch to the matching subcommand handler."""
    args = _build_parser().parse_args(argv)
    return _DISPATCH[args.command](args)

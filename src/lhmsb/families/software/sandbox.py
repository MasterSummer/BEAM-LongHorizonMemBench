"""Offline, resource-bounded pytest sandbox for the Software-Dev family.

``run_tests_sandboxed`` runs a hidden pytest suite against agent-written code in
an isolated subprocess with hard limits (spec/04-datasets.md §2.1):

  - **network**: denied. A generated ``conftest.py`` monkeypatches ``socket`` so
    no outbound connection (raw socket, ``urllib``, ``pip install``) can be made.
    Root-free and deterministic — no real packets are ever attempted.
  - **no install**: ``PATH`` is cleared (no ``pip``/``git``/``curl`` binary) and
    ``PIP_NO_INDEX`` is set, so a package install cannot reach an index.
  - **time**: ``RLIMIT_CPU`` + a wall-clock ``timeout`` (default 30s).
  - **memory**: ``RLIMIT_AS`` (default 256 MB) on Linux.  macOS does not
    permit lowering ``RLIMIT_AS`` from an inherited unlimited limit in a
    ``preexec_fn``; the portable fallback keeps CPU/file/network limits and
    relies on the process boundary.
  - **output**: captured stdout/stderr truncated to ``output_limit_bytes`` (10 KB).
  - **file size**: ``RLIMIT_FSIZE`` caps any file the suite writes.

Results are parsed from a JUnit XML report (pytest core, no plugin) so per-test
pass/fail is deterministic for deterministic code (same patch -> same result).
``resource`` is Linux-only, which is the supported platform.
"""

from __future__ import annotations

import os
import resource
import site
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

# Default caps (spec/04-datasets.md §2.1).
DEFAULT_TIME_LIMIT_S = 30
DEFAULT_MEM_LIMIT_MB = 256
DEFAULT_OUTPUT_LIMIT_BYTES = 10240
_FSIZE_LIMIT_BYTES = 50 * 1024 * 1024  # generous: junit/report writes, not user output
_JUNIT_NAME = ".lhmsb_junit.xml"

# Injected into the sandbox dir; blocks all outbound network at pytest import time.
_NETWORK_BLOCK_CONFTEST = '''\
"""lhmsb sandbox guard: deny all network access (injected, do not edit)."""

import socket


def _deny(*args, **kwargs):
    raise OSError("network disabled in lhmsb sandbox")


class _GuardedSocket(socket.socket):
    def connect(self, *args, **kwargs):
        _deny()

    def connect_ex(self, *args, **kwargs):
        _deny()


socket.socket = _GuardedSocket
socket.create_connection = _deny
socket.getaddrinfo = _deny
'''


@dataclass(frozen=True)
class TestResult:
    """Parsed outcome of a sandboxed pytest run.

    ``duration_s`` is diagnostic only and MUST NOT feed scoring (it would break
    determinism). ``all_passed`` requires a non-empty suite that fully passed.
    """

    __test__ = False  # not a pytest test class (name starts with "Test")

    returncode: int
    passed: int
    failed: int
    errors: int
    skipped: int
    total: int
    failed_tests: list[str] = field(default_factory=list)
    timed_out: bool = False
    stdout: str = ""
    stderr: str = ""
    duration_s: float = 0.0

    @property
    def all_passed(self) -> bool:
        """True iff the suite ran at least one test and none failed or errored."""
        return (
            self.returncode == 0
            and self.failed == 0
            and self.errors == 0
            and not self.timed_out
            and self.total > 0
        )


def _coerce_text(value: object) -> str:
    """Coerce a subprocess stdout/stderr payload (str | bytes | None) to str."""
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    if isinstance(value, str):
        return value
    return ""


def _make_limit_setter(mem_limit_mb: int, time_limit_s: int) -> Callable[[], None]:
    """Build a portable ``preexec_fn`` for the forked sandbox child."""

    def _set_limits() -> None:
        cpu = max(1, time_limit_s)
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
        # On Darwin, lowering RLIMIT_AS from an inherited unlimited hard limit
        # raises ``ValueError: current limit exceeds maximum limit``.  Skipping
        # only this limit keeps the sandbox usable on the development platform;
        # Linux CI still receives the hard address-space cap.
        if sys.platform.startswith("linux"):
            mem = mem_limit_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
        resource.setrlimit(resource.RLIMIT_FSIZE, (_FSIZE_LIMIT_BYTES, _FSIZE_LIMIT_BYTES))

    return _set_limits


def _sandbox_env(work_dir: str) -> dict[str, str]:
    """A scrubbed env: no PATH (no install binaries), deterministic, offline."""
    env = dict(os.environ)
    # Drop anything that would inject coverage/addopts into the child pytest, plus any
    # proxy config (defence-in-depth: no route to a network even if a socket slipped through).
    for key in list(env):
        upper = key.upper()
        if upper.startswith(("PYTEST", "COV", "COVERAGE")) or upper.endswith("_PROXY"):
            env.pop(key, None)
    env["PATH"] = ""  # no pip/git/curl resolvable -> install attempts cannot exec
    env["NO_PROXY"] = "*"
    env["PYTHONHASHSEED"] = "0"  # determinism
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PIP_NO_INDEX"] = "1"
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    env["HOME"] = work_dir
    env["TMPDIR"] = work_dir
    env.pop("PYTHONPATH", None)
    # The sandbox changes HOME to the temporary package directory.  On macOS
    # pytest is commonly installed in the user's site-packages, which would
    # otherwise disappear with that HOME change.  Re-add only interpreter
    # installation directories (never the caller's project PYTHONPATH).
    package_paths = [*site.getsitepackages(), site.getusersitepackages()]
    env["PYTHONPATH"] = os.pathsep.join(path for path in package_paths if Path(path).is_dir())
    return env


def _truncate(text: str, limit_bytes: int) -> str:
    """Truncate ``text`` to at most ``limit_bytes`` UTF-8 bytes (with a marker)."""
    encoded = text.encode("utf-8", errors="replace")
    if len(encoded) <= limit_bytes:
        return text
    marker = b"\n...[truncated]"
    keep = max(0, limit_bytes - len(marker))
    return (encoded[:keep] + marker[: limit_bytes - keep]).decode("utf-8", errors="replace")


def _parse_junit(xml_path: Path) -> tuple[int, int, int, int, int, list[str]]:
    """Parse a JUnit XML report -> (passed, failed, errors, skipped, total, failed_names)."""
    root = ET.parse(xml_path).getroot()
    suites = list(root.iter("testsuite"))
    failed = errors = skipped = total = 0
    failed_names: list[str] = []
    seen_cases = False
    for suite in suites:
        for case in suite.iter("testcase"):
            seen_cases = True
            total += 1
            name = case.get("name", "<unknown>")
            if case.find("failure") is not None:
                failed += 1
                failed_names.append(name)
            elif case.find("error") is not None:
                errors += 1
                failed_names.append(name)
            elif case.find("skipped") is not None:
                skipped += 1
    if not seen_cases and suites:
        # Collection error: no testcases but suite-level counters may be set.
        suite = suites[0]
        total = int(suite.get("tests", "0"))
        failed = int(suite.get("failures", "0"))
        errors = int(suite.get("errors", "0"))
        skipped = int(suite.get("skipped", "0"))
    passed = max(0, total - failed - errors - skipped)
    return passed, failed, errors, skipped, total, failed_names


def run_tests_sandboxed(
    package_dir: str,
    test_files: list[str],
    *,
    time_limit_s: int = DEFAULT_TIME_LIMIT_S,
    mem_limit_mb: int = DEFAULT_MEM_LIMIT_MB,
    output_limit_bytes: int = DEFAULT_OUTPUT_LIMIT_BYTES,
) -> TestResult:
    """Run ``pytest test_files`` inside ``package_dir`` under a hardened sandbox.

    The network-blocking ``conftest.py`` is injected into ``package_dir`` (it must
    not already define one). Returns a parsed :class:`TestResult`; never raises for
    test failures, timeouts, or a missing report — those map to a non-passing result.
    """
    work = Path(package_dir)
    conftest = work / "conftest.py"
    if conftest.exists() and "lhmsb sandbox guard" not in conftest.read_text(encoding="utf-8"):
        raise ValueError("package_dir already defines a conftest.py; refusing to overwrite")
    conftest.write_text(_NETWORK_BLOCK_CONFTEST, encoding="utf-8")

    junit = work / _JUNIT_NAME
    junit.unlink(missing_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        *test_files,
        "-q",
        "-p",
        "no:cacheprovider",
        "-o",
        "addopts=",  # neutralise any inherited --cov/--strict addopts
        "--rootdir",
        str(work),
        f"--junitxml={junit}",
    ]
    start = time.monotonic()
    timed_out = False
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(work),
            env=_sandbox_env(str(work)),
            capture_output=True,
            text=True,
            timeout=time_limit_s,
            preexec_fn=_make_limit_setter(mem_limit_mb, time_limit_s),
            check=False,
            start_new_session=True,
        )
        returncode = proc.returncode
        stdout, stderr = proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        returncode = -1
        stdout = _coerce_text(exc.stdout)
        stderr = _coerce_text(exc.stderr)
    duration = time.monotonic() - start

    if junit.exists():
        passed, failed, errors, skipped, total, failed_names = _parse_junit(junit)
        junit.unlink(missing_ok=True)
    else:
        # No report: interpreter killed (rlimit), timeout, or a hard crash.
        passed = skipped = total = 0
        failed = 0
        errors = 1
        failed_names = ["<no-junit-report>"]

    return TestResult(
        returncode=returncode,
        passed=passed,
        failed=failed,
        errors=errors,
        skipped=skipped,
        total=total,
        failed_tests=failed_names,
        timed_out=timed_out,
        stdout=_truncate(stdout, output_limit_bytes),
        stderr=_truncate(stderr, output_limit_bytes),
        duration_s=duration,
    )

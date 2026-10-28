#!/usr/bin/env bash
# Offline runner checks only: this does not run SWE-agent or produce paper results.
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
command -v git >/dev/null
python -c 'import sys; assert sys.version_info >= (3, 11), "Python 3.11+ required"'
python -m pytest -q -o addopts='' tests/realworld
python -m lhmsb.realworld.cli --help
printf '\nFramework preflight passed. No model API was called.\n'
printf 'Formal runs still require a pinned task manifest and a validated SWE-agent adapter.\n'
printf 'Memory, file hiding, and failure checkpoint repair are not implemented by this runner.\n'

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/lib/mem0_common.sh
source "${SCRIPT_DIR}/lib/mem0_common.sh"

REPO_ROOT="$(mem0_repo_root)"
DATA_ROOT="${LHMSB_DATA_ROOT:-/data/lhmsb}"
ENV_FILE="${LHMSB_ENV_FILE:-${REPO_ROOT}/.env}"
RUN_NAME="${LHMSB_SMOKE_RUN_NAME:-smoke}"
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: scripts/run_mem0_smoke.sh [options]

Generate/freeze a four-session fixture and execute the live smoke matrix.

Options:
  --data-root PATH  Persistent benchmark root (default: /data/lhmsb)
  --env-file PATH   Compose environment file (default: .env)
  --run-name NAME   Run directory name (default: smoke)
  --dry-run         Print Compose commands without executing them
  -h, --help        Show this help
EOF
}

while (($#)); do
  case "$1" in
    --data-root)
      mem0_require_value "$1" "${2:-}" || exit 2
      DATA_ROOT="$2"
      shift 2
      ;;
    --env-file)
      mem0_require_value "$1" "${2:-}" || exit 2
      ENV_FILE="$2"
      shift 2
      ;;
    --run-name)
      mem0_require_value "$1" "${2:-}" || exit 2
      RUN_NAME="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      mem0_unknown_argument "$1" || exit $?
      ;;
  esac
done

export LHMSB_DATA_ROOT="${DATA_ROOT}"
export LHMSB_LIVE_QUALIFICATION=1

mem0_compose "${REPO_ROOT}" "${ENV_FILE}" up --detach --wait \
  qdrant embedding reranker
mem0_compose "${REPO_ROOT}" "${ENV_FILE}" run --rm \
  --entrypoint /app/.venv/bin/python worker \
  -m lhmsb.datasets generate-mem0-stateful \
  --seeds 42 --n-episodes 1 --n-sessions 4 \
  --out /data/lhmsb/datasets/software_mem0_smoke_stage
mem0_compose "${REPO_ROOT}" "${ENV_FILE}" run --rm \
  --entrypoint /app/.venv/bin/python worker \
  -m lhmsb.datasets freeze-mem0-stateful \
  --src /data/lhmsb/datasets/software_mem0_smoke_stage \
  --out /data/lhmsb/datasets/software_mem0_smoke
mem0_compose "${REPO_ROOT}" "${ENV_FILE}" run --rm worker \
  smoke --dataset /data/lhmsb/datasets/software_mem0_smoke \
  --config /app/configs/experiments/mem0_qualification.yaml \
  --out "/data/lhmsb/runs/mem0/${RUN_NAME}" \
  --json "/data/lhmsb/runs/mem0/${RUN_NAME}/smoke-status.json"

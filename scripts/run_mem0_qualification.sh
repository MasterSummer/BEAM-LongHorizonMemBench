#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/lib/mem0_common.sh
source "${SCRIPT_DIR}/lib/mem0_common.sh"

REPO_ROOT="$(mem0_repo_root)"
DATA_ROOT="${LHMSB_DATA_ROOT:-/data/lhmsb}"
ENV_FILE="${LHMSB_ENV_FILE:-${REPO_ROOT}/.env}"
RUN_NAME="${LHMSB_RUN_NAME:-qualification}"
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: scripts/run_mem0_qualification.sh [options]

Plan and execute the frozen 16-session Mem0 qualification matrix.

Options:
  --data-root PATH  Persistent benchmark root (default: /data/lhmsb)
  --env-file PATH   Compose environment file (default: .env)
  --run-name NAME   Run directory name (default: qualification)
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
RUN_DIR="/data/lhmsb/runs/mem0/${RUN_NAME}"

if [[ "${DRY_RUN}" == "1" ]]; then
  mem0_print_command mem0_verify_runtime_images "${DATA_ROOT}"
else
  mem0_verify_runtime_images "${DATA_ROOT}"
fi

mem0_compose "${REPO_ROOT}" "${ENV_FILE}" up --detach --wait \
  qdrant embedding reranker
mem0_compose "${REPO_ROOT}" "${ENV_FILE}" run --rm worker \
  plan --dataset /data/lhmsb/datasets/software_mem0_v2 \
  --config /app/configs/experiments/mem0_controlled_zen.yaml \
  --out "${RUN_DIR}"
mem0_compose "${REPO_ROOT}" "${ENV_FILE}" run --rm worker \
  run-matrix --run-dir "${RUN_DIR}" \
  --keep-going \
  --json "${RUN_DIR}/matrix-status.json"
mem0_compose "${REPO_ROOT}" "${ENV_FILE}" run --rm worker \
  validate --report "${RUN_DIR}/report" \
  --json "${RUN_DIR}/validation.json"
